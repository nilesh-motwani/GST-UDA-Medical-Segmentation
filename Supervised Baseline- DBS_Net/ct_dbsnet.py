import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
import cv2
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from collections import defaultdict

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'IMG_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D\Images\CT',
    'MASK_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D\Masks\CT',
    'SAVE_DIR': r'D:\Nilesh\GST-UDA\AMOS22\DBSNet_CT_4Organs', 
    'EPOCHS': 50, 
    'BATCH_SIZE': 32, 
    'LR': 3e-4,
    'NUM_WORKERS': 4, 
    'NUM_CLASSES': 5  # 4 Major Organs + 1 Background
}

os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)

# Generate 5 distinct colors for plotting
COLORS = np.array([
    [0, 0, 0],       # 0: Background (Black)
    [255, 0, 0],     # 1: Spleen (Red)
    [0, 255, 0],     # 2: Right Kidney (Green)
    [0, 0, 255],     # 3: Left Kidney (Blue)
    [255, 255, 0]    # 4: Liver (Yellow)
], dtype=np.uint8)

# Target Mapping from original AMOS22 to our 4-Organ Subset
# Original: 1=Spleen, 2=R.Kidney, 3=L.Kidney, 6=Liver
SUBSET_MAPPING = {1: 1, 2: 2, 3: 3, 6: 4}
SUBSET_NAMES = ["Spleen", "Right Kidney", "Left Kidney", "Liver"]

# ==============================================================================
# ARCHITECTURE (DBSNet)
# ==============================================================================
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class AxialAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super().__init__()
        self.heads = heads
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)

    def forward(self, x, axis='h'):
        B, C, H, W = x.shape
        if axis == 'h': x_perm = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
        else: x_perm = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        attn_out, attn_weights = self.mha(x_perm, x_perm, x_perm)
        if axis == 'h': out = attn_out.view(B, W, H, C).permute(0, 3, 2, 1)
        else: out = attn_out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out, attn_weights

class ESA_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.half_c = in_channels // 2
        self.conv_branch = nn.Sequential(
            nn.Conv2d(self.half_c, self.half_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.half_c),
            nn.ReLU()
        )
        self.axial_h = AxialAttention(self.half_c, heads=8)
        self.axial_w = AxialAttention(self.half_c, heads=8)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_conv, x_attn = torch.split(x, self.half_c, dim=1)
        out_conv = self.conv_branch(x_conv)
        out_attn_h, map_h = self.axial_h(x_attn, axis='h')
        out_attn_w, map_w = self.axial_w(out_attn_h, axis='w')
        out = torch.cat([out_conv, out_attn_w], dim=1)
        out = self.bn(self.conv_out(out))
        return out + x, map_h, map_w

class ETR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.esa = ESA_Block(in_channels)
    def forward(self, x):
        x_proj = self.proj(x)
        out, map_h, map_w = self.esa(x_proj)
        return out, map_h, map_w

class ConcatPlus(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, decoder_feat, encoder_feat):
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            encoder_feat = F.interpolate(encoder_feat, size=decoder_feat.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([decoder_feat + encoder_feat, encoder_feat], dim=1)

class DBSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=5): 
        super().__init__()
        self.enc1 = Conv3x3BNReLU(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = Conv3x3BNReLU(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = Conv3x3BNReLU(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = Conv3x3BNReLU(256, 512)
        self.etr = ETR(512)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d1 = Conv3x3BNReLU(512, 256)
        self.concat_plus1 = ConcatPlus()
        self.fuse1 = Conv3x3BNReLU(512, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d2 = Conv3x3BNReLU(256, 128)
        self.concat_plus2 = ConcatPlus()
        self.fuse2 = Conv3x3BNReLU(256, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_d3 = Conv3x3BNReLU(128, 64)
        self.concat_plus3 = ConcatPlus()
        self.fuse3 = Conv3x3BNReLU(128, 64)
        self.body_extractor = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.out_body = nn.Conv2d(64, out_channels, 1)
        self.out_edge = nn.Conv2d(64, out_channels, 1)
        self.out_final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        x5, map_h, map_w = self.etr(e4)
        d1 = self.conv_d1(self.up1(x5))
        d1_fused = self.fuse1(self.concat_plus1(d1, e3))
        d2 = self.conv_d2(self.up2(d1_fused))
        d2_fused = self.fuse2(self.concat_plus2(d2, e2))
        d3 = self.conv_d3(self.up3(d2_fused))
        F_feat = self.fuse3(self.concat_plus3(d3, e1))
        F_body = self.body_extractor(F_feat)
        F_edge = F_feat - F_body
        return self.out_body(F_body), self.out_edge(F_edge), self.out_final(F_feat), {'F': F_feat, 'F_body': F_body}

# ==============================================================================
# DATASET & LOSS
# ==============================================================================
class AMOS22SubsetDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_np = np.load(self.image_paths[idx])
        img_resized = cv2.resize(img_np[0], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        mask_raw = np.array(Image.open(self.mask_paths[idx]), dtype=np.uint8)
        mask_raw_resized = cv2.resize(mask_raw, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask_mapped = np.zeros_like(mask_raw_resized)
        for orig_idx, new_idx in SUBSET_MAPPING.items():
            mask_mapped[mask_raw_resized == orig_idx] = new_idx

        masks_list, edges, bodies = [], [], []
        
        for c in range(CONFIG['NUM_CLASSES']): 
            m_c = (mask_mapped == c).astype(np.uint8) * 255
            
            if c == 0 or np.max(m_c) == 0: 
                edge_c = np.zeros_like(m_c)
                body_c = m_c
            else:
                edge_c = cv2.Canny(m_c, 100, 200)
                body_c = cv2.erode(m_c, np.ones((3,3), np.uint8), iterations=1)
                
            masks_list.append(m_c / 255.0)
            edges.append(edge_c / 255.0)
            bodies.append(body_c / 255.0)

        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.stack(masks_list, axis=0)).float() 
        edge_tensor = torch.from_numpy(np.stack(edges, axis=0)).float()      
        body_tensor = torch.from_numpy(np.stack(bodies, axis=0)).float()      

        return img_tensor, mask_tensor, edge_tensor, body_tensor

class JointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss() 
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = (2. * inter + 1.0) / (union + 1.0)
        return 1 - dice.mean() 

    def forward(self, pred_body, pred_edge, pred_final, mask, edge, body):
        l_b = self.bce(pred_body, body) + self.dice_loss(pred_body, body)
        l_e = self.bce(pred_edge, edge) + self.dice_loss(pred_edge, edge)
        l_f = self.bce(pred_final, mask) + self.dice_loss(pred_final, mask)
        return l_b + l_e + l_f, l_b.item(), l_e.item(), l_f.item()

# ==============================================================================
# METRICS & VISUALIZATION UTILS
# ==============================================================================
def compute_hd95_asd(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if np.sum(pred) == 0 and np.sum(gt) == 0: return 0.0, 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0: return 373.12, 373.12 
    
    struct = np.ones((3,3,3))
    pred_border = pred ^ binary_erosion(pred, structure=struct)
    gt_border = gt ^ binary_erosion(gt, structure=struct)
    
    pred_pts = np.argwhere(pred_border)
    gt_pts = np.argwhere(gt_border)
    
    if len(pred_pts) == 0 or len(gt_pts) == 0: return 373.12, 373.12
        
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)
    
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    hd95 = np.percentile(all_dists, 95)
    asd = (np.sum(dist_pred_to_gt) + np.sum(dist_gt_to_pred)) / (len(pred_pts) + len(gt_pts))
    
    return hd95, asd

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.gradients = None; self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def __call__(self, x):
        self.gradients = None; self.activations = None
        output, _, _, _ = self.model(x)
        self.model.zero_grad()
        output.sum().backward() 
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]): self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        return ((F.relu(heatmap) - F.relu(heatmap).min()) / (F.relu(heatmap).max() - F.relu(heatmap).min() + 1e-8)).detach().cpu().numpy()

class BioMedicalValidator:
    def __init__(self, model, device, save_dir):
        self.model = model; self.device = device; self.save_dir = save_dir

    def plot_spectral_proof(self, sample_img_tensor):
        hooks = {}
        def get_activation(name):
            def hook(model, input, output): hooks[name] = output.detach()
            return hook
        h1 = self.model.fuse3.register_forward_hook(get_activation('Total_Feature'))
        h2 = self.model.body_extractor.register_forward_hook(get_activation('Body_Feature'))

        with torch.no_grad(): _ = self.model(sample_img_tensor.unsqueeze(0).to(self.device))

        f_total = hooks['Total_Feature'][0, 0].cpu().numpy()
        f_body = hooks['Body_Feature'][0, 0].cpu().numpy()
        f_edge = f_total - f_body 
        h1.remove(); h2.remove()

        def calc_fft(img):
            f = np.fft.fft2(img); fshift = np.fft.fftshift(f)
            return 20*np.log(np.abs(fshift) + 1e-8)

        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax[0,0].imshow(f_total, cmap='gray'); ax[0,0].set_title('Feature: Total')
        ax[0,1].imshow(f_body, cmap='gray'); ax[0,1].set_title('Feature: Body (Low Freq)')
        ax[0,2].imshow(f_edge, cmap='gray'); ax[0,2].set_title('Feature: Edge (High Freq)')
        ax[1,0].imshow(calc_fft(f_total), cmap='inferno'); ax[1,0].set_title('Spectrum: Total')
        ax[1,1].imshow(calc_fft(f_body), cmap='inferno'); ax[1,1].set_title('Spectrum: Body (Center)')
        ax[1,2].imshow(calc_fft(f_edge), cmap='inferno'); ax[1,2].set_title('Spectrum: Edge (Spread)')
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/Proof_Spectral_Analysis.png")

    def plot_volumetric_consistency(self, sorted_loader):
        vol_stack = []
        with torch.no_grad():
            for img, _, _, _ in sorted_loader:
                _, _, p_final, _ = self.model(img.to(self.device))
                pred = torch.argmax(p_final, dim=1).cpu().numpy().squeeze() 
                vol_stack.append(pred)
                if len(vol_stack) > 150: break 

        vol = np.array(vol_stack) 
        coronal = vol[:, vol.shape[1]//2, :] 
        
        coronal_rgb = np.zeros((coronal.shape[0], coronal.shape[1], 3), dtype=np.uint8)
        for c in range(1, CONFIG['NUM_CLASSES']):
            coronal_rgb[coronal == c] = COLORS[c]

        coronal_vis = cv2.resize(coronal_rgb, (256, 256*3), interpolation=cv2.INTER_NEAREST)

        plt.figure(figsize=(4, 10))
        plt.imshow(coronal_vis)
        plt.title("Reconstructed Coronal View\n(4 Major Organs - CT)")
        plt.axis('off')
        plt.savefig(f"{self.save_dir}/Proof_Coronal_Consistency.png")

def save_individual_images(model, dataset, device, epoch, save_dir):
    idx = np.random.randint(0, len(dataset))
    img, mask, _, _ = dataset[idx] 

    model.eval()
    with torch.no_grad():
        _, _, p_final, _ = model(img.unsqueeze(0).to(device))

    cam = GradCAM(model, model.fuse3.block[0])
    heatmap = cam(img.unsqueeze(0).to(device))

    img_vis = img[0].numpy()
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8) * 255
    img_vis = img_vis.astype(np.uint8)

    gt_np = torch.argmax(mask, dim=0).numpy() 
    pred_final = torch.argmax(p_final[0], dim=0).cpu().numpy() 

    def create_overlay(bg_img, mask_array):
        rgb = np.zeros((bg_img.shape[0], bg_img.shape[1], 3), dtype=np.uint8)
        for c in range(1, CONFIG['NUM_CLASSES']):
            rgb[mask_array == c] = COLORS[c]
        return cv2.addWeighted(np.stack([bg_img]*3, axis=-1), 0.6, rgb, 0.4, 0)

    gt_overlay = create_overlay(img_vis, gt_np)
    pred_overlay = create_overlay(img_vis, pred_final)

    heatmap_c = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB) / 255.0
    cam_overlay = 0.6 * np.stack([img_vis/255.0]*3, -1) + 0.4 * cv2.resize(heatmap_c, (256,256))

    sub_dir = f"{save_dir}/Individual_Epoch_{epoch}"
    os.makedirs(sub_dir, exist_ok=True)

    def save(name, data, cmap=None):
        plt.figure(figsize=(5,5)); plt.imshow(data, cmap=cmap); plt.axis('off')
        plt.savefig(f"{sub_dir}/{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    save("1_Input_CT", img_vis, cmap='gray')
    save("2_GT_Overlay", gt_overlay)
    save("3_Pred_Overlay", pred_overlay)
    save("4_GradCAM", cam_overlay)

def calculate_full_3d_metrics(model, dataset, device):
    model.eval()
    patient_preds = defaultdict(list)
    patient_gts = defaultdict(list)

    print("\nGathering Full 3D Volumes for CT...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img, mask, _, _ = dataset[i]
            patient_id = os.path.basename(dataset.image_paths[i]).split('_slice')[0]

            img = img.unsqueeze(0).to(device)
            _, _, p_final, _ = model(img)
            
            pred = torch.argmax(p_final[0], dim=0).cpu().numpy() 
            gt = torch.argmax(mask, dim=0).numpy() 
            
            patient_preds[patient_id].append(pred)
            patient_gts[patient_id].append(gt)

    all_dices, all_hd95s, all_asds = [], [], []

    print("\nComputing 3D Dice, HD95, and ASD for subset organs...")
    for pid in tqdm(patient_preds.keys()):
        vol_pred = np.stack(patient_preds[pid], axis=0) 
        vol_gt = np.stack(patient_gts[pid], axis=0) 
        
        pt_dice, pt_hd95, pt_asd = [], [], []
        
        for c in range(1, CONFIG['NUM_CLASSES']): 
            p_c = (vol_pred == c)
            g_c = (vol_gt == c)
            
            inter = np.sum(p_c * g_c)
            union = np.sum(p_c) + np.sum(g_c)
            pt_dice.append((2 * inter + 1e-8) / (union + 1e-8))
            
            hd, asd = compute_hd95_asd(p_c, g_c)
            pt_hd95.append(hd)
            pt_asd.append(asd)
            
        all_dices.append(pt_dice)
        all_hd95s.append(pt_hd95)
        all_asds.append(pt_asd)

    mean_dices = np.mean(np.array(all_dices), axis=0)
    mean_hd95s = np.mean(np.array(all_hd95s), axis=0)
    mean_asds = np.mean(np.array(all_asds), axis=0)

    return mean_dices.mean(), mean_dices, mean_hd95s.mean(), mean_hd95s, mean_asds.mean(), mean_asds

# ==============================================================================
# MAIN EXECUTION (Training + 3D Evaluation)
# ==============================================================================
if __name__ == '__main__':
    if not os.path.exists(CONFIG['IMG_DIR']) or not os.path.exists(CONFIG['MASK_DIR']):
        raise FileNotFoundError(f"Check your paths! Cannot find {CONFIG['IMG_DIR']} or {CONFIG['MASK_DIR']}")

    print(f"\n--- Initializing AMOS22 CT Pipeline (4-Organ Subset) ---")
    
    all_files = sorted(os.listdir(CONFIG['IMG_DIR']))
    case_ids = sorted(list(set([f.split('_slice')[0] for f in all_files if f.endswith('.npy')])))
    print(f"Found {len(case_ids)} unique CT patients.")

    train_cases, temp_cases = train_test_split(case_ids, test_size=0.2, random_state=42)
    val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)

    def filter_files(case_list, img_dir, mask_dir):
        img_paths, mask_paths = [], []
        for f in sorted(os.listdir(img_dir)):
            if not f.endswith('.npy'): continue
            if any(c in f for c in case_list):
                img_path = os.path.join(img_dir, f)
                mask_path = os.path.join(mask_dir, f.replace('.npy', '.png'))
                if os.path.exists(mask_path):
                    img_paths.append(img_path)
                    mask_paths.append(mask_path)
        return img_paths, mask_paths

    train_i, train_m = filter_files(train_cases, CONFIG['IMG_DIR'], CONFIG['MASK_DIR'])
    val_i, val_m = filter_files(val_cases, CONFIG['IMG_DIR'], CONFIG['MASK_DIR'])
    test_i, test_m = filter_files(test_cases, CONFIG['IMG_DIR'], CONFIG['MASK_DIR'])

    train_ds = AMOS22SubsetDataset(train_i, train_m)
    val_ds = AMOS22SubsetDataset(val_i, val_m)
    test_ds = AMOS22SubsetDataset(test_i, test_m)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DBSNet(in_channels=1, out_channels=CONFIG['NUM_CLASSES']).to(device) 
    criterion = JointLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu') 

    best_dice = 0.0

    print("Starting Training...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for images, masks, edges, bodies in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)
            edges, bodies = edges.to(device), bodies.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                p_body, p_edge, p_final, _ = model(images)
                loss, _, _, _ = criterion(p_body, p_edge, p_final, masks, edges, bodies)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        model.eval()
        dice_sum = 0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                p_final = model(images.to(device))[2]
                pred = (torch.sigmoid(p_final) > 0.5).float()
                inter = (pred * masks.to(device)).sum(dim=(0, 2, 3))
                union = pred.sum(dim=(0, 2, 3)) + masks.to(device).sum(dim=(0, 2, 3))
                dice_sum += ((2 * inter) / (union + 1e-8)).mean().item()
        
        current_dice = dice_sum / len(val_loader)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | 2D Val Dice: {current_dice:.4f}")

        if current_dice > best_dice:
            best_dice = current_dice
            torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_DIR'], "best_model_CT_4Organs.pth"))
            print("  🌟 Best CT model saved!")

        if (epoch+1) % 5 == 0:
            save_individual_images(model, val_ds, device, epoch+1, CONFIG['SAVE_DIR'])

    # --- FINAL 3D EVALUATION ---
    print(f"\nLoading Best Model for Final Volumetric Evaluation...")
    model.load_state_dict(torch.load(os.path.join(CONFIG['SAVE_DIR'], "best_model_CT_4Organs.pth"), weights_only=True))

    vol_avg_dice, d_cls, vol_avg_hd, hd_cls, vol_avg_asd, asd_cls = calculate_full_3d_metrics(model, test_ds, device)
    
    print("\n" + "="*60)
    print("FINAL 3D METRICS (4 Major Organs - CT)")
    print("="*60)
    print(f"Overall Average 3D Dice: {vol_avg_dice*100:.2f}%")
    print(f"Overall Average 3D HD95: {vol_avg_hd:.2f} mm")
    print(f"Overall Average 3D ASD:  {vol_avg_asd:.2f} mm")
    print("="*60 + "\n")
    
    for i in range(4):
        print(f"{SUBSET_NAMES[i]:<15} -> Dice: {d_cls[i]*100:>5.2f}% | HD95: {hd_cls[i]:>6.2f} | ASD: {asd_cls[i]:>5.2f}")

    print("\n=== Generating Final Reviewer Proofs ===")
    validator = BioMedicalValidator(model, device, CONFIG['SAVE_DIR'])
    sample_img = next(iter(test_loader))[0][0]
    validator.plot_spectral_proof(sample_img)
    validator.plot_volumetric_consistency(test_loader)
    save_individual_images(model, test_ds, device, "FINAL", CONFIG['SAVE_DIR'])

    print(f"\n✅ All Proofs and Models Saved to {CONFIG['SAVE_DIR']}")