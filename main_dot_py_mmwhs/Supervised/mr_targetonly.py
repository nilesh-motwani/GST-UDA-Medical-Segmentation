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
matplotlib.use('Agg') # Forces headless rendering for remote/background execution
import matplotlib.pyplot as plt
import cv2
from scipy import stats
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

# Set seeds for absolute reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. CONFIGURATION (Updated for MR)
# ==============================================================================
CONFIG = {
    'IMG_DIR': r'D:\Nilesh\GST-UDA\MMWHS\MMWHS_Data_2D\Images\MR',
    'MASK_DIR': r'D:\Nilesh\GST-UDA\MMWHS\MMWHS_Data_2D\Masks\MR',
    'SAVE_DIR': r'D:\Nilesh\GST-UDA\MMWHS\DBSNet_MR_Baseline', 
    'EPOCHS': 200,     
    'BATCH_SIZE': 32,  
    'LR': 3e-4,
    'NUM_WORKERS': 4,
    'IMG_SIZE': 256
}

os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)

# ==============================================================================
# 2. DBS-NET ARCHITECTURE (1-IN, 4-OUT)
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

class TBNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4): 
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
# 3. DATA HANDLING (No Resizing) & LOSS
# ==============================================================================
class TBNetDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self): 
        return len(self.image_paths)

    def pad_or_crop(self, image, is_mask=False):
        """Dynamically pads or center-crops the image to img_size x img_size."""
        h, w = image.shape
        
        # 1. Pad if smaller
        pad_h = max(0, self.img_size - h)
        pad_w = max(0, self.img_size - w)
        if pad_h > 0 or pad_w > 0:
            pad_val = 0 if is_mask else np.min(image)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), 
                           mode='constant', constant_values=pad_val)
            h, w = image.shape

        # 2. Center crop if larger
        start_h = (h - self.img_size) // 2
        start_w = (w - self.img_size) // 2
        return image[start_h:start_h + self.img_size, start_w:start_w + self.img_size]

    def __getitem__(self, idx):
        # Load Single-Channel Image
        img_np = np.load(self.image_paths[idx])
        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)

        # Load Integer Mask
        mask_pil = Image.open(self.mask_paths[idx])
        mask_np = np.array(mask_pil)

        # Apply spatial pad/crop to 256x256
        img_cropped = self.pad_or_crop(img_np, is_mask=False)
        mask_cropped = self.pad_or_crop(mask_np, is_mask=True)

        # Paper Mapping Order: AA, LAC, LVC, MYO
        mapping_order = [4, 2, 3, 1] 
        
        mask_channels = []
        for target_int in mapping_order: 
            mask_channels.append((mask_cropped == target_int).astype(np.float32))
        mask_binary = np.stack(mask_channels, axis=-1)

        edges, bodies = [], []
        for c in range(4): 
            m_c = (mask_binary[:, :, c] * 255).astype(np.uint8)
            edge_c = cv2.Canny(m_c, 100, 200) / 255.0
            body_c = cv2.erode(m_c, np.ones((3,3), np.uint8), iterations=1) / 255.0
            edges.append(edge_c)
            bodies.append(body_c)

        img_tensor = torch.from_numpy(img_cropped).unsqueeze(0).float() 
        mask_tensor = torch.from_numpy(mask_binary).permute(2, 0, 1).float() 
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
# 4. VOLUMETRIC METRICS (HD95 & 3D Dice)
# ==============================================================================
def compute_hd95(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 373.12 
    
    struct = np.ones((3,3,3))
    pred_border = pred ^ binary_erosion(pred, structure=struct)
    gt_border = gt ^ binary_erosion(gt, structure=struct)
    
    pred_pts = np.argwhere(pred_border)
    gt_pts = np.argwhere(gt_border)
    
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 373.12
        
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)
    
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return np.percentile(all_dists, 95)

def calculate_full_3d_metrics(model, dataset, device):
    model.eval()
    patient_preds = defaultdict(list)
    patient_gts = defaultdict(list)

    print("\nGathering Full 3D Volumes for MR...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img, mask, _, _ = dataset[i]
            # Assumes format MMWHS_MR_1001_slice001.npy
            patient_id = os.path.basename(dataset.image_paths[i]).split('_slice')[0]

            img = img.unsqueeze(0).to(device)
            _, _, p_final, _ = model(img)
            pred = (torch.sigmoid(p_final) > 0.5).float().cpu().numpy()[0] 
            
            patient_preds[patient_id].append(pred)
            patient_gts[patient_id].append(mask.numpy())

    all_dices, all_hd95s = [], []

    print("\nComputing 3D Dice and HD95...")
    for pid in tqdm(patient_preds.keys()):
        # Stack slices back into a 3D Volume
        vol_pred = np.stack(patient_preds[pid], axis=0).transpose(1, 0, 2, 3)
        vol_gt = np.stack(patient_gts[pid], axis=0).transpose(1, 0, 2, 3)
        
        pt_dice = []
        pt_hd95 = []
        
        for c in range(4): # 4 structures
            p_c = vol_pred[c]
            g_c = vol_gt[c]
            
            inter = np.sum(p_c * g_c)
            union = np.sum(p_c) + np.sum(g_c)
            pt_dice.append((2 * inter + 1e-8) / (union + 1e-8))
            
            pt_hd95.append(compute_hd95(p_c, g_c))
            
        all_dices.append(pt_dice)
        all_hd95s.append(pt_hd95)

    mean_dices = np.mean(np.array(all_dices), axis=0)
    mean_hd95s = np.mean(np.array(all_hd95s), axis=0)

    return mean_dices.mean(), mean_dices, mean_hd95s.mean(), mean_hd95s

# ==============================================================================
# 5. QUALITATIVE VISUALIZATION & VALIDATION
# ==============================================================================
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
        plt.tight_layout(); plt.savefig(f"{self.save_dir}/Proof_Spectral_Analysis_MR.png")
        print("✅ Spectral Proof Generated for MR.")

def save_individual_images(model, dataset, device, epoch, save_dir):
    idx = np.random.randint(0, len(dataset))
    img, mask, edge, body = dataset[idx]

    model.eval()
    with torch.no_grad():
        p_body, p_edge, p_final, _ = model(img.unsqueeze(0).to(device))

    cam = GradCAM(model, model.fuse3.block[0])
    heatmap = cam(img.unsqueeze(0).to(device))

    img_bg = img.squeeze().numpy()
    img_vis = img_bg - img_bg.min()
    img_vis = (img_vis / (img_vis.max() + 1e-8) * 255).astype(np.uint8)

    gt_np = mask.numpy() 
    pred_final = (torch.sigmoid(p_final) > 0.5).float().squeeze().cpu().numpy() 

    def create_overlay(bg_img, mask_array):
        rgb = np.zeros((bg_img.shape[0], bg_img.shape[1], 3), dtype=np.uint8)
        for i in range(3): rgb[:,:,i] = bg_img
        rgb[mask_array[0] == 1] = [255, 255, 0] # AA
        rgb[mask_array[1] == 1] = [0, 255, 0]   # LAC
        rgb[mask_array[2] == 1] = [0, 0, 255]   # LVC
        rgb[mask_array[3] == 1] = [255, 0, 0]   # MYO
        return cv2.addWeighted(np.stack([bg_img]*3, axis=-1), 0.5, rgb, 0.5, 0)

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

    save("1_Input_MR", img_vis, cmap='gray')
    save("2_GT_Overlay", gt_overlay)
    save("3_Pred_Overlay", pred_overlay)
    save("4_GradCAM", cam_overlay)

# ==============================================================================
# 6. TRAINING LOOP
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_meter = {'total': 0.0, 'body': 0.0, 'edge': 0.0, 'final': 0.0}
    for images, masks, edges, bodies in tqdm(loader, leave=False, desc="Train"):
        images, masks = images.to(device), masks.to(device)
        edges, bodies = edges.to(device), bodies.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            p_body, p_edge, p_final, _ = model(images)
            loss, l_b, l_e, l_f = criterion(p_body, p_edge, p_final, masks, edges, bodies)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_meter['total'] += loss.item(); loss_meter['body'] += l_b
        loss_meter['edge'] += l_e; loss_meter['final'] += l_f
    return {k: v / len(loader) for k, v in loss_meter.items()}

def validate_metrics(model, loader, device):
    model.eval()
    dice_sums = np.zeros(4) 
    with torch.no_grad():
        for images, masks, edges, bodies in loader:
            images, masks = images.to(device), masks.to(device)
            _, _, p_final, _ = model(images)
            pred = (torch.sigmoid(p_final) > 0.5).float()
            
            inter = (pred * masks).sum(dim=(0, 2, 3)).cpu().numpy()
            union = (pred.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))).cpu().numpy()
            dice_sums += (2 * inter) / (union + 1e-8)
            
    avg_dices = dice_sums / len(loader)
    return 0.0, {
        'Dice_AA': avg_dices[0], 
        'Dice_LAC': avg_dices[1], 
        'Dice_LVC': avg_dices[2], 
        'Dice_MYO': avg_dices[3],
        'Dice_Avg': avg_dices.mean()
    }

if __name__ == '__main__':
    if not os.path.exists(CONFIG['IMG_DIR']) or not os.path.exists(CONFIG['MASK_DIR']):
        raise FileNotFoundError(f"Check your paths! Cannot find {CONFIG['IMG_DIR']}")

    print(f"\n--- Initializing FULL MR Pipeline for MMWHS ---")
    
    all_files = sorted(os.listdir(CONFIG['IMG_DIR']))
    case_ids = sorted(list(set([f.split('_slice')[0] for f in all_files if f.endswith('.npy')])))
    print(f"Found {len(case_ids)} unique MR patients.")

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

    train_ds = TBNetDataset(train_i, train_m, img_size=CONFIG['IMG_SIZE'])
    val_ds = TBNetDataset(val_i, val_m, img_size=CONFIG['IMG_SIZE'])
    test_ds = TBNetDataset(test_i, test_m, img_size=CONFIG['IMG_SIZE'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CONFIG['NUM_WORKERS'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TBNet(in_channels=1, out_channels=4).to(device)
    criterion = JointLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)
    scaler = GradScaler(device.type) 

    history = {'train_loss': [], 'lr': [], 'Dice_AA': [], 'Dice_LAC': [], 'Dice_LVC': [], 'Dice_MYO': [], 'Dice_Avg': []}
    best_dice = 0.0

    print(f"Starting Training for {CONFIG['EPOCHS']} Epochs...")
    for epoch in range(CONFIG['EPOCHS']):
        loss_dict = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        _, val_metrics = validate_metrics(model, val_loader, device)
        current_dice = val_metrics['Dice_Avg']
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(loss_dict['total'])
        history['lr'].append(current_lr)
        history['Dice_AA'].append(val_metrics['Dice_AA'])
        history['Dice_LAC'].append(val_metrics['Dice_LAC'])
        history['Dice_LVC'].append(val_metrics['Dice_LVC'])
        history['Dice_MYO'].append(val_metrics['Dice_MYO'])
        history['Dice_Avg'].append(current_dice)

        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | LR: {current_lr:.2e} | Total Loss: {loss_dict['total']:.4f} | Avg Dice: {current_dice:.4f} (AA: {val_metrics['Dice_AA']:.4f}, LAC: {val_metrics['Dice_LAC']:.4f}, LVC: {val_metrics['Dice_LVC']:.4f}, MYO: {val_metrics['Dice_MYO']:.4f})")

        if current_dice > best_dice:
            best_dice = current_dice
            save_path = os.path.join(CONFIG['SAVE_DIR'], f"best_model_MMWHS_MR.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    🌟 New Best 2D Avg Dice! Model saved to {save_path}")

        if (epoch+1) % 10 == 0:
            save_individual_images(model, val_ds, device, epoch+1, CONFIG['SAVE_DIR'])

    # --- FINAL EVALUATION ---
    print(f"\nLoading Best Model for Final Volumetric Evaluation...")
    model.load_state_dict(torch.load(os.path.join(CONFIG['SAVE_DIR'], f"best_model_MMWHS_MR.pth"), weights_only=True))

    vol_avg_dice, d_cls, vol_avg_hd, hd_cls = calculate_full_3d_metrics(model, test_ds, device)
    
    print("\n" + "="*50)
    print("FINAL 3D METRICS (Matched to MATrans Paper)")
    print("="*50)
    print(f"DICE (%) -> AA: {d_cls[0]*100:.2f} | LAC: {d_cls[1]*100:.2f} | LVC: {d_cls[2]*100:.2f} | MYO: {d_cls[3]*100:.2f} | AVG: {vol_avg_dice*100:.2f}")
    print(f"HD95 (mm) -> AA: {hd_cls[0]:.2f} | LAC: {hd_cls[1]:.2f} | LVC: {hd_cls[2]:.2f} | MYO: {hd_cls[3]:.2f} | AVG: {vol_avg_hd:.2f}")
    print("="*50 + "\n")

    print("\n=== Generating Reviewer Proofs ===")
    validator = BioMedicalValidator(model, device, CONFIG['SAVE_DIR'])

    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label='Total Loss', color='black')
    plt.plot(history['Dice_AA'], label='AA Dice', color='yellow')
    plt.plot(history['Dice_LAC'], label='LAC Dice', color='green')
    plt.plot(history['Dice_LVC'], label='LVC Dice', color='blue')
    plt.plot(history['Dice_MYO'], label='MYO Dice', color='red')
    plt.xlabel('Epochs'); plt.ylabel('Score/Loss'); plt.legend(); plt.grid(True)
    plt.title("MR Training Curves")
    plt.savefig(f"{CONFIG['SAVE_DIR']}/Proof_Training_Curves_MR.png")

    sample_img = next(iter(test_loader))[0][0]
    validator.plot_spectral_proof(sample_img)
    save_individual_images(model, test_ds, device, "FINAL", CONFIG['SAVE_DIR'])

    print(f"\n✅ All Proofs Saved to {CONFIG['SAVE_DIR']}")