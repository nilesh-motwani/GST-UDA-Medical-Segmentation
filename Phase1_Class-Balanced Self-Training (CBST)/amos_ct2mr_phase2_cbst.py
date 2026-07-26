import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm.auto import tqdm

# Set seeds for absolute reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. AMOS22 UDA CONFIGURATION (CT -> MR)
# ==============================================================================
CONFIG = {
    'SOURCE_IMG_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Images\CT',
    'SOURCE_MASK_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Masks\CT',
    'TARGET_IMG_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Images\MR',
    'TARGET_MASK_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Masks\MR',
    
    # 🌟 LOAD AMOS CT SUPERVISED BASELINE (5-Class)
    'PRETRAINED_SEGMENTOR': r'D:\Nilesh\AMOS22\Supervised_CT_Baseline\best_model_AMOS_CT.pth',
    'SAVE_DIR': r'D:\Nilesh\AMOS22\UDA_CT2MR_Phase1', 
    
    'EPOCHS': 100,               
    'BATCH_SIZE': 16,             
    'LR_GAN': 2e-4,              
    'LR_SEG': 1e-4,              
    'NUM_WORKERS': 8,
    'IMG_SIZE': 256,
    
    'NUM_CLASSES': 5, # 0=Background, 1=Liver, 2=RKid, 3=LKid, 4=Spleen
    
    'LAMBDA_STRUCT': 10.0,       
    'LAMBDA_ADV_FEAT': 0.01,     
    'LAMBDA_PSEUDO': 0.5,        
    
    'FDA_BETA': 0.05,
    'CONFIDENCE_THRESH': 0.70   # MR Target -> Softer boundaries, relaxed threshold
}

os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
COLORS = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0]], dtype=np.uint8)

# ==============================================================================
# 2. FOURIER DOMAIN ADAPTATION (FDA)
# ==============================================================================
def extract_amp_spectrum(img):
    fft = torch.fft.fft2(img)
    fft = torch.fft.fftshift(fft)
    amp = torch.abs(fft)
    pha = torch.angle(fft)
    return amp, pha

def FDA_source_to_target(src_img, trg_img, L=0.05):
    amp_src, pha_src = extract_amp_spectrum(src_img)
    amp_trg, _ = extract_amp_spectrum(trg_img)
    
    _, h, w = src_img.shape
    b = int(np.floor(min(h, w) * L))
    center_h, center_w = h // 2, w // 2
    
    amp_src_cloned = amp_src.clone()
    amp_src_cloned[:, center_h-b:center_h+b, center_w-b:center_w+b] = amp_trg[:, center_h-b:center_h+b, center_w-b:center_w+b]
        
    fft_src_ = amp_src_cloned * torch.exp(1j * pha_src)
    fft_src_ = torch.fft.ifftshift(fft_src_)
    src_in_trg = torch.fft.ifft2(fft_src_)
    src_in_trg = torch.real(src_in_trg)
    
    return torch.clamp(src_in_trg, -1.0, 1.0)

# ==============================================================================
# 3. GENERATOR & DISCRIMINATORS
# ==============================================================================
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FeatureDiscriminator(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        )
    def forward(self, x, alpha=1.0): return self.net(GradientReversalLayer.apply(x, alpha))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_blocks=9):
        super().__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features = 64; out_features = 128
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features; out_features *= 2
        for _ in range(n_blocks): model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features; out_features //= 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

class ImageDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            if normalize: layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False), *discriminator_block(64, 128),
            *discriminator_block(128, 256), *discriminator_block(256, 512), spectral_norm(nn.Conv2d(512, 1, 3, padding=1))
        )
    def forward(self, x): return self.model(x)

# ==============================================================================
# 4. UNIFIED DBS-NET ARCHITECTURE (5-Channel Softmax)
# ==============================================================================
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class AxialAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
    def forward(self, x, axis='h'):
        B, C, H, W = x.shape
        if axis == 'h': x_perm = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
        else: x_perm = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        attn_out, _ = self.mha(x_perm, x_perm, x_perm)
        if axis == 'h': out = attn_out.view(B, W, H, C).permute(0, 3, 2, 1)
        else: out = attn_out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out, None

class ESA_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.half_c = in_channels // 2
        self.conv_branch = nn.Sequential(nn.Conv2d(self.half_c, self.half_c, kernel_size=3, padding=1), nn.BatchNorm2d(self.half_c), nn.ReLU())
        self.axial_h = AxialAttention(self.half_c, heads=8)
        self.axial_w = AxialAttention(self.half_c, heads=8)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        x_conv, x_attn = torch.split(x, self.half_c, dim=1)
        out_conv = self.conv_branch(x_conv)
        out_attn_h, _ = self.axial_h(x_attn, axis='h')
        out_attn_w, _ = self.axial_w(out_attn_h, axis='w')
        out = torch.cat([out_conv, out_attn_w], dim=1)
        return self.bn(self.conv_out(out)) + x, None, None

class ETR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.esa = ESA_Block(in_channels)
    def forward(self, x): return self.esa(self.proj(x))[0], None, None

class ConcatPlus(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, dec_feat, enc_feat):
        if dec_feat.shape[2:] != enc_feat.shape[2:]:
            enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([dec_feat + enc_feat, enc_feat], dim=1)

class DBSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=5): 
        super().__init__()
        self.enc1 = Conv3x3BNReLU(in_channels, 64); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = Conv3x3BNReLU(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = Conv3x3BNReLU(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = Conv3x3BNReLU(256, 512)
        self.etr = ETR(512)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True); self.conv_d1 = Conv3x3BNReLU(512, 256); self.cp1 = ConcatPlus(); self.fuse1 = Conv3x3BNReLU(512, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True); self.conv_d2 = Conv3x3BNReLU(256, 128); self.cp2 = ConcatPlus(); self.fuse2 = Conv3x3BNReLU(256, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True); self.conv_d3 = Conv3x3BNReLU(128, 64); self.cp3 = ConcatPlus(); self.fuse3 = Conv3x3BNReLU(128, 64)
        self.body_extractor = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.out_body = nn.Conv2d(64, out_channels, 1)
        self.out_edge = nn.Conv2d(64, out_channels, 1)
        self.out_final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.enc4(self.pool3(e3)) 
        x5, _, _ = self.etr(bottleneck)
        d1_fused = self.fuse1(self.cp1(self.conv_d1(self.up1(x5)), e3))
        d2_fused = self.fuse2(self.cp2(self.conv_d2(self.up2(d1_fused)), e2))
        F_feat = self.fuse3(self.cp3(self.conv_d3(self.up3(d2_fused)), e1))
        F_body = self.body_extractor(F_feat)
        F_edge = F_feat - F_body
        
        return self.out_body(F_body), self.out_edge(F_edge), self.out_final(F_feat), bottleneck

# ==============================================================================
# 5. SUPERVISED LOSS (SOFTMAX)
# ==============================================================================
class StructureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    def forward(self, real_img, fake_img):
        dx_r, dy_r = real_img[:, :, :, 1:] - real_img[:, :, :, :-1], real_img[:, :, 1:, :] - real_img[:, :, :-1, :]
        dx_f, dy_f = fake_img[:, :, :, 1:] - fake_img[:, :, :, :-1], fake_img[:, :, 1:, :] - fake_img[:, :, :-1, :]
        return self.l1(dx_r, dx_f) + self.l1(dy_r, dy_f)

class SoftmaxWeightedLoss(nn.Module):
    def __init__(self, class_weights=[0.1, 1.0, 5.0, 5.0, 2.0]):
        super().__init__()
        self.class_weights = class_weights
        
    def dice_loss(self, pred_logits, target_onehot):
        weights = torch.tensor(self.class_weights, device=pred_logits.device).view(1, 5, 1, 1)
        pred_softmax = torch.softmax(pred_logits, dim=1)
        inter = (pred_softmax * target_onehot * weights).sum(dim=(0, 2, 3))
        union = (pred_softmax * weights).sum(dim=(0, 2, 3)) + (target_onehot * weights).sum(dim=(0, 2, 3))
        return 1 - ((2. * inter + 1e-5) / (union + 1e-5)).mean()

    def forward(self, pred_body, pred_edge, pred_final, mask_long, mask_onehot, edge, body):
        ce_weights = torch.tensor(self.class_weights, device=pred_final.device)
        ce = nn.CrossEntropyLoss(weight=ce_weights)(pred_final, mask_long)
        l_f_dice = self.dice_loss(pred_final, mask_onehot)
        l_b = self.dice_loss(pred_body, body)
        l_e = self.dice_loss(pred_edge, edge)
        return ce + l_f_dice + (0.5 * l_b) + (0.5 * l_e)

# ==============================================================================
# 6. UNPAIRED DATASET
# ==============================================================================
class UnpairedUDADataset(Dataset):
    def __init__(self, src_img_paths, src_mask_paths, trg_img_paths, trg_mask_paths=None, img_size=256):
        self.src_imgs, self.src_masks = src_img_paths, src_mask_paths
        self.trg_imgs, self.trg_masks = trg_img_paths, trg_mask_paths 
        self.img_size, self.trg_len = img_size, len(trg_img_paths)

    def __len__(self): return max(len(self.src_imgs), self.trg_len) if self.trg_masks is None else len(self.trg_imgs)

    def pad_or_crop(self, image, is_mask=False):
        h, w = image.shape
        pad_h, pad_w = max(0, self.img_size - h), max(0, self.img_size - w)
        if pad_h > 0 or pad_w > 0:
            pad_val = 0 if is_mask else np.min(image)
            image = np.pad(image, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant', constant_values=pad_val)
        h, w = image.shape
        start_h, start_w = (h - self.img_size) // 2, (w - self.img_size) // 2
        return image[start_h:start_h + self.img_size, start_w:start_w + self.img_size]

    def process_item(self, img_path, mask_path=None):
        img_np = np.load(img_path).squeeze()
        img_cropped = self.pad_or_crop(img_np, is_mask=False)
        img_tensor = torch.from_numpy(img_cropped).unsqueeze(0).float()
        
        # Scale to [-1, 1] for GAN
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8) * 2.0 - 1.0

        if mask_path is None: return img_tensor

        mask_np = np.array(Image.open(mask_path))
        mask_cropped = self.pad_or_crop(mask_np, is_mask=True)

        mask_binary = np.zeros((self.img_size, self.img_size, 5), dtype=np.float32)
        for c in range(5): mask_binary[:, :, c] = (mask_cropped == c).astype(np.float32)

        edges, bodies = [], []
        for c in range(5): 
            m_c = (mask_binary[:, :, c] * 255).astype(np.uint8)
            edges.append(cv2.Canny(m_c, 100, 200) / 255.0)
            bodies.append(cv2.erode(m_c, np.ones((3,3), np.uint8), iterations=1) / 255.0)

        mask_long = torch.from_numpy(mask_cropped).long()
        return img_tensor, mask_long, torch.from_numpy(mask_binary).permute(2, 0, 1).float(), \
               torch.from_numpy(np.stack(edges, axis=0)).float(), torch.from_numpy(np.stack(bodies, axis=0)).float()

    def __getitem__(self, idx):
        if self.trg_masks is not None:
            t_img, t_mask_long, _, _, _ = self.process_item(self.trg_imgs[idx], self.trg_masks[idx])
            return t_img, t_mask_long, self.trg_imgs[idx]
            
        src_idx = idx % len(self.src_imgs)
        trg_idx = np.random.randint(0, self.trg_len)
        
        s_img, s_mask_long, s_mask_onehot, s_edge, s_body = self.process_item(self.src_imgs[src_idx], self.src_masks[src_idx])
        t_img = self.process_item(self.trg_imgs[trg_idx])
        
        s_img_fda = FDA_source_to_target(s_img, t_img, L=CONFIG['FDA_BETA'])
        
        return s_img, s_img_fda, s_mask_long, s_mask_onehot, s_edge, s_body, t_img
    
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size; self.data = []
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element); to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone()); self.data[i] = element
                else: to_return.append(element)
        return torch.cat(to_return)

# ==============================================================================
# 7. ROBUST METRICS EVALUATION
# ==============================================================================
def compute_surface_metrics(pred, gt):
    try:
        pred = pred.astype(bool); gt = gt.astype(bool)
        if np.sum(pred) == 0 and np.sum(gt) == 0: return 0.0, 0.0
        if np.sum(pred) == 0 or np.sum(gt) == 0: return 373.12, 373.12 
        
        struct = np.ones((3,3,3), dtype=bool)
        pred_border = pred ^ binary_erosion(pred, structure=struct)
        gt_border = gt ^ binary_erosion(gt, structure=struct)
        
        pred_pts = np.argwhere(pred_border); gt_pts = np.argwhere(gt_border)
        if len(pred_pts) == 0 or len(gt_pts) == 0: return 373.12, 373.12
            
        tree_pred = cKDTree(pred_pts); tree_gt = cKDTree(gt_pts)
        dist_pred_to_gt, _ = tree_gt.query(pred_pts)
        dist_gt_to_pred, _ = tree_pred.query(gt_pts)
        
        all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
        return float(np.percentile(all_dists, 95)), float(np.mean(all_dists))
    except Exception:
        return 373.12, 373.12

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)

def evaluate_3d_metrics(model, dataloader, device):
    model.eval()
    patient_preds, patient_gts = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for t_img, t_mask_long, path in dataloader:
            pid = os.path.basename(path[0]).split('_slice')[0]
            _, _, p_final, _ = model(t_img.to(device))
            
            # 🌟 Argmax Evaluation
            pred = torch.argmax(p_final, dim=1).cpu().numpy()[0] 
            gt = t_mask_long[0].numpy()
            
            patient_preds[pid].append(pred)
            patient_gts[pid].append(gt)

    all_dices, all_hd95s, all_asds = [], [], []
    for pid in patient_preds.keys():
        vol_pred = np.stack(patient_preds[pid], axis=0)
        vol_gt = np.stack(patient_gts[pid], axis=0)
        
        pt_dice, pt_hd95, pt_asd = [], [], []
        # Evaluate 1=LIV, 2=RKID, 3=LKID, 4=SPL
        for c in range(1, 5): 
            p_c = (vol_pred == c).astype(np.uint8)
            g_c = (vol_gt == c).astype(np.uint8)
            
            inter, union = np.sum(p_c * g_c), np.sum(p_c) + np.sum(g_c)
            pt_dice.append((2 * inter + 1e-8) / (union + 1e-8))
            
            hd95, asd = compute_surface_metrics(p_c, g_c)
            pt_hd95.append(hd95); pt_asd.append(asd)
            
        all_dices.append(pt_dice); all_hd95s.append(pt_hd95); all_asds.append(pt_asd)

    m_dices, m_hd95s, m_asds = np.mean(all_dices, axis=0), np.mean(all_hd95s, axis=0), np.mean(all_asds, axis=0)
    return m_dices.mean(), m_dices, m_hd95s.mean(), m_hd95s, m_asds.mean(), m_asds

def save_qualitative_results(G, S, dataset, device, save_dir, epoch):
    idx = np.random.randint(0, len(dataset))
    s_img, s_img_fda, s_mask_long, _, _, _, t_img = dataset[idx]
    G.eval(); S.eval()
    with torch.no_grad():
        fake_mr = G(s_img_fda.unsqueeze(0).to(device))
        _, _, p_final, _ = S(fake_mr)
        _, _, t_final, _ = S(t_img.unsqueeze(0).to(device))
        
    def to_img(t): return ((t.squeeze().cpu().numpy() - t.min().item()) / (t.max().item() - t.min().item() + 1e-8) * 255).astype(np.uint8)
    
    c_img, c_fda, f_mr, r_mr = to_img(s_img), to_img(s_img_fda), to_img(fake_mr), to_img(t_img)
    gt = s_mask_long.numpy()
    p_fake = torch.argmax(p_final[0], dim=0).cpu().numpy()
    p_real = torch.argmax(t_final[0], dim=0).cpu().numpy()

    def overlay(bg, m_argmax):
        rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)
        for c in range(1, 5): rgb[m_argmax == c] = COLORS[c]
        return cv2.addWeighted(np.stack([bg]*3, -1), 0.6, rgb, 0.4, 0)

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax[0,0].imshow(c_img, cmap='gray'); ax[0,0].set_title("Source CT (AMOS22)")
    ax[0,1].imshow(c_fda, cmap='gray'); ax[0,1].set_title("FDA Transformed CT")
    ax[0,2].imshow(f_mr, cmap='gray'); ax[0,2].set_title("Generator Fake MR")
    ax[0,3].imshow(overlay(f_mr, gt)); ax[0,3].set_title("Fake MR + GT Mask")
    
    ax[1,0].imshow(r_mr, cmap='gray'); ax[1,0].set_title("Real MR (AMOS22 Unseen)")
    ax[1,1].imshow(overlay(f_mr, p_fake)); ax[1,1].set_title("Seg Pred on Fake MR")
    ax[1,2].imshow(overlay(r_mr, p_real)); ax[1,2].set_title("Seg Pred on Real MR")
    ax[1,3].axis('off')
    [a.axis('off') for a in ax.flatten()]
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"AMOS_CT2MR_Qualitative_{epoch}.png"), dpi=300)
    plt.close()
    G.train(); S.train()

# ==============================================================================
# 8. MAIN EXECUTION (CT -> MR Phase 1)
# ==============================================================================
if __name__ == '__main__':
    print(f"\n--- Initializing AMOS22 UDA SOTA (CT -> MR) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = Generator().to(device)
    D_img = ImageDiscriminator().to(device)
    
    S = DBSNet(out_channels=CONFIG['NUM_CLASSES']).to(device)
    
    if not os.path.exists(CONFIG['PRETRAINED_SEGMENTOR']):
        print(f"⚠️ ERROR: Could not find Supervised CT Baseline at {CONFIG['PRETRAINED_SEGMENTOR']}")
        print("Please run the supervised training script on AMOS CT first!")
        exit(1)
        
    print(f"Loading Pre-trained AMOS CT Segmentor: {os.path.basename(CONFIG['PRETRAINED_SEGMENTOR'])}")
    S.load_state_dict(torch.load(CONFIG['PRETRAINED_SEGMENTOR'], map_location=device, weights_only=True))
    
    S_Teacher = DBSNet(out_channels=CONFIG['NUM_CLASSES']).to(device)
    S_Teacher.load_state_dict(S.state_dict())
    for param in S_Teacher.parameters(): param.requires_grad = False
        
    D_feat = FeatureDiscriminator().to(device)
    
    criterion_GAN = nn.MSELoss() 
    criterion_Struct = StructureLoss()
    criterion_Seg = SoftmaxWeightedLoss()
    
    opt_G = optim.Adam(G.parameters(), lr=CONFIG['LR_GAN'], betas=(0.5, 0.999))
    opt_D_img = optim.Adam(D_img.parameters(), lr=CONFIG['LR_GAN'] / 2.0, betas=(0.5, 0.999)) 
    opt_S = optim.Adam(S.parameters(), lr=CONFIG['LR_SEG'])
    opt_D_feat = optim.Adam(D_feat.parameters(), lr=CONFIG['LR_SEG'] / 2.0) 
    
    def lr_lambda(epoch): return 1.0 - max(0, epoch - 50) / float(100 - 50)
    sch_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sch_D_img = optim.lr_scheduler.LambdaLR(opt_D_img, lr_lambda=lr_lambda)
    sch_S = optim.lr_scheduler.LambdaLR(opt_S, lr_lambda=lr_lambda)
    sch_D_feat = optim.lr_scheduler.LambdaLR(opt_D_feat, lr_lambda=lr_lambda)

    fake_ct_buffer = ReplayBuffer(max_size=50)

    src_files = sorted([f for f in os.listdir(CONFIG['SOURCE_IMG_DIR']) if f.endswith('.npy')])
    s_i = [os.path.join(CONFIG['SOURCE_IMG_DIR'], f) for f in src_files]
    s_m = [os.path.join(CONFIG['SOURCE_MASK_DIR'], f.replace('.npy', '.png')) for f in src_files]
    
    trg_files = sorted([f for f in os.listdir(CONFIG['TARGET_IMG_DIR']) if f.endswith('.npy')])
    t_cases = sorted(list(set([f.split('_slice')[0] for f in trg_files])))
    t_train_c, t_temp = train_test_split(t_cases, test_size=0.2, random_state=42)
    t_val_c, t_test_c = train_test_split(t_temp, test_size=0.5, random_state=42)

    def filter_t(cases, masks=False):
        i = [os.path.join(CONFIG['TARGET_IMG_DIR'], f) for f in trg_files if any(c in f for c in cases)]
        if not masks: return i
        m = [os.path.join(CONFIG['TARGET_MASK_DIR'], f.replace('.npy', '.png')) for f in trg_files if any(c in f for c in cases)]
        return i, m

    train_ds = UnpairedUDADataset(s_i, s_m, filter_t(t_train_c), img_size=CONFIG['IMG_SIZE'])
    val_ds = UnpairedUDADataset(None, None, *filter_t(t_val_c, True), img_size=CONFIG['IMG_SIZE'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    best_dice = 0.0
    history = {'L_G':[], 'L_D_img':[], 'L_Seg':[], 'L_D_feat':[], 'L_Pseudo':[], 'Val_Dice':[]}
    
    def get_alpha(epoch): return 2. / (1. + np.exp(-10 * (epoch / (CONFIG['EPOCHS'] * 1.5)))) - 1
    def get_pseudo_alpha(epoch, ramp_epochs=20):
        if epoch < ramp_epochs: return (epoch / ramp_epochs) * CONFIG['LAMBDA_PSEUDO']
        return CONFIG['LAMBDA_PSEUDO']

    print(f"\nStarting SOTA FDA+CBST UDA for {CONFIG['EPOCHS']} Epochs...")
    for epoch in range(CONFIG['EPOCHS']):
        G.train(); D_img.train(); S.train(); D_feat.train()
        alpha_grl, alpha_pseudo = get_alpha(epoch), get_pseudo_alpha(epoch)
        meter = {'G':0, 'D_i':0, 'S':0, 'D_f':0, 'P':0}
        
        for batch_idx, (s_img, s_img_fda, s_mask_long, s_mask_onehot, s_edge, s_body, t_img) in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")):
            s_img_fda, s_mask_long, s_mask_onehot = s_img_fda.to(device), s_mask_long.to(device), s_mask_onehot.to(device)
            s_edge, s_body, t_img = s_edge.to(device), s_body.to(device), t_img.to(device)
            
            valid = torch.full((s_img.size(0), 1, 16, 16), 0.9, device=device)
            fake = torch.zeros((s_img.size(0), 1, 16, 16), device=device)
            
            # 1. Train Discriminator
            opt_D_img.zero_grad()
            fake_mr = G(s_img_fda) 
            fake_mr_buffered = fake_ct_buffer.push_and_pop(fake_mr)
            loss_D_img = 0.5 * (criterion_GAN(D_img(t_img), valid) + criterion_GAN(D_img(fake_mr_buffered.detach()), fake))
            loss_D_img.backward()
            opt_D_img.step()
            
            # 2. Train Generator
            opt_G.zero_grad()
            loss_G = criterion_GAN(D_img(fake_mr), valid) + CONFIG['LAMBDA_STRUCT'] * criterion_Struct(s_img_fda, fake_mr)
            loss_G.backward()
            opt_G.step()
            
            # ----------------------------------
            # 3. Train Segmentor (Softmax CBST)
            # ----------------------------------
            opt_S.zero_grad(); opt_D_feat.zero_grad()
            
            p_b_real, p_e_real, p_f_real, _ = S(s_img_fda)
            loss_Seg_Real = criterion_Seg(p_b_real, p_e_real, p_f_real, s_mask_long, s_mask_onehot, s_edge, s_body)
            
            p_b_fake, p_e_fake, p_f_fake, feat_fake = S(fake_mr.detach()) 
            loss_Seg_Fake = criterion_Seg(p_b_fake, p_e_fake, p_f_fake, s_mask_long, s_mask_onehot, s_edge, s_body)
            
            loss_Seg = (loss_Seg_Real + loss_Seg_Fake) * 0.5
            
            # 🌟 MEAN TEACHER LOGIC: SOFTMAX CLASS-BALANCED CBST
            with torch.no_grad():
                _, _, p_t_final_teacher, _ = S_Teacher(t_img)
                prob_teacher = torch.softmax(p_t_final_teacher, dim=1).detach()
                
                argmax_teacher = torch.argmax(prob_teacher, dim=1, keepdim=True)
                pseudo_labels = torch.zeros_like(prob_teacher).scatter_(1, argmax_teacher, 1.0)
                
                thresh = CONFIG['CONFIDENCE_THRESH']
                max_prob_teacher, _ = torch.max(prob_teacher, dim=1, keepdim=True)
                conf_mask = (max_prob_teacher > thresh).float()
                conf_mask = conf_mask.expand_as(prob_teacher) 
                
            _, _, p_t_final_student, feat_real = S(t_img)
            prob_student = torch.softmax(p_t_final_student, dim=1)
            
            prob_student_organs = prob_student[:, 1:, :, :]
            pseudo_labels_organs = pseudo_labels[:, 1:, :, :]
            conf_mask_organs = conf_mask[:, 1:, :, :]
            
            class_weights = torch.tensor([1.0, 5.0, 5.0, 2.0], device=device).view(1, 4, 1, 1)
            
            bce = F.binary_cross_entropy(prob_student_organs, pseudo_labels_organs, reduction='none')
            weighted_bce = bce * class_weights
            masked_bce = (weighted_bce * conf_mask_organs).sum() / (conf_mask_organs.sum() + 1e-8)
            
            inter = (prob_student_organs * pseudo_labels_organs * conf_mask_organs * class_weights).sum(dim=(0, 2, 3))
            union = ((prob_student_organs + pseudo_labels_organs) * conf_mask_organs * class_weights).sum(dim=(0, 2, 3)) 
            masked_dice = 1 - ((2. * inter + 1e-8) / (union + 1e-8)).mean()
            
            loss_Consistency = masked_bce + masked_dice
            
            # Feature Alignment
            v_feat = torch.full((s_img.size(0), 1, 8, 8), 0.9, device=device)
            f_feat = torch.zeros((s_img.size(0), 1, 8, 8), device=device)
            bce_logits = nn.BCEWithLogitsLoss()
            loss_D_feat = 0.5 * (bce_logits(D_feat(feat_fake, alpha_grl), f_feat) + bce_logits(D_feat(feat_real, alpha_grl), v_feat))
            
            loss_S_Total = loss_Seg + CONFIG['LAMBDA_ADV_FEAT'] * loss_D_feat + alpha_pseudo * loss_Consistency
            loss_S_Total.backward()
            
            torch.nn.utils.clip_grad_norm_(S.parameters(), max_norm=1.0)
            opt_S.step(); opt_D_feat.step()
            
            global_step = epoch * len(train_loader) + batch_idx 
            update_ema_variables(S, S_Teacher, alpha=0.99, global_step=global_step)
            
            meter['G'] += loss_G.item(); meter['D_i'] += loss_D_img.item()
            meter['S'] += loss_Seg.item(); meter['D_f'] += loss_D_feat.item()
            meter['P'] += loss_Consistency.item()

        v_dice, d_cls, _, _, _, _ = evaluate_3d_metrics(S, val_loader, device)
        sch_G.step(); sch_D_img.step(); sch_S.step(); sch_D_feat.step()
        
        n = len(train_loader)
        history['L_G'].append(meter['G']/n); history['L_D_img'].append(meter['D_i']/n)
        history['L_Seg'].append(meter['S']/n); history['L_D_feat'].append(meter['D_f']/n)
        history['L_Pseudo'].append(meter['P']/n); history['Val_Dice'].append(v_dice)

        print(f"Ep {epoch+1:03d} | W_CBST:{alpha_pseudo:.2f} | L_S:{meter['S']/n:.3f} | L_Pseu:{meter['P']/n:.3f} | Target Val 3D Dice: {v_dice:.4f} (LIV:{d_cls[0]:.2f} RK:{d_cls[1]:.2f} LK:{d_cls[2]:.2f} SPL:{d_cls[3]:.2f})")

        if v_dice > best_dice:
            best_dice = v_dice
            torch.save(S.state_dict(), os.path.join(CONFIG['SAVE_DIR'], "best_AMOS_Segmentor_CT2MR.pth"))
            torch.save(G.state_dict(), os.path.join(CONFIG['SAVE_DIR'], "best_AMOS_Generator_CT2MR.pth"))
            print("   🌟 New AMOS UDA Phase 1 Checkpoint Saved!")
            
        if (epoch+1) % 10 == 0: save_qualitative_results(G, S, train_ds, device, CONFIG['SAVE_DIR'], epoch+1)

    print("\n" + "="*70)
    print("🚀 FINAL ZERO-SHOT TEST RESULTS (AMOS CT -> MR)")
    print("="*70)
    
    test_ds = UnpairedUDADataset(None, None, *filter_t(t_test_c, True), img_size=CONFIG['IMG_SIZE'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    S.load_state_dict(torch.load(os.path.join(CONFIG['SAVE_DIR'], "best_AMOS_Segmentor_CT2MR.pth"), weights_only=True))
    test_dice, d_cls, test_hd, hd_cls, test_asd, asd_cls = evaluate_3d_metrics(S, test_loader, device)
    
    print(f"DICE (%)  -> LIV: {d_cls[0]*100:5.2f} | RK: {d_cls[1]*100:5.2f} | LK: {d_cls[2]*100:5.2f} | SPL: {d_cls[3]*100:5.2f} | AVG: {test_dice*100:5.2f}")
    print(f"HD95 (mm) -> LIV: {hd_cls[0]:5.2f} | RK: {hd_cls[1]:5.2f} | LK: {hd_cls[2]:5.2f} | SPL: {hd_cls[3]:5.2f} | AVG: {test_hd:5.2f}")
    print(f"ASD (mm)  -> LIV: {asd_cls[0]:5.2f} | RK: {asd_cls[1]:5.2f} | LK: {asd_cls[2]:5.2f} | SPL: {asd_cls[3]:5.2f} | AVG: {test_asd:5.2f}")
    print("="*70 + "\n")