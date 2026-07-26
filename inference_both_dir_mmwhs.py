import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import pandas as pd
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm.auto import tqdm
from math import pi

# ==============================================================================
# JOURNAL-QUALITY MATPLOTLIB CONFIGURATION
# ==============================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 14, 
    'axes.labelsize': 16, 
    'axes.titlesize': 18,
    'pdf.fonttype': 42,  
    'ps.fonttype': 42,
    'figure.autolayout': True  # Ideal for charts; safely bypassed for images below
})

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. EVALUATION CONFIGURATION
# ==============================================================================
TARGET_TASK = 'CT2MR'  # Switch between 'CT2MR' and 'MR2CT'

CONFIG = {
    'IMG_DIR': r'D:\Nilesh\GST-UDA\MMWHS\MMWHS_Data_2D\Images',
    'MASK_DIR': r'D:\Nilesh\GST-UDA\MMWHS\MMWHS_Data_2D\Masks',
    'TASK': TARGET_TASK,
    'IMG_SIZE': 256,
    'NUM_WORKERS': 4
}

if TARGET_TASK == 'CT2MR':
    CONFIG['TARGET'] = 'MR'
elif TARGET_TASK == 'MR2CT':
    CONFIG['TARGET'] = 'CT'
else:
    raise ValueError("Invalid TARGET_TASK. Must be 'CT2MR' or 'MR2CT'")

CONFIG['SAVE_DIR'] = f"D:\\Nilesh\\GST-UDA\\MMWHS\\GST_PHASE2_{CONFIG['TASK']}"
CONFIG['SEG_WEIGHTS'] = os.path.join(CONFIG['SAVE_DIR'], f"best_SIFA_FDA_Segmentor_{CONFIG['TASK']}.pth")

# Output directories for isolated PDF exports
CONFIG['OUT_DIR'] = os.path.join(CONFIG['SAVE_DIR'], f"Publication_Evals_{CONFIG['TASK']}")
CONFIG['PLOT_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Metrics_Radar_PDFs')
CONFIG['QUAL_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Qualitative_Slices_PDFs')
CONFIG['INTER_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Internal_Features_PDFs')
CONFIG['BLAND_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Bland_Altman_PDFs')

for d in [CONFIG['PLOT_DIR'], CONFIG['QUAL_DIR'], CONFIG['INTER_DIR'], CONFIG['BLAND_DIR']]:
    os.makedirs(d, exist_ok=True)

CLASSES = ['AA', 'LAC', 'LVC', 'MYO']
COLORS_HEX = ['#FFD700', '#00FF00', '#0000FF', '#FF0000']
COLORS_RGB = [(255,255,0), (0,255,0), (0,0,255), (255,0,0)]

# ==============================================================================
# 2. DBS-NET ARCHITECTURE (Segmentor Only)
# ==============================================================================
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class AxialAttention(nn.Module):
    def __init__(self, in_c, heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(in_c, heads, batch_first=True)
    def forward(self, x, axis='h'):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 3, 2, 1).contiguous().view(B*W, H, C) if axis == 'h' else x.permute(0, 2, 3, 1).contiguous().view(B*H, W, C)
        out, _ = self.mha(x_perm, x_perm, x_perm)
        return out.view(B, W, H, C).permute(0, 3, 2, 1) if axis == 'h' else out.view(B, H, W, C).permute(0, 3, 1, 2), None

class ESA_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.hc = in_c // 2
        self.conv_branch = nn.Sequential(nn.Conv2d(self.hc, self.hc, 3, padding=1), nn.BatchNorm2d(self.hc), nn.ReLU())
        self.axial_h = AxialAttention(self.hc, 8); self.axial_w = AxialAttention(self.hc, 8)
        self.conv_out = nn.Conv2d(in_c, in_c, 1); self.bn = nn.BatchNorm2d(in_c)
    def forward(self, x):
        xc, xa = torch.split(x, self.hc, dim=1)
        ah, _ = self.axial_h(xa, 'h'); aw, _ = self.axial_w(ah, 'w')
        return self.bn(self.conv_out(torch.cat([self.conv_branch(xc), aw], dim=1))) + x, None, None

class ETR(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, in_c, 1); self.esa = ESA_Block(in_c)
    def forward(self, x): return self.esa(self.proj(x))[0], None, None

class ConcatPlus(nn.Module):
    def forward(self, dec_feat, enc_feat):
        if dec_feat.shape[2:] != enc_feat.shape[2:]:
            enc_feat = F.interpolate(enc_feat, size=dec_feat.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([dec_feat + enc_feat, enc_feat], dim=1)

class DBSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
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
        return self.out_body(F_body), self.out_edge(F_feat - F_body), self.out_final(F_feat), bottleneck

# ==============================================================================
# 3. DATASET (Target Domain Only)
# ==============================================================================
class TargetTestDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=256):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self): return len(self.img_paths)

    def pad_crop(self, img, is_mask=False):
        h, w = img.shape
        ph, pw = max(0, self.img_size - h), max(0, self.img_size - w)
        if ph > 0 or pw > 0:
            img = np.pad(img, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)), mode='constant', constant_values=0 if is_mask else np.min(img))
        h, w = img.shape
        sh, sw = (h - self.img_size) // 2, (w - self.img_size) // 2
        return img[sh:sh+self.img_size, sw:sw+self.img_size]

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])
        if img.ndim == 3 and img.shape[0] == 1: img = img.squeeze(0)
        img_t = torch.from_numpy(self.pad_crop(img)).unsqueeze(0).float()
        img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 2.0 - 1.0
        
        m_c = self.pad_crop(np.array(Image.open(self.mask_paths[idx])), True)
        m_b = np.stack([(m_c == t).astype(np.float32) for t in [4, 2, 3, 1]], axis=-1)
        m_t = torch.from_numpy(m_b).permute(2,0,1).float()
        return img_t, m_t, self.img_paths[idx]

# ==============================================================================
# 4. ISOLATED JOURNAL PDF GENERATION UTILITIES
# ==============================================================================
def compute_surface_distances(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if np.sum(pred) == 0 and np.sum(gt) == 0: return 0.0, 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0: return 373.12, 373.12
    struct = np.ones((3,3,3))
    p_pts = np.argwhere(pred ^ binary_erosion(pred, structure=struct))
    g_pts = np.argwhere(gt ^ binary_erosion(gt, structure=struct))
    if len(p_pts) == 0 or len(g_pts) == 0: return 373.12, 373.12
    d_p2g, _ = cKDTree(g_pts).query(p_pts)
    d_g2p, _ = cKDTree(p_pts).query(g_pts)
    return np.percentile(np.concatenate([d_p2g, d_g2p]), 95), (np.mean(d_p2g) + np.mean(d_g2p)) / 2.0

def save_pure_pdf_image(img_array, filepath, cmap=None):
    """
    Saves image with a STRICT zero-margin 4x4 inch PDF container.
    This guarantees a properly sized PDF page that is 100% filled by the image,
    preventing PDF viewers from adding massive black background borders.
    """
    # Temporarily suspend ALL matplotlib padding and layout logic
    with plt.rc_context({'figure.autolayout': False, 'savefig.bbox': None, 'savefig.pad_inches': 0}):
        # Set a fixed, reasonable physical page size (4x4 inches)
        fig = plt.figure(figsize=(4, 4), dpi=100, frameon=False)
        
        # Add axes spanning exactly from 0 to 1 (100% of the figure)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Plot the image, forcing it to stretch to the exact axis bounds
        if cmap:
            ax.imshow(img_array, cmap=cmap, aspect='auto', interpolation='none')
        else:
            ax.imshow(img_array, aspect='auto', interpolation='none')
            
        fig.savefig(filepath, format='pdf')
        plt.close(fig)

def generate_isolated_bland_altman(vols_gt, vols_pred, save_dir):
    vols_gt, vols_pred = np.array(vols_gt), np.array(vols_pred)
    
    for c_idx, class_name in enumerate(CLASSES):
        gt_c, pred_c = vols_gt[:, c_idx], vols_pred[:, c_idx]
        mean = (gt_c + pred_c) / 2.0
        diff = pred_c - gt_c
        md = np.mean(diff)
        sd = np.std(diff, axis=0)
        
        plt.figure(figsize=(7, 6))
        plt.scatter(mean, diff, alpha=0.8, color=COLORS_HEX[c_idx], edgecolor='black', s=80, linewidths=1)
        plt.axhline(md, color='red', linestyle='-', lw=2.5, label=f'Mean Diff: {md:.0f}')
        plt.axhline(md + 1.96*sd, color='gray', linestyle='--', lw=2.5, label=f'+1.96 SD: {md + 1.96*sd:.0f}')
        plt.axhline(md - 1.96*sd, color='gray', linestyle='--', lw=2.5, label=f'-1.96 SD: {md - 1.96*sd:.0f}')
        
        plt.title(f'Bland-Altman Agreement: {class_name}', fontweight='bold', pad=15)
        plt.xlabel('Mean Volume (Pred + GT) / 2')
        plt.ylabel('Volume Difference (Pred - GT)')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        plt.savefig(os.path.join(save_dir, f'Bland_Altman_{class_name}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

def generate_isolated_radar_charts(dices, hd95s, asds, save_dir):
    metrics_data = {
        'DSC (%)': np.mean(dices, axis=0) * 100,
        'HD95 (mm)': np.mean(hd95s, axis=0),
        'ASD (mm)': np.mean(asds, axis=0)
    }
    
    angles = [n / float(len(CLASSES)) * 2 * pi for n in range(len(CLASSES))]
    angles += angles[:1]
    
    for metric_name, values in metrics_data.items():
        vals = values.tolist()
        vals += vals[:1]
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        
        ax.plot(angles, vals, color='teal', linewidth=3, linestyle='solid')
        ax.fill(angles, vals, color='teal', alpha=0.3)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(CLASSES, fontsize=14, fontweight='bold')
        ax.set_title(metric_name.split()[0], size=18, fontweight='bold', pad=20)
        
        if 'DSC' in metric_name: ax.set_ylim(0, 100)
        ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
        
        clean_name = metric_name.split()[0].replace('%', '').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(save_dir, f'Radar_{clean_name}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

def export_isolated_qualitative_pdfs(img_tensor, mask_tensor, pred_tensor, edge_tensor, btl_tensor, pid, slice_idx):
    def to_img(t): return ((t.squeeze().cpu().numpy() - t.min().item()) / (t.max().item() - t.min().item() + 1e-8) * 255).astype(np.uint8)
    def overlay(bg, m):
        rgb = np.stack([bg]*3, -1)
        for i in range(4): rgb[m[i]==1] = COLORS_RGB[i]
        return cv2.addWeighted(np.stack([bg]*3, -1), 0.6, rgb, 0.4, 0)
    
    img_np = to_img(img_tensor)
    gt_np = mask_tensor.squeeze().cpu().numpy()
    pred_np = pred_tensor.squeeze().cpu().numpy()
    
    # Generate multi-class Error Map
    gt_flat = np.argmax(np.concatenate([np.zeros((1, 256, 256)), gt_np], axis=0), axis=0)
    pred_flat = np.argmax(np.concatenate([np.zeros((1, 256, 256)), pred_np], axis=0), axis=0)
    error_map = np.stack([img_np]*3, axis=-1).astype(np.uint8) // 2 
    
    TP_mask = (gt_flat == pred_flat) & (gt_flat > 0)
    FP_mask = (pred_flat > 0) & (gt_flat != pred_flat)
    FN_mask = (gt_flat > 0) & (gt_flat != pred_flat)
    
    error_map[TP_mask] = [0, 255, 0]   
    error_map[FP_mask] = [255, 0, 0]   
    error_map[FN_mask] = [0, 0, 255]   

    base_name = f"{pid}_slice{slice_idx:03d}"
    
    # Save standard outputs individually as True Edge-to-Edge PDFs
    save_pure_pdf_image(img_np, os.path.join(CONFIG['QUAL_DIR'], f"{base_name}_01_Image.pdf"), cmap='gray')
    save_pure_pdf_image(overlay(img_np, gt_np), os.path.join(CONFIG['QUAL_DIR'], f"{base_name}_02_GT.pdf"))
    save_pure_pdf_image(overlay(img_np, pred_np), os.path.join(CONFIG['QUAL_DIR'], f"{base_name}_03_Pred.pdf"))
    save_pure_pdf_image(error_map, os.path.join(CONFIG['QUAL_DIR'], f"{base_name}_04_ErrorMap.pdf"))

    # Extract & Save Internal Features individually
    edge_probs = torch.sigmoid(edge_tensor).squeeze().cpu().numpy()
    combined_edge = np.max(edge_probs, axis=0)
    save_pure_pdf_image(combined_edge, os.path.join(CONFIG['INTER_DIR'], f"{base_name}_05_UDA_Edge.pdf"), cmap='magma')
    
    btl_act = btl_tensor.mean(dim=1).squeeze().cpu().numpy()
    btl_act_resized = cv2.resize(btl_act, (256, 256), interpolation=cv2.INTER_CUBIC)
    btl_norm = (btl_act_resized - btl_act_resized.min()) / (btl_act_resized.max() - btl_act_resized.min() + 1e-8)
    save_pure_pdf_image(btl_norm, os.path.join(CONFIG['INTER_DIR'], f"{base_name}_06_UDA_Bottleneck.pdf"), cmap='jet')

# ==============================================================================
# 5. MAIN EVALUATION PIPELINE
# ==============================================================================
if __name__ == '__main__':
    print(f"\n{'='*70}\n🚀 INITIATING ZERO-SHOT PDF INFERENCE EXPORT ({CONFIG['TASK']})\n{'='*70}")
    
    if not os.path.exists(CONFIG['SEG_WEIGHTS']):
        raise FileNotFoundError(f"Weights missing! Ensure {CONFIG['SEG_WEIGHTS']} exists.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trg_dir = os.path.join(CONFIG['IMG_DIR'], CONFIG['TARGET'])
    trg_mask_dir = os.path.join(CONFIG['MASK_DIR'], CONFIG['TARGET'])
    
    trg_files = sorted([f for f in os.listdir(trg_dir) if f.endswith('.npy')])
    t_cases = sorted(list(set([f.split('_slice')[0] for f in trg_files])))
    
    _, t_temp = train_test_split(t_cases, test_size=0.2, random_state=42)
    _, t_test_c = train_test_split(t_temp, test_size=0.5, random_state=42)

    test_imgs = [os.path.join(trg_dir, f) for f in trg_files if any(c in f for c in t_test_c)]
    test_masks = [os.path.join(trg_mask_dir, f.replace('.npy', '.png')) for f in trg_files if any(c in f for c in t_test_c)]

    test_ds = TargetTestDataset(test_imgs, test_masks, img_size=CONFIG['IMG_SIZE'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    S = DBSNet().to(device)
    S.load_state_dict(torch.load(CONFIG['SEG_WEIGHTS'], map_location=device, weights_only=True))
    S.eval()

    patient_preds, patient_gts, patient_imgs = defaultdict(list), defaultdict(list), defaultdict(list)
    patient_edges, patient_btls = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for t_img, t_mask, path in tqdm(test_loader, desc="3D Volume Reconstruction"):
            pid = os.path.basename(path[0]).split('_slice')[0]
            _, p_edge, p_final, bottleneck = S(t_img.to(device))
            
            pred_binary = (torch.sigmoid(p_final) > 0.5).float()
            
            patient_preds[pid].append(pred_binary.cpu().numpy()[0])
            patient_gts[pid].append(t_mask[0].numpy())
            patient_imgs[pid].append(t_img)
            patient_edges[pid].append(p_edge)
            patient_btls[pid].append(bottleneck)

    all_dices, all_hd95s, all_asds = [], [], []
    vols_gt, vols_pred = [], []
    pids = list(patient_preds.keys())

    for pid in tqdm(pids, desc="Computing Metrics & Rendering PDF Analytics"):
        vol_pred = np.stack(patient_preds[pid], axis=0).transpose(1, 0, 2, 3)
        vol_gt = np.stack(patient_gts[pid], axis=0).transpose(1, 0, 2, 3)
        
        pt_dice, pt_hd, pt_asd = [], [], []
        pt_vol_gt, pt_vol_pred = [], []
        
        for c in range(4): 
            pt_vol_gt.append(np.sum(vol_gt[c]))
            pt_vol_pred.append(np.sum(vol_pred[c]))
            
            inter, union = np.sum(vol_pred[c] * vol_gt[c]), np.sum(vol_pred[c]) + np.sum(vol_gt[c])
            pt_dice.append((2 * inter + 1e-8) / (union + 1e-8))
            hd, asd = compute_surface_distances(vol_pred[c], vol_gt[c])
            pt_hd.append(hd); pt_asd.append(asd)
            
        all_dices.append(pt_dice); all_hd95s.append(pt_hd); all_asds.append(pt_asd)
        vols_gt.append(pt_vol_gt); vols_pred.append(pt_vol_pred)

        mid_z = len(patient_imgs[pid]) // 2
        export_isolated_qualitative_pdfs(
            patient_imgs[pid][mid_z], 
            torch.from_numpy(patient_gts[pid][mid_z]), 
            torch.from_numpy(patient_preds[pid][mid_z]), 
            patient_edges[pid][mid_z],
            patient_btls[pid][mid_z],
            pid, mid_z
        )

    # -------------------------------------------------------------------------
    # DATA EXPORT & TERMINAL REPORTING 
    # -------------------------------------------------------------------------
    csv_path = os.path.join(CONFIG['OUT_DIR'], f'GST_Test_Metrics_{CONFIG["TASK"]}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient_ID', 'Metric'] + CLASSES)
        for i, pid in enumerate(pids):
            writer.writerow([pid, 'DSC'] + list(np.array(all_dices[i]) * 100))
            writer.writerow([pid, 'HD95'] + all_hd95s[i])
            writer.writerow([pid, 'ASD'] + all_asds[i])
            writer.writerow([pid, 'GT_Vol'] + vols_gt[i])
            writer.writerow([pid, 'Pred_Vol'] + vols_pred[i])
            
    generate_isolated_radar_charts(all_dices, all_hd95s, all_asds, CONFIG['PLOT_DIR'])
    generate_isolated_bland_altman(vols_gt, vols_pred, CONFIG['BLAND_DIR'])

    m_dice = np.mean(all_dices, axis=0)
    m_hd = np.mean(all_hd95s, axis=0)
    m_asd = np.mean(all_asds, axis=0)

    print("\n" + "="*70)
    print(f"🏆 FINAL ZERO-SHOT TEST RESULTS ({CONFIG['TASK']})")
    print("="*70)
    print(f"DICE (%)  -> AA: {m_dice[0]*100:.2f} | LAC: {m_dice[1]*100:.2f} | LVC: {m_dice[2]*100:.2f} | MYO: {m_dice[3]*100:.2f} | AVG: {m_dice.mean()*100:.2f}")
    print(f"HD95 (mm) -> AA: {m_hd[0]:.2f} | LAC: {m_hd[1]:.2f} | LVC: {m_hd[2]:.2f} | MYO: {m_hd[3]:.2f} | AVG: {m_hd.mean():.2f}")
    print(f"ASD (mm)  -> AA: {m_asd[0]:.2f} | LAC: {m_asd[1]:.2f} | LVC: {m_asd[2]:.2f} | MYO: {m_asd[3]:.2f} | AVG: {m_asd.mean():.2f}")
    print("="*70)

    print(f"✅ Isolated PDF Radar Charts saved to:  {CONFIG['PLOT_DIR']}")
    print(f"✅ Isolated PDF Bland-Altman saved to:  {CONFIG['BLAND_DIR']}")
    print(f"✅ Zero-Margin PDF Overlays saved to:   {CONFIG['QUAL_DIR']}")
    print(f"✅ Zero-Margin PDF Features saved to:   {CONFIG['INTER_DIR']}")