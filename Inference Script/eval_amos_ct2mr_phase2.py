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
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm.auto import tqdm

# ==============================================================================
# 0. PUBLICATION THEME & SEEDS
# ==============================================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. EVALUATION CONFIGURATION (CT2MR EXCLUSIVE)
# ==============================================================================
CONFIG = {
    'IMG_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Images',
    'MASK_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Masks',
    'TASK': 'CT2MR',
    'TARGET': 'MR',
    'IMG_SIZE': 256,
    'NUM_CLASSES': 5, # 0=BG, 1=Liver, 2=RKid, 3=LKid, 4=Spleen
    'NUM_WORKERS': 8
}

CONFIG['SAVE_DIR'] = f"D:\\Nilesh\\AMOS22\\UDA_{CONFIG['TASK']}_Phase2_GST"
CONFIG['MODEL_PATH'] = os.path.join(CONFIG['SAVE_DIR'], f"best_AMOS_Segmentor_{CONFIG['TASK']}_Phase2_GST.pth")

# Create completely isolated output directories for separate standalone figures
CONFIG['OUT_DIR'] = os.path.join(CONFIG['SAVE_DIR'], f"Isolated_Publication_Outputs_{CONFIG['TASK']}_FINAL")
CONFIG['PLOT_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Metrics_Vector_Charts')
CONFIG['QUAL_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Qualitative_Slices_Isolated')
CONFIG['INTER_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Intermediate_Features_Isolated')
CONFIG['BLAND_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Bland_Altman_Plots')

for d in [CONFIG['PLOT_DIR'], CONFIG['QUAL_DIR'], CONFIG['INTER_DIR'], CONFIG['BLAND_DIR']]:
    os.makedirs(d, exist_ok=True)

CLASSES = ['Liver', 'Right Kidney', 'Left Kidney', 'Spleen']
COLORS_HEX = ['#FFFF00', '#00FF00', '#0000FF', '#FF0000'] # Yellow, Green, Blue, Red
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
        return self.out_body(F_body), self.out_edge(F_feat - F_body), self.out_final(F_feat), bottleneck

# ==============================================================================
# 3. ROBUST TARGET DATASET (NO CT CLIPPING FOR MR TARGETS)
# ==============================================================================
class ExactEvalDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=256):
        self.img_paths, self.mask_paths, self.img_size = img_paths, mask_paths, img_size

    def __len__(self): return len(self.img_paths)

    def pad_crop(self, img, is_mask=False):
        h, w = img.shape
        ph, pw = max(0, self.img_size - h), max(0, self.img_size - w)
        if ph > 0 or pw > 0: 
            pad_val = 0 if is_mask else np.min(img)
            img = np.pad(img, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)), mode='constant', constant_values=pad_val)
        h, w = img.shape
        sh, sw = (h - self.img_size) // 2, (w - self.img_size) // 2
        return img[sh:sh+self.img_size, sw:sw+self.img_size]

    def __getitem__(self, idx):
        # 1. Safely load and ensure 2D structure
        img_raw = np.load(self.img_paths[idx]).squeeze()
        
        # 2. Pad/Crop exactly to size
        img_processed = self.pad_crop(img_raw, is_mask=False)
        
        # 3. Pure Min-Max scaling to [-1, 1] - MR targets do not need HU windowing
        img_t = torch.from_numpy(img_processed).unsqueeze(0).float()
        img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-8) * 2.0 - 1.0
        
        # 4. Process Mask
        mask_raw = np.array(Image.open(self.mask_paths[idx]))
        mask_processed = self.pad_crop(mask_raw, is_mask=True)
        
        return img_t, torch.from_numpy(mask_processed).long(), self.img_paths[idx]

# ==============================================================================
# 4. ROBUST 3D POST-PROCESSING & METRICS
# ==============================================================================
def keep_largest_connected_component(mask_volume):
    out_mask = np.zeros_like(mask_volume)
    for c in range(1, 5): 
        organ_mask = mask_volume == c
        if not np.any(organ_mask): continue
        labeled_array, num_features = label(organ_mask)
        if num_features == 1:
            out_mask[organ_mask] = c
            continue
        sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
        largest_label = np.argmax(sizes) + 1
        out_mask[labeled_array == largest_label] = c
    return out_mask

def compute_surface_distances(pred, gt):
    try:
        pred, gt = pred.astype(bool), gt.astype(bool)
        if np.sum(pred) == 0 and np.sum(gt) == 0: return 0.0, 0.0
        if np.sum(pred) == 0 or np.sum(gt) == 0: return 373.12, 373.12 
        
        struct = np.ones((3,3,3), dtype=bool)
        pred_border = pred ^ binary_erosion(pred, structure=struct)
        gt_border = gt ^ binary_erosion(gt, structure=struct)
        
        p_pts, g_pts = np.argwhere(pred_border), np.argwhere(gt_border)
        if len(p_pts) == 0 or len(g_pts) == 0: return 373.12, 373.12
            
        tree_pred, tree_gt = cKDTree(p_pts), cKDTree(g_pts)
        d_p2g, _ = tree_gt.query(p_pts)
        d_g2p, _ = tree_pred.query(g_pts)
        
        all_dists = np.concatenate([d_p2g, d_g2p])
        return float(np.percentile(all_dists, 95)), float(np.mean(all_dists))
    except Exception: return 373.12, 373.12

# ==============================================================================
# 5. EXPORTERS, ERROR MAPS, AND BLAND-ALTMAN ANALYSIS
# ==============================================================================
def plot_bland_altman(pred_vols, gt_vols, class_name, color_hex, save_dir):
    p_vols, g_vols = np.array(pred_vols), np.array(gt_vols)
    means, diffs = (p_vols + g_vols) / 2.0, p_vols - g_vols
    mean_bias, std_bias = np.mean(diffs), np.std(diffs)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.7, color=color_hex, edgecolor='black', s=50)
    plt.axhline(mean_bias, color='black', linestyle='-', linewidth=2, label=f'Mean Bias: {mean_bias:.1f}')
    plt.axhline(mean_bias + 1.96 * std_bias, color='gray', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {mean_bias + 1.96 * std_bias:.1f}')
    plt.axhline(mean_bias - 1.96 * std_bias, color='gray', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {mean_bias - 1.96 * std_bias:.1f}')
    
    plt.title(f'Bland-Altman Analysis: {class_name}', fontweight='bold', pad=15)
    plt.xlabel('Average Volume (Voxels)', fontweight='bold')
    plt.ylabel('Difference (Predicted - Ground Truth)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Bland_Altman_{class_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def save_standalone_pdf_boxplot(df, metric_col, title, filename, save_dir):
    plt.figure(figsize=(7, 6))
    sns.boxplot(x='Structure', y=metric_col, hue='Structure', data=df, palette=COLORS_HEX, width=0.6, boxprops=dict(alpha=0.8), legend=False)
    sns.stripplot(x='Structure', y=metric_col, hue='Structure', data=df, color='black', alpha=0.3, jitter=True, legend=False)
    plt.title(title, fontweight='bold', pad=15)
    if metric_col == 'DSC (%)': plt.ylim(0, 105)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def save_pure_image(img_array, filename, save_dir, cmap='gray'):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array, cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def create_error_map(bg_img, pred, gt):
    rgb = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2RGB)
    overlay = np.zeros_like(rgb)
    tp = (pred == gt) & (gt > 0)
    fp = (pred > 0) & (pred != gt)
    fn = (gt > 0) & (pred != gt)
    overlay[tp], overlay[fp], overlay[fn] = [0, 255, 0], [255, 0, 0], [0, 0, 255]
    mask = tp | fp | fn
    rgb[mask] = cv2.addWeighted(rgb[mask], 0.4, overlay[mask], 0.6, 0)
    return rgb

def export_best_slice_visuals(img_tensor, gt_argmax, pred_argmax, edge_tensor, bottleneck_tensor, pid, slice_idx, dirs):
    def to_img(t): return ((t.squeeze().cpu().numpy() - t.min().item()) / (t.max().item() - t.min().item() + 1e-8) * 255).astype(np.uint8)
    def overlay_mask(bg, m_argmax):
        rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)
        for c in range(1, 5): rgb[m_argmax == c] = COLORS_RGB[c-1]
        return cv2.addWeighted(np.stack([bg]*3, -1), 0.6, rgb, 0.4, 0)

    img_np = to_img(img_tensor)
    base_name = f"{pid}_best_slice{slice_idx:03d}"
    
    save_pure_image(img_np, f"{base_name}_01_Image.png", dirs['qual'])
    save_pure_image(overlay_mask(img_np, gt_argmax), f"{base_name}_02_GT.png", dirs['qual'])
    save_pure_image(overlay_mask(img_np, pred_argmax), f"{base_name}_03_Pred.png", dirs['qual'])
    save_pure_image(create_error_map(img_np, pred_argmax, gt_argmax), f"{base_name}_04_ErrorMap.png", dirs['qual'])

    edge_probs = torch.sigmoid(edge_tensor).squeeze().cpu().numpy()
    save_pure_image(np.max(edge_probs[1:], axis=0), f"{base_name}_05_UDA_Edge_Constraints.png", dirs['inter'], cmap='magma')
    
    btl_act = bottleneck_tensor.mean(dim=1).squeeze().cpu().numpy()
    btl_act_resized = cv2.resize(btl_act, (256, 256), interpolation=cv2.INTER_CUBIC)
    btl_act_norm = (btl_act_resized - btl_act_resized.min()) / (btl_act_resized.max() - btl_act_resized.min() + 1e-8)
    save_pure_image(btl_act_norm, f"{base_name}_06_UDA_Bottleneck.png", dirs['inter'], cmap='jet')

# ==============================================================================
# 6. PURE INFERENCE PIPELINE
# ==============================================================================
if __name__ == '__main__':
    print(f"\n{'='*70}\n🚀 INITIATING SOTA ZERO-SHOT EVALUATION ({CONFIG['TASK']})\n{'='*70}")
    
    if not os.path.exists(CONFIG['MODEL_PATH']):
        raise FileNotFoundError(f"Weights missing! Ensure {CONFIG['MODEL_PATH']} exists.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trg_dir = os.path.join(CONFIG['IMG_DIR'], CONFIG['TARGET'])
    trg_mask_dir = os.path.join(CONFIG['MASK_DIR'], CONFIG['TARGET'])
    
    all_files = sorted([f for f in os.listdir(trg_dir) if f.endswith('.npy')])
    all_cases = sorted(list(set([f.split('_slice')[0] for f in all_files])))
    
    _, temp_cases = train_test_split(all_cases, test_size=0.2, random_state=42)
    _, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)

    test_imgs = [os.path.join(trg_dir, f) for f in all_files if any(c in f for c in test_cases)]
    test_masks = [os.path.join(trg_mask_dir, f.replace('.npy', '.png')) for f in all_files if any(c in f for c in test_cases)]

    test_ds = ExactEvalDataset(test_imgs, test_masks, img_size=CONFIG['IMG_SIZE'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CONFIG['NUM_WORKERS'])

    print(f"Loading AMOS Segmentor Weights: {os.path.basename(CONFIG['MODEL_PATH'])}")
    S = DBSNet(out_channels=CONFIG['NUM_CLASSES']).to(device)
    S.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=device, weights_only=True))
    S.eval()

    patient_preds, patient_gts, patient_imgs = defaultdict(list), defaultdict(list), defaultdict(list)
    patient_edges, patient_btls = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for t_img, t_mask_long, path in tqdm(test_loader, desc="3D Volume Reconstruction"):
            pid = os.path.basename(path[0]).split('_slice')[0]
            
            _, p_edge, p_final, bottleneck = S(t_img.to(device))
            pred_argmax = torch.argmax(p_final, dim=1).cpu().numpy()[0]
            
            patient_preds[pid].append(pred_argmax)
            patient_gts[pid].append(t_mask_long[0].numpy())
            patient_imgs[pid].append(t_img)
            patient_edges[pid].append(p_edge)
            patient_btls[pid].append(bottleneck)

    all_dices, all_hd95s, all_asds = [], [], []
    
    # 🌟 Storage arrays for Bland-Altman volumetric calculations
    vol_preds_list = {c: [] for c in range(1, 5)}
    vol_gts_list = {c: [] for c in range(1, 5)}
    
    pids = list(patient_preds.keys())
    dirs = {'qual': CONFIG['QUAL_DIR'], 'inter': CONFIG['INTER_DIR'], 'bland': CONFIG['BLAND_DIR']}

    for pid in tqdm(pids, desc="Executing LCC Filtering & Visuals"):
        vol_pred = np.stack(patient_preds[pid], axis=0)
        vol_gt = np.stack(patient_gts[pid], axis=0)
        
        vol_pred = keep_largest_connected_component(vol_pred)
        
        pt_dice, pt_hd, pt_asd = [], [], []
        
        # Classes 1=Liv, 2=RKid, 3=LKid, 4=Spl
        for c in range(1, 5): 
            p_c = (vol_pred == c).astype(np.uint8)
            g_c = (vol_gt == c).astype(np.uint8)
            
            # Store absolute voxel counts for Bland-Altman analysis
            vol_preds_list[c].append(np.sum(p_c))
            vol_gts_list[c].append(np.sum(g_c))
            
            inter, union = np.sum(p_c * g_c), np.sum(p_c) + np.sum(g_c)
            pt_dice.append((2 * inter + 1e-8) / (union + 1e-8))
            
            hd, asd = compute_surface_distances(p_c, g_c)
            pt_hd.append(hd); pt_asd.append(asd)
            
        all_dices.append(pt_dice); all_hd95s.append(pt_hd); all_asds.append(pt_asd)

        # Find the BEST slice based on 2D overlap
        best_slice_idx, best_slice_dice = 0, -1
        for z in range(vol_gt.shape[0]):
            gt_z, pred_z = vol_gt[z], vol_pred[z]
            if np.sum(gt_z) == 0 and np.sum(pred_z) == 0: continue
            
            inter = np.sum((pred_z > 0) & (pred_z == gt_z))
            union = np.sum(pred_z > 0) + np.sum(gt_z > 0)
            slice_dice = 2 * inter / (union + 1e-8)
            
            if slice_dice > best_slice_dice:
                best_slice_dice = slice_dice; best_slice_idx = z

        export_best_slice_visuals(
            patient_imgs[pid][best_slice_idx], vol_gt[best_slice_idx], vol_pred[best_slice_idx], 
            patient_edges[pid][best_slice_idx], patient_btls[pid][best_slice_idx], pid, best_slice_idx, dirs
        )

    # -------------------------------------------------------------------------
    # BLAND-ALTMAN ANALYSIS EXPORT
    # -------------------------------------------------------------------------
    print("Generating Bland-Altman Volumetric Bias Plots...")
    for c_idx in range(4):
        actual_class_int = c_idx + 1 # 1=Liv, 2=RKid, 3=LKid, 4=Spl
        plot_bland_altman(
            vol_preds_list[actual_class_int], 
            vol_gts_list[actual_class_int], 
            CLASSES[c_idx], 
            COLORS_HEX[c_idx], 
            dirs['bland']
        )

    # -------------------------------------------------------------------------
    # DATA EXPORT & VECTOR CHART GENERATION
    # -------------------------------------------------------------------------
    csv_path = os.path.join(CONFIG['PLOT_DIR'], f'GST_Test_Metrics_{CONFIG["TASK"]}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient_ID', 'Metric'] + CLASSES)
        for i, pid in enumerate(pids):
            writer.writerow([pid, 'DSC'] + list(np.array(all_dices[i]) * 100))
            writer.writerow([pid, 'HD95'] + all_hd95s[i])
            writer.writerow([pid, 'ASD'] + all_asds[i])
            
    data = []
    for pid_idx in range(len(all_dices)):
        for c_idx in range(4):
            data.append({
                'Structure': CLASSES[c_idx], 
                'DSC (%)': all_dices[pid_idx][c_idx] * 100, 
                'HD95 (mm)': all_hd95s[pid_idx][c_idx], 
                'ASD (mm)': all_asds[pid_idx][c_idx]
            })
    df = pd.DataFrame(data)

    save_standalone_pdf_boxplot(df, 'DSC (%)', 'Dice Similarity Coefficient (↑)', f"Vector_Chart_01_DSC_{CONFIG['TASK']}.pdf", CONFIG['PLOT_DIR'])
    save_standalone_pdf_boxplot(df, 'HD95 (mm)', 'Hausdorff Distance 95% (↓)', f"Vector_Chart_02_HD95_{CONFIG['TASK']}.pdf", CONFIG['PLOT_DIR'])
    save_standalone_pdf_boxplot(df, 'ASD (mm)', 'Average Surface Distance (↓)', f"Vector_Chart_03_ASD_{CONFIG['TASK']}.pdf", CONFIG['PLOT_DIR'])
    
    m_dice, m_hd, m_asd = np.mean(all_dices, axis=0), np.mean(all_hd95s, axis=0), np.mean(all_asds, axis=0)

    print("\n" + "="*70)
    print(f"🏆 FINAL AMOS22 UDA TEST RESULTS ({CONFIG['TASK']})")
    print("="*70)
    
    # Matches original CT2MR logic: [0=Liver, 1=RKid, 2=LKid, 3=Spleen]
    print(f"DICE (%)  -> LIV: {m_dice[0]*100:5.2f} | RK: {m_dice[1]*100:5.2f} | LK: {m_dice[2]*100:5.2f} | SPL: {m_dice[3]*100:5.2f} | AVG: {m_dice.mean()*100:5.2f}")
    print(f"HD95 (mm) -> LIV: {m_hd[0]:5.2f} | RK: {m_hd[1]:5.2f} | LK: {m_hd[2]:5.2f} | SPL: {m_hd[3]:5.2f} | AVG: {m_hd.mean():5.2f}")
    print(f"ASD (mm)  -> LIV: {m_asd[0]:5.2f} | RK: {m_asd[1]:5.2f} | LK: {m_asd[2]:5.2f} | SPL: {m_asd[3]:5.2f} | AVG: {m_asd.mean():5.2f}")
    
    print("="*70)
    print(f"✅ Separated Vector Charts (PDF):       {CONFIG['PLOT_DIR']}")
    print(f"✅ Bland-Altman Volumetric Plots (PDF): {CONFIG['BLAND_DIR']}")
    print(f"✅ Raw metrics exported (CSV):          {csv_path}")
    print(f"✅ Best Slice (GT, Pred, Error Maps):   {CONFIG['QUAL_DIR']}")
    print(f"✅ UDA feature & edge maps (PNG):       {CONFIG['INTER_DIR']}")