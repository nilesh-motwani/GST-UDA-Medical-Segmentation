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
plt.rcParams.update({
    'font.size': 14, 
    'axes.labelsize': 16, 
    'axes.titlesize': 18,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. EVALUATION CONFIGURATION
# ==============================================================================
CONFIG = {
    'TASK': 'MR2CT_Fixed',
    
    # 🌟 PATHS
    'IMG_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Images\CT',
    'MASK_DIR': r'D:\Nilesh\AMOS22\AMOS22_Data_2D_Fixed\Masks\CT',
    'MODEL_PATH': r'D:\Nilesh\AMOS22\22 July\UDA_MR2CT_Phase2_GST\best_AMOS_Segmentor_MR2CT_Phase2_GST.pth',
    
    'IMG_SIZE': 256,
    'NUM_CLASSES': 5,
    'NUM_WORKERS': 8
}

CONFIG['OUT_DIR'] = f"D:\\Nilesh\\AMOS22\\Journal_Evaluation_{CONFIG['TASK']}"
CONFIG['PLOT_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Metrics_Vector_Charts')
CONFIG['QUAL_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Qualitative_Best_Slices')
CONFIG['INTER_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'UDA_Internal_Features')
CONFIG['BLAND_DIR'] = os.path.join(CONFIG['OUT_DIR'], 'Bland_Altman_Plots')

for d in [CONFIG['PLOT_DIR'], CONFIG['QUAL_DIR'], CONFIG['INTER_DIR'], CONFIG['BLAND_DIR']]:
    os.makedirs(d, exist_ok=True)

CLASSES = ['Liver', 'Right Kidney', 'Left Kidney', 'Spleen']
COLORS_HEX = ['#FFFF00', '#00FF00', '#0000FF', '#FF0000'] # Yellow, Green, Blue, Red
COLORS_RGB = [(255,255,0), (0,255,0), (0,0,255), (255,0,0)] 

# ==============================================================================
# 2. DBS-NET ARCHITECTURE
# ==============================================================================
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class AxialAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
    def forward(self, x, axis='h'):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 3, 2, 1).contiguous().view(B*W, H, C) if axis == 'h' else x.permute(0, 2, 3, 1).contiguous().view(B*H, W, C)
        out, _ = self.mha(x_perm, x_perm, x_perm)
        return out.view(B, W, H, C).permute(0, 3, 2, 1) if axis == 'h' else out.view(B, H, W, C).permute(0, 3, 1, 2), None

class ESA_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.hc = in_channels // 2
        self.conv_branch = nn.Sequential(nn.Conv2d(self.hc, self.hc, 3, padding=1), nn.BatchNorm2d(self.hc), nn.ReLU())
        self.axial_h = AxialAttention(self.hc, 8); self.axial_w = AxialAttention(self.hc, 8)
        self.conv_out = nn.Conv2d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        xc, xa = torch.split(x, self.hc, dim=1)
        ah, _ = self.axial_h(xa, 'h'); aw, _ = self.axial_w(ah, 'w')
        return self.bn(self.conv_out(torch.cat([self.conv_branch(xc), aw], dim=1))) + x, None, None

class ETR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, 1); self.esa = ESA_Block(in_channels)
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
# 3. TARGET INFERENCE DATASET
# ==============================================================================
class ExactEvalDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=256):
        self.img_paths, self.mask_paths, self.img_size = img_paths, mask_paths, img_size

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_np = np.load(self.img_paths[idx]).squeeze()
        img_resized = cv2.resize(img_np, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8) * 2.0 - 1.0
        
        mask_np = np.array(Image.open(self.mask_paths[idx]))
        mask_resized = cv2.resize(mask_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return img_tensor, torch.from_numpy(mask_resized).long(), self.img_paths[idx]

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
# 5. EXPORTERS & VISUALIZATION UTILITIES
# ==============================================================================
def plot_bland_altman(pred_vols, gt_vols, class_name, color_hex, save_dir):
    """Generates an upgraded, overflow-safe Bland-Altman analysis plot."""
    p_vols = np.array(pred_vols, dtype=np.float64)
    g_vols = np.array(gt_vols, dtype=np.float64)
    
    means = (p_vols + g_vols) / 2.0
    diffs = p_vols - g_vols
    
    mean_bias = np.mean(diffs)
    std_bias = np.std(diffs)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.7, color=color_hex, edgecolor='black', s=60, linewidths=1)
    
    plt.axhline(mean_bias, color='black', linestyle='-', linewidth=2, label=f'Mean Bias: {mean_bias:.1f}')
    plt.axhline(mean_bias + 1.96 * std_bias, color='red', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {mean_bias + 1.96 * std_bias:.1f}')
    plt.axhline(mean_bias - 1.96 * std_bias, color='red', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {mean_bias - 1.96 * std_bias:.1f}')
    
    plt.title(f'Bland-Altman Analysis: {class_name}', fontweight='bold', pad=15)
    plt.xlabel('Average Volume (Voxels)', fontweight='bold')
    plt.ylabel('Difference (Predicted - Ground Truth)', fontweight='bold')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Bland_Altman_{class_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def save_standalone_pdf_cohort_chart(df, metric_col, title, filename, save_dir):
    """Plots cohort-wide Mean +/- SD performance tracking at the 3D patient level."""
    plt.figure(figsize=(7, 6))
    summary = df.groupby('Structure')[metric_col].agg(['mean', 'std']).reindex(CLASSES)
    
    bars = plt.bar(summary.index, summary['mean'], yerr=summary['std'], 
                   color=COLORS_HEX, edgecolor='black', alpha=0.8, capsize=8, width=0.55, linewidth=1.2)
    
    # Annotate absolute values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (summary['std'].max() * 0.04), 
                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.title(title, fontweight='bold', pad=15)
    if metric_col == 'DSC (%)': plt.ylim(0, 112)
    plt.ylabel(metric_col, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), format='pdf', dpi=300, bbox_inches='tight')
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
    
    overlay[tp] = [0, 255, 0]   # Green: TP
    overlay[fp] = [255, 0, 0]   # Red: FP
    overlay[fn] = [0, 0, 255]   # Blue: FN
    
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
    
    error_map = create_error_map(img_np, pred_argmax, gt_argmax)
    save_pure_image(error_map, f"{base_name}_04_ErrorMap.png", dirs['qual'])

    edge_probs = torch.sigmoid(edge_tensor).squeeze().cpu().numpy()
    combined_edge = np.max(edge_probs[1:], axis=0) 
    save_pure_image(combined_edge, f"{base_name}_05_UDA_Edge_Constraints.png", dirs['inter'], cmap='magma')
    
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
    
    all_files = sorted([f for f in os.listdir(CONFIG['IMG_DIR']) if f.endswith('.npy')])
    all_cases = sorted(list(set([f.split('_slice')[0] for f in all_files])))
    
    _, temp_cases = train_test_split(all_cases, test_size=0.2, random_state=42)
    _, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)

    test_imgs = [os.path.join(CONFIG['IMG_DIR'], f) for f in all_files if any(c in f for c in test_cases)]
    test_masks = [os.path.join(CONFIG['MASK_DIR'], f.replace('.npy', '.png')) for f in all_files if any(c in f for c in test_cases)]

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
    vol_preds_list = {c: [] for c in range(1, 5)}
    vol_gts_list = {c: [] for c in range(1, 5)}
    
    pids = list(patient_preds.keys())
    dirs = {'qual': CONFIG['QUAL_DIR'], 'inter': CONFIG['INTER_DIR'], 'bland': CONFIG['BLAND_DIR']}

    for pid in tqdm(pids, desc="Executing LCC Filtering & Generating Maps"):
        vol_pred = np.stack(patient_preds[pid], axis=0)
        vol_gt = np.stack(patient_gts[pid], axis=0)
        vol_pred = keep_largest_connected_component(vol_pred)
        
        pt_dice, pt_hd, pt_asd = [], [], []
        for c in range(1, 5): 
            p_c = (vol_pred == c).astype(np.uint8)
            g_c = (vol_gt == c).astype(np.uint8)

            vol_preds_list[c].append(np.sum(p_c))
            vol_gts_list[c].append(np.sum(g_c))            
            
            inter, union = np.sum(p_c * g_c), np.sum(p_c) + np.sum(g_c)
            pt_dice.append((2 * inter + 1e-8) / (union + 1e-8))
            
            hd, asd = compute_surface_distances(p_c, g_c)
            pt_hd.append(hd); pt_asd.append(asd)
            
        all_dices.append(pt_dice); all_hd95s.append(pt_hd); all_asds.append(pt_asd)

        # Find the BEST slice (highest 2D Dice) to visualize
        best_slice_idx = 0
        best_slice_dice = -1
        for z in range(vol_gt.shape[0]):
            gt_z, pred_z = vol_gt[z], vol_pred[z]
            if np.sum(gt_z) == 0 and np.sum(pred_z) == 0: continue
            
            inter = np.sum((pred_z > 0) & (pred_z == gt_z))
            union = np.sum(pred_z > 0) + np.sum(gt_z > 0)
            slice_dice = 2 * inter / (union + 1e-8)
            if slice_dice > best_slice_dice:
                best_slice_dice = slice_dice
                best_slice_idx = z

        export_best_slice_visuals(
            patient_imgs[pid][best_slice_idx], vol_gt[best_slice_idx], vol_pred[best_slice_idx], 
            patient_edges[pid][best_slice_idx], patient_btls[pid][best_slice_idx], pid, best_slice_idx, dirs
        )

    # -------------------------------------------------------------------------
    # BLAND-ALTMAN EXPORT (OVERFLOW-SAFE)
    # -------------------------------------------------------------------------
    print("Generating Bland-Altman Volumetric Bias Plots...")
    for c_idx in range(4):
        actual_class_int = c_idx + 1
        plot_bland_altman(vol_preds_list[actual_class_int], vol_gts_list[actual_class_int], CLASSES[c_idx], COLORS_HEX[c_idx], dirs['bland'])

    # -------------------------------------------------------------------------
    # DATA EXPORT & SUMMARY COHORT PLOTS
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

    # Output aggregate average tracking charts at the 3D Patient level
    save_standalone_pdf_cohort_chart(df, 'DSC (%)', 'Mean Dice Similarity Coefficient (↑)', f"Vector_Chart_01_DSC_{CONFIG['TASK']}.pdf", CONFIG['PLOT_DIR'])
    save_standalone_pdf_cohort_chart(df, 'HD95 (mm)', 'Mean Hausdorff Distance 95% (↓)', f"Vector_Chart_02_HD95_{CONFIG['TASK']}.pdf", CONFIG['PLOT_DIR'])
    save_standalone_pdf_cohort_chart(df, 'ASD (mm)', 'Mean Average Surface Distance (↓)', f"Vector_Chart_03_ASD_{CONFIG['TASK']}.pdf", CONFIG['PLOT_DIR'])
    
    m_dice, m_hd, m_asd = np.mean(all_dices, axis=0), np.mean(all_hd95s, axis=0), np.mean(all_asds, axis=0)

    print("\n" + "="*70)
    print(f"🏆 FINAL AMOS22 UDA TEST RESULTS ({CONFIG['TASK']})")
    print("="*70)
    print(f"DICE (%)  -> LIV: {m_dice[0]*100:5.2f} | RK: {m_dice[1]*100:5.2f} | LK: {m_dice[2]*100:5.2f} | SPL: {m_dice[3]*100:5.2f} | AVG: {m_dice.mean()*100:5.2f}")
    print(f"HD95 (mm) -> LIV: {m_hd[0]:5.2f} | RK: {m_hd[1]:5.2f} | LK: {m_hd[2]:5.2f} | SPL: {m_hd[3]:5.2f} | AVG: {m_hd.mean():5.2f}")
    print(f"ASD (mm)  -> LIV: {m_asd[0]:5.2f} | RK: {m_asd[1]:5.2f} | LK: {m_asd[2]:5.2f} | SPL: {m_asd[3]:5.2f} | AVG: {m_asd.mean():5.2f}")
    print("="*70)