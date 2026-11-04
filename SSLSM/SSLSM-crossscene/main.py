from train import train
from utils import applyPCA, createImageCubes, splitTrainTestSet, loadCrossSceneDataset
import torch
import numpy as np
import os

#定义超参数
class_num = 7  # 7 classes (1-7, excluding background 0)
EPOCH = 100 #迭代次数
patch_size = 15   # 每个像素周围提取 patch 的尺寸
N_RUNS = 1 #跑多次取平均精度
pca_components = 15 # 使用 PCA 降维
batch_size = 64  #批处理量
mask_ratio = 0.5 #掩膜比例
encoder_depth = 8 #编码器深度
decoder_depth = 2 #解码器深度

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 使用GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create necessary directories
os.makedirs('./log', exist_ok=True)
os.makedirs('./results/crossscene', exist_ok=True)

train_path = r'./log/net_crossscene_train.pkl'
fine_path = r'./log/net_crossscene_fine.pkl'
accuracy_path = r'./results/crossscene/'

# ========== Load Cross-Scene Dataset ==========
print('Loading Cross-Scene Dataset (Houston 2013 & 2018)...')
X13, y13, X18, y18 = loadCrossSceneDataset('./data')

print(f'\nHouston 2013 - HSI shape: {X13.shape}, Labels shape: {y13.shape}')
print(f'Houston 2018 - HSI shape: {X18.shape}, Labels shape: {y18.shape}')
print(f'Value range - H13: [{X13.min():.4f}, {X13.max():.4f}], H18: [{X18.min():.4f}, {X18.max():.4f}]')

# ========== Apply PCA ==========
print(f'\n... ... PCA transformation (components={pca_components}) ... ...')
X13_pca = applyPCA(X13, numComponents=pca_components)
X18_pca = applyPCA(X18, numComponents=pca_components)
print(f'H13 after PCA: {X13_pca.shape}, H18 after PCA: {X18_pca.shape}')

# ========== Create Image Cubes ==========
print('\nCreating image cubes...')

# Houston-13: Use ALL labeled pixels (no split) - for pre-training + Exp1 fine-tuning
print('Houston-13 (Source): Creating patches from ALL labeled pixels...')
X13_full, Y13_full = createImageCubes(X13_pca, y13, windowSize=patch_size, removeZeroLabels=True)
print(f'  H13 patches: {X13_full.shape[0]:,} samples (100% for pre-training + Exp1 fine-tuning)')

# Houston-18: Split 80/20 - 80% for Exp2 fine-tuning, 20% for testing BOTH experiments
print('Houston-18 (Target): Creating patches and splitting 80/20...')
X18_patches, Y18_patches = createImageCubes(X18_pca, y18, windowSize=patch_size, removeZeroLabels=True)
X18_TR, X18_TE, Y18_TR, Y18_TE = splitTrainTestSet(X18_patches, Y18_patches, testRatio=0.2, randomState=seed)
print(f'  H18 Train (80%): {X18_TR.shape[0]:,} samples (for Exp2 fine-tuning)')
print(f'  H18 Test (20%): {X18_TE.shape[0]:,} samples (for testing BOTH experiments)')

# ========== Experiment Protocol ==========
print('\n' + '='*60)
print('CROSS-SCENE EXPERIMENT PROTOCOL')
print('='*60)
print(f'Pre-training: 100% of Houston-13 ({X13_full.shape[0]:,} samples) - ONCE')
print(f'Exp1 (Zero-shot): Fine-tune on H13, Test on H18')
print(f'Exp2 (Few-shot): Fine-tune on H18, Test on H18')
print('='*60)

# ========== Run Training ==========
accuracy = train(X13_full, Y13_full, X18_TR, Y18_TR, X18_TE, Y18_TE, class_num, N_RUNS,
      patch_size, pca_components, batch_size, device, EPOCH, train_path, fine_path, mask_ratio, encoder_depth, decoder_depth, seed)
print(accuracy)

# Save results
with open(accuracy_path + 'crossscene_results.csv', 'a', encoding='utf-8') as f:
    f.write(accuracy)