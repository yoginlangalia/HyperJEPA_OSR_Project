from train import train
from utils import applyPCA, createImageCubes, splitTrainTestSet
import scipy.io as sio
import torch
import numpy as np
import os

#定义超参数
class_num = 6  # 地物类别 (Trento has 6 classes: labels 1-6 become 0-5 after removing background)
EPOCH = 100 #迭代次数
patch_size = 15   # 每个像素周围提取 patch 的尺寸
N_RUNS = 1 #跑多次取平均精度
pca_components = 63 # 使用所有63个波段 (PCA with whitening normalizes data)
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

#路径信息
data = sio.loadmat('./data/trento.mat')
X = data['HSI_data']  # Hyperspectral image data
y = data['ground']    # Ground truth labels

# Create necessary directories
os.makedirs('./log', exist_ok=True)
os.makedirs('./results/trento', exist_ok=True)

train_path = r'./log/net_trento_train.pkl'
fine_path = r'./log/net_trento_fine.pkl'
accuracy_path = r'./results/trento/'

print('Hyperspectral data shape: ', X.shape)
print('Ground truth labels shape: ', y.shape)
print('Number of bands: ', X.shape[2])

# Apply PCA with all 63 components (whiten=True in applyPCA normalizes the data)
# This is CRITICAL: PCA whitening handles normalization, same as the working version
print('\n... ... PCA transformation (whitening normalizes data) ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

# Create image cubes from all labeled pixels
print('\nCreating image cubes...')
X_full, Y_full = createImageCubes(X_pca, y, windowSize=patch_size, removeZeroLabels=True)
print(f'  Full patches: {X_full.shape[0]:,} samples')

# Create 80/20 train/test split (Trento has no built-in split)
print('Creating 80/20 train/test split...')
X_TR, X_TE, Y_TR, Y_TE = splitTrainTestSet(X_full, Y_full, testRatio=0.2, randomState=seed)
print(f'  Training patches (TR): {X_TR.shape[0]:,} samples')
print(f'  Test patches (TE): {X_TE.shape[0]:,} samples')

#训练测试阶段
# - Pre-training: Uses 100% of X_TR (self-supervised, labels not used)
# - Fine-tuning: Samples few-shot from X_TR (with labels)
# - Testing: Uses 100% of X_TE
accuracy = train(X_TR, Y_TR, X_TE, Y_TE, class_num, N_RUNS,
      patch_size, pca_components, batch_size, device, EPOCH, train_path, fine_path, mask_ratio, encoder_depth, decoder_depth, seed)
print(accuracy)
with open(accuracy_path + 'acc.csv', 'a', encoding='utf-8') as f:
    f.write(accuracy)