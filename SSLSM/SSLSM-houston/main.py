from train import train
from utils import applyPCA,createImageCubes
import scipy.io as sio
import torch
import numpy as np
import os

#定义超参数
class_num = 16  # 地物类别 (16 classes including background)
EPOCH = 100 #迭代次数
patch_size = 15   # 每个像素周围提取 patch 的尺寸
N_RUNS = 1 #跑多次取平均精度
pca_components = 15 # 使用 PCA 降维，得到主成分的数量
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

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#路径信息
#--数据集路径
data = sio.loadmat('./data/houston.mat')
X = data['input']
TR = data['TR']  # Training labels
TE = data['TE']  # Test labels
# Create necessary directories
os.makedirs('./log', exist_ok=True)
os.makedirs('./results/houston', exist_ok=True)

train_path = r'./log/net_houston_train.pkl'  #自训练模型保存路径
fine_path= r'./log/net_houston_fine.pkl'  #微调模型保存路径
accuracy_path = r'./results/houston/' #精度保存路径

print('Hyperspectral data shape: ', X.shape)
print('TR (Training labels) shape: ', TR.shape)
print('TE (Test labels) shape: ', TE.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

# Create image cubes for TR (used for pre-training AND fine-tuning)
print('\nCreating image cubes...')
print('Creating image cubes for TR (used for pre-training and fine-tuning)...')
X_TR, Y_TR = createImageCubes(X_pca, TR, windowSize=patch_size, removeZeroLabels=True)
print(f'  TR patches: {X_TR.shape[0]:,} samples')

# Create image cubes for TE (used for testing)
print('Creating image cubes for TE (used for testing)...')
X_TE, Y_TE = createImageCubes(X_pca, TE, windowSize=patch_size, removeZeroLabels=True)
print(f'  TE patches: {X_TE.shape[0]:,} samples')

#训练测试阶段
# Updated protocol (NO data leakage):
# - Pre-training: Uses 100% of X_TR (self-supervised, labels not used)
# - Fine-tuning: Samples few-shot from X_TR (with labels)
# - Testing: Uses 100% of X_TE
accuracy = train(X_TR, Y_TR, X_TE, Y_TE, class_num, N_RUNS,
      patch_size, pca_components, batch_size, device, EPOCH, train_path, fine_path, mask_ratio, encoder_depth, decoder_depth, seed)
print(accuracy)
with open(accuracy_path + str('acc') + '.csv', 'a', encoding='utf-8') as f:
      f.write(accuracy)