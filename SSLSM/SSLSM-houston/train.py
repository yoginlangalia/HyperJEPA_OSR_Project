import torch
from utils import TrainDS,TestDS,kappa,weighted_f1_score
import os
from train_fine import train_fine
import numpy as np
import time

def train(X_TR, Y_TR, X_TE, Y_TE, class_num, N_RUNS,
          patch_size, pca_components, batch_size, device, EPOCH, path, fine_path, mask_ratio, encoder_depth, decoder_depth, seed):
    #定义初始存储
    text = ''
    # Store metrics for each samples_per_class configuration
    samples_per_class_list = [1, 5, 10]
    metrics_per_config = {n_samples: {'Kappa': [], 'OA': [], 'AA': [], 'F1': [], 'CA': np.zeros([class_num, N_RUNS]), 'time': []} 
                          for n_samples in samples_per_class_list}
    from net import vit, sslsm
    for run in range(N_RUNS):

        #定义模型
        v = vit.ViT(
            image_size=9,
            patch_size=3,
            num_classes=class_num,
            dim=patch_size ** 2,
            depth=encoder_depth,
            heads=8,
            mlp_dim=2048,
            bands=pca_components,  # 波段数
        )
        model = sslsm.SSLSM(
            encoder=v,
            num_classes=class_num,
            masking_ratio=mask_ratio,  # the paper recommended 75% masked patches
            decoder_dim=512,  # paper showed good results with just 512
            decoder_depth=decoder_depth,  # anywhere from 1 to 8
            bands = pca_components, #波段数
            dim=patch_size ** 2,
        ).to(device)
        
        # Pre-training: Use 100% of X_TR (NO data leakage)
        # X_TR is in shape (num_samples, patch_size, patch_size, pca_components)
        # Transpose for model input: (num_samples, pca_components, patch_size, patch_size)
        Xtrain_pretrain = X_TR.reshape(-1, patch_size, patch_size, pca_components)
        Xtrain_pretrain = Xtrain_pretrain.transpose(0, 3, 1, 2)

        # Create dummy labels for pretraining (self-supervised, class labels NOT used)
        Y_pretrain_dummy = np.zeros(len(Xtrain_pretrain))

        # 创建 trainloader - use 100% of TR patches for pretraining
        trainset = TrainDS(Xtrain_pretrain, Y_pretrain_dummy)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f"Pre-training with {len(Xtrain_pretrain):,} samples (100% of TR)")

        #定义优化函数
        opt = torch.optim.Adam(model.parameters(), lr = 3e-4)

        #初始化损失和时间
        epoch_loss = 0
        time1 = time.time()
        #进行迭代训练
        for epoch in range(EPOCH*2):
            for _,(data, label) in enumerate(train_loader):
                data = data.to(device)
                loss,_ = model(data)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss / len(train_loader)
        #计算训练时间
        time2 = time.time()
        train_time = (time2 - time1)
        #保存训练模型
        cnn_save_path = os.path.join(path)
        torch.save(model.state_dict(), cnn_save_path)

        print("patch_size为{}时的第{}次运行自训练阶段完成".format(patch_size, run))

        # 微调阶段 - Run 3 times with different samples per class: 1, 5, 10
        for samples_per_class in samples_per_class_list:
            print("\n=== Fine-tuning with {} samples per class ===".format(samples_per_class))
            
            # Create unique fine_path for each sample count
            fine_path_samples = fine_path.replace('.pkl', '_{}samples.pkl'.format(samples_per_class))
            
            conf_mat, fine_time, y_true_test, y_pred_test = train_fine(
                X_TR, Y_TR, X_TE, Y_TE, samples_per_class, class_num, run,
                patch_size, pca_components, batch_size, device,
                EPOCH, path, fine_path_samples, encoder_depth, decoder_depth, seed)

            #计算微调测试的精度
            text += ("\n=== Run {}: Results with {} samples per class ===\n".format(run+1, samples_per_class))
            m = 0
            for i in range(class_num):
                ca = conf_mat[i, i] / np.sum(conf_mat[i, :]) if np.sum(conf_mat[i, :]) > 0 else 0  #计算ca
                text += ("class:{:<2},{:.4f}\n".format(i, ca)) #将每一次的ca加入到text中
                m += ca #将多类的ca进行相加
                # Store per-class accuracy for this configuration
                metrics_per_config[samples_per_class]['CA'][i, run] = ca
            #计算Kappa
            K = kappa(conf_mat)
            #计算OA
            oa = np.trace(conf_mat) / np.sum(conf_mat)
            #计算F1分数（加权）
            f1 = weighted_f1_score(y_true_test, y_pred_test)
            #计算AA
            aa = m / class_num
            text += ('AA,{:.4f}\n kappa,{:.4f}\n OA,{:.4f}\n F1,{:.4f}\n fine_time,{:.4f}\n'.format(
                aa, K, oa, f1, fine_time))
            
            # Store metrics for this configuration
            metrics_per_config[samples_per_class]['Kappa'].append(K)
            metrics_per_config[samples_per_class]['OA'].append(oa)
            metrics_per_config[samples_per_class]['F1'].append(f1)
            metrics_per_config[samples_per_class]['AA'].append(aa)
            metrics_per_config[samples_per_class]['time'].append(train_time + fine_time)
        # Add total training time for this run
        text += ('pre_train_time,{:.4f}\n'.format(train_time))
    
    #统计多次训练的平均精度 for each samples_per_class configuration
    text += ("\n=== Average Results (across {} runs) ===\n".format(N_RUNS))
    for samples_per_class in samples_per_class_list:
        text += ("\n--- Average Results with {} samples per class ---\n".format(samples_per_class))
        config = metrics_per_config[samples_per_class]
        average_ca = np.sum(config['CA'], axis=1) / N_RUNS
        for i in range(class_num):
            text += ("total_ca:{:<2},{:.4f}\n".format(i, average_ca[i]))
        average_kappa = np.sum(config['Kappa']) / N_RUNS if len(config['Kappa']) > 0 else 0
        average_oa = np.sum(config['OA']) / N_RUNS if len(config['OA']) > 0 else 0
        average_aa = np.sum(config['AA']) / N_RUNS if len(config['AA']) > 0 else 0
        average_f1 = np.sum(config['F1']) / N_RUNS if len(config['F1']) > 0 else 0
        average_time = np.sum(config['time']) / N_RUNS if len(config['time']) > 0 else 0
        text += ('average_aa,{:.4f}\n average_kappa,{:.4f}\n average_oa,{:.4f}\n average_f1,{:.4f}\n average_time,{:.4f}\n'
            .format(average_aa, average_kappa, average_oa, average_f1, average_time))

    return text
