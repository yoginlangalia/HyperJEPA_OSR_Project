import torch
from utils import TrainDS, TestDS, kappa, weighted_f1_score, samplePerClass
import os
from train_fine import train_fine
import numpy as np
import time

def train(X13_full, Y13_full, X18_TR, Y18_TR, X18_TE, Y18_TE, class_num, N_RUNS,
          patch_size, pca_components, batch_size, device, EPOCH, path, fine_path, mask_ratio, encoder_depth, decoder_depth, seed):
    """
    Cross-scene SSLSM training:
    - Pre-training: ONCE on 100% of Houston-13
    - Exp1 (Zero-shot): Fine-tune on H13, test on H18
    - Exp2 (Few-shot): Fine-tune on H18, test on H18
    """
    text = ''
    
    # Few-shot configurations
    samples_per_class_list = [1, 5, 10]
    
    # Metrics storage for Exp1 (Zero-shot) and Exp2 (Few-shot)
    exp1_metrics = {n: {'Kappa': [], 'OA': [], 'AA': [], 'F1': [], 'CA': np.zeros([class_num, N_RUNS]), 'time': []} 
                   for n in samples_per_class_list}
    exp2_metrics = {n: {'Kappa': [], 'OA': [], 'AA': [], 'F1': [], 'CA': np.zeros([class_num, N_RUNS]), 'time': []} 
                   for n in samples_per_class_list}
    
    from net import vit, sslsm
    
    for run in range(N_RUNS):
        print(f'\n{"="*60}')
        print(f'RUN {run+1}/{N_RUNS}')
        print(f'{"="*60}')
        
        # ========== Define Model ==========
        v = vit.ViT(
            image_size=9,
            patch_size=3,
            num_classes=class_num,
            dim=patch_size ** 2,
            depth=encoder_depth,
            heads=8,
            mlp_dim=2048,
            bands=pca_components,
        )
        model = sslsm.SSLSM(
            encoder=v,
            num_classes=class_num,
            masking_ratio=mask_ratio,
            decoder_dim=512,
            decoder_depth=decoder_depth,
            bands=pca_components,
            dim=patch_size ** 2,
        ).to(device)
        
        # ========== PRE-TRAINING: 100% of Houston-13 (ONCE) ==========
        print(f'\n=== PRE-TRAINING on Houston-13 (100%) ===')
        
        # Prepare pre-training data
        Xtrain_pretrain = X13_full.transpose(0, 3, 1, 2)
        Y_pretrain_dummy = np.zeros(len(Xtrain_pretrain))
        
        trainset = TrainDS(Xtrain_pretrain, Y_pretrain_dummy)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        print(f'Pre-training samples: {len(Xtrain_pretrain):,} | Total epochs: {EPOCH*2} | Batch size: {batch_size}')
        
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        epoch_loss = 0
        time1 = time.time()
        
        for epoch in range(EPOCH*2):
            for _, (data, label) in enumerate(train_loader):
                data = data.to(device)
                loss, _ = model(data)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item() / len(train_loader)
                del loss, data
            
            if (epoch + 1) % 20 == 0 or (epoch + 1) == EPOCH*2:
                elapsed_time = time.time() - time1
                print(f'  Epoch [{epoch+1}/{EPOCH*2}] - Time: {elapsed_time:.2f}s')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        time2 = time.time()
        pretrain_time = time2 - time1
        
        # Save pre-trained model
        torch.save(model.state_dict(), path)
        print(f'Pre-training completed in {pretrain_time:.2f}s')
        
        # ========== EXPERIMENT 1: Zero-Shot (Fine-tune on H13, Test on H18) ==========
        text += f'\n{"="*60}\n'
        text += f'EXPERIMENT 1: ZERO-SHOT TRANSFER (H13 -> H18)\n'
        text += f'{"="*60}\n'
        print(f'\n=== EXPERIMENT 1: Zero-Shot (Fine-tune on H13, Test on H18) ===')
        
        for samples_per_class in samples_per_class_list:
            print(f'\n--- Zero-Shot: {samples_per_class} samples/class from H13 ---')
            
            fine_path_exp = fine_path.replace('.pkl', f'_exp1_{samples_per_class}shot.pkl')
            
            # Fine-tune on H13, test on H18
            conf_mat, fine_time, y_true_test, y_pred_test = train_fine(
                X13_full, Y13_full,  # Source for fine-tuning: H13
                X18_TE, Y18_TE,      # Target for testing: H18
                samples_per_class, class_num, run,
                patch_size, pca_components, batch_size, device,
                EPOCH, path, fine_path_exp, encoder_depth, decoder_depth, seed)
            
            # Calculate metrics
            text += f'\n=== Exp1 Run {run+1}: {samples_per_class} samples/class (H13->H18) ===\n'
            m = 0
            for i in range(class_num):
                ca = conf_mat[i, i] / np.sum(conf_mat[i, :]) if np.sum(conf_mat[i, :]) > 0 else 0
                text += f'class:{i:<2},{ca:.4f}\n'
                m += ca
                exp1_metrics[samples_per_class]['CA'][i, run] = ca
            
            K = kappa(conf_mat)
            oa = np.trace(conf_mat) / np.sum(conf_mat)
            f1 = weighted_f1_score(y_true_test, y_pred_test)
            aa = m / class_num
            text += f'AA,{aa:.4f}\n kappa,{K:.4f}\n OA,{oa:.4f}\n F1,{f1:.4f}\n fine_time,{fine_time:.4f}\n'
            
            exp1_metrics[samples_per_class]['Kappa'].append(K)
            exp1_metrics[samples_per_class]['OA'].append(oa)
            exp1_metrics[samples_per_class]['F1'].append(f1)
            exp1_metrics[samples_per_class]['AA'].append(aa)
            exp1_metrics[samples_per_class]['time'].append(pretrain_time + fine_time)
        
        # ========== EXPERIMENT 2: Few-Shot (Fine-tune on H18, Test on H18) ==========
        text += f'\n{"="*60}\n'
        text += f'EXPERIMENT 2: FEW-SHOT TRANSFER (H18 -> H18)\n'
        text += f'{"="*60}\n'
        print(f'\n=== EXPERIMENT 2: Few-Shot (Fine-tune on H18, Test on H18) ===')
        
        for samples_per_class in samples_per_class_list:
            print(f'\n--- Few-Shot: {samples_per_class} samples/class from H18 ---')
            
            fine_path_exp = fine_path.replace('.pkl', f'_exp2_{samples_per_class}shot.pkl')
            
            # Fine-tune on H18 train, test on H18 test
            conf_mat, fine_time, y_true_test, y_pred_test = train_fine(
                X18_TR, Y18_TR,      # Source for fine-tuning: H18 train
                X18_TE, Y18_TE,      # Target for testing: H18 test
                samples_per_class, class_num, run,
                patch_size, pca_components, batch_size, device,
                EPOCH, path, fine_path_exp, encoder_depth, decoder_depth, seed)
            
            # Calculate metrics
            text += f'\n=== Exp2 Run {run+1}: {samples_per_class} samples/class (H18->H18) ===\n'
            m = 0
            for i in range(class_num):
                ca = conf_mat[i, i] / np.sum(conf_mat[i, :]) if np.sum(conf_mat[i, :]) > 0 else 0
                text += f'class:{i:<2},{ca:.4f}\n'
                m += ca
                exp2_metrics[samples_per_class]['CA'][i, run] = ca
            
            K = kappa(conf_mat)
            oa = np.trace(conf_mat) / np.sum(conf_mat)
            f1 = weighted_f1_score(y_true_test, y_pred_test)
            aa = m / class_num
            text += f'AA,{aa:.4f}\n kappa,{K:.4f}\n OA,{oa:.4f}\n F1,{f1:.4f}\n fine_time,{fine_time:.4f}\n'
            
            exp2_metrics[samples_per_class]['Kappa'].append(K)
            exp2_metrics[samples_per_class]['OA'].append(oa)
            exp2_metrics[samples_per_class]['F1'].append(f1)
            exp2_metrics[samples_per_class]['AA'].append(aa)
            exp2_metrics[samples_per_class]['time'].append(pretrain_time + fine_time)
        
        text += f'pre_train_time,{pretrain_time:.4f}\n'
    
    # ========== Average Results ==========
    text += f'\n{"="*60}\n'
    text += f'AVERAGE RESULTS (across {N_RUNS} runs)\n'
    text += f'{"="*60}\n'
    
    # Exp1 averages
    text += '\n--- EXPERIMENT 1: Zero-Shot (H13 -> H18) ---\n'
    for n in samples_per_class_list:
        text += f'\n{n} samples/class:\n'
        config = exp1_metrics[n]
        avg_ca = np.sum(config['CA'], axis=1) / N_RUNS
        for i in range(class_num):
            text += f'avg_ca:{i:<2},{avg_ca[i]:.4f}\n'
        text += f'avg_aa,{np.mean(config["AA"]):.4f}\n'
        text += f'avg_kappa,{np.mean(config["Kappa"]):.4f}\n'
        text += f'avg_oa,{np.mean(config["OA"]):.4f}\n'
        text += f'avg_f1,{np.mean(config["F1"]):.4f}\n'
        text += f'avg_time,{np.mean(config["time"]):.4f}\n'
    
    # Exp2 averages
    text += '\n--- EXPERIMENT 2: Few-Shot (H18 -> H18) ---\n'
    for n in samples_per_class_list:
        text += f'\n{n} samples/class:\n'
        config = exp2_metrics[n]
        avg_ca = np.sum(config['CA'], axis=1) / N_RUNS
        for i in range(class_num):
            text += f'avg_ca:{i:<2},{avg_ca[i]:.4f}\n'
        text += f'avg_aa,{np.mean(config["AA"]):.4f}\n'
        text += f'avg_kappa,{np.mean(config["Kappa"]):.4f}\n'
        text += f'avg_oa,{np.mean(config["OA"]):.4f}\n'
        text += f'avg_f1,{np.mean(config["F1"]):.4f}\n'
        text += f'avg_time,{np.mean(config["time"]):.4f}\n'
    
    return text
