"""
GGCL Trainer - Guided Group Contrastive Learning for HSI Classification
Based on the IEEE TGRS 2024 paper by Li et al.

FIXES:
1. Uses odd-even spectral augmentation (HSI-appropriate)
2. Matches original training mechanics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools

from models import HSINet
from augment import augment_odd_even
from utils import AverageMeter
from dataloader import InfiniteDataLoader


class GGCLTrainer:
    """
    Guided Group Contrastive Learning Trainer
    
    Combines:
    1. Supervised cross-entropy loss on labeled data
    2. Contrastive loss via odd-even spectral augmentation
    3. Pseudo-labeling for unlabeled samples
    """
    
    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device
        
        # Extract parameters
        self.train_params = params.get('train', {})
        self.data_params = params.get('data', {})
        self.net_params = params.get('net', {})
        
        # Training settings
        self.epochs = self.train_params.get('epochs', 100)
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.temperature = self.train_params.get('temp', 0.5)
        self.eval_interval = self.train_params.get('eval_interval', 10)
        
        # Contrastive settings
        self.use_unlabel = self.train_params.get('use_unlabel', True)
        self.lambda_contra = self.train_params.get('lambda_contra', 0.1)
        
        # Initialize model, optimizer, losses
        self._build_model()
        
        # Best model tracking
        self.best_oa = 0
        self.best_metrics = {}
    
    def _build_model(self):
        """Initialize model, optimizer, and loss functions"""
        # Model
        self.net = HSINet(self.params).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Gradient clipping
        self.clip = 15
    
    def weight_func(self, epoch):
        """
        Weight for contrastive loss.
        Increases from 0 to lambda_contra over first half of training.
        """
        progress = min(1.0, epoch / max(1, self.epochs // 2))
        return self.lambda_contra * progress
    
    def unlabel_ratio_func(self, epoch):
        """
        How much unlabeled data to mix in.
        Increases from 0 to 1 over first half of training.
        """
        progress = min(1.0, epoch / max(1, self.epochs // 2))
        return progress
    
    def infoNCE_loss(self, A_vecs, B_vecs, temperature=10):
        """
        InfoNCE contrastive loss using diagonal only.
        Matches original GGCL implementation.
        """
        if A_vecs is None or B_vecs is None:
            return torch.tensor(0.0, device=self.device)
        
        # Normalize features
        A_vecs = F.normalize(A_vecs, p=2, dim=1)
        B_vecs = F.normalize(B_vecs, p=2, dim=1)
        
        # Compute similarity matrix with temperature
        matrix_logits = torch.matmul(A_vecs, B_vecs.T) * temperature
        
        # Softmax by dim=1
        matrix_softmax = torch.softmax(matrix_logits, dim=1)
        
        # Log and take diagonal mean
        matrix_log = -1 * torch.log(matrix_softmax + 1e-8)
        loss_nce = torch.mean(torch.diag(matrix_log))
        
        return loss_nce
    
    def train_epoch(self, labeled_loader, unlabeled_loader=None, epoch=0):
        """Train for one epoch"""
        self.net.train()
        
        loss_meter = AverageMeter()
        
        # Create infinite iterator for unlabeled data
        if unlabeled_loader is not None and self.use_unlabel:
            unlabel_iter = enumerate(itertools.cycle(unlabeled_loader))
        else:
            unlabel_iter = None
        
        # Get weights for this epoch
        contra_weight = self.weight_func(epoch)
        unlabel_ratio = self.unlabel_ratio_func(epoch)
        
        pbar = tqdm(labeled_loader, desc=f'Epoch {epoch+1}')
        
        for data, target in pbar:
            data = data.to(self.device)
            target = target.to(self.device)
            label_batch = data.size(0)
            
            # Mix in unlabeled data
            if unlabel_iter is not None and unlabel_ratio > 0:
                _, (unlabel_data, _) = next(unlabel_iter)
                unlabel_data = unlabel_data.to(self.device)
                
                # Take a portion based on unlabel_ratio
                unlabel_num = int(unlabel_ratio * unlabel_data.size(0))
                if unlabel_num > 0:
                    unlabel_data = unlabel_data[:unlabel_num]
                    unlabel_target = torch.full(
                        (unlabel_num,), -1, 
                        dtype=torch.long, device=self.device
                    )
                    
                    data = torch.cat([data, unlabel_data], dim=0)
                    target = torch.cat([target, unlabel_target], dim=0)
            
            # Apply odd-even spectral augmentation
            left_data, right_data = augment_odd_even(data)
            
            # Forward pass
            logits, feat_left, feat_right = self.net(data, left_data, right_data)
            
            # Contrastive loss (pass temperature directly, not inverted)
            loss_contra = self.infoNCE_loss(
                feat_left, feat_right, 
                temperature=self.temperature  # Fixed: was 1.0/temperature
            ) * contra_weight
            
            # Assign pseudo-labels to unlabeled samples
            if label_batch < target.size(0):
                target[label_batch:] = torch.argmax(logits[label_batch:], dim=1)
            
            # Cross-entropy loss (ignore_index=-1 handles unlabeled)
            loss_ce = nn.CrossEntropyLoss(ignore_index=-1)(logits, target) * (1 - contra_weight)
            
            # Combined loss
            loss = loss_contra + loss_ce
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
            self.optimizer.step()
            
            # Update meters
            loss_meter.update(loss.item())
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'w': f'{contra_weight:.3f}',
                'u': f'{unlabel_ratio:.2f}'
            })
        
        return loss_meter.avg
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.net.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                logits, _, _ = self.net(data, None, None)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds.tolist())
                all_labels.extend(target.numpy().tolist())
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
        
        oa = accuracy_score(all_labels, all_preds) * 100
        kappa = cohen_kappa_score(all_labels, all_preds) * 100
        
        cm = confusion_matrix(all_labels, all_preds)
        per_class = []
        for i in range(len(cm)):
            if np.sum(cm[i, :]) > 0:
                per_class.append(cm[i, i] / np.sum(cm[i, :]) * 100)
            else:
                per_class.append(0)
        
        # AA includes all classes (fixed bug)
        aa = np.mean(per_class)
        
        # F1 macro
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        return {
            'OA': oa,
            'AA': aa,
            'Kappa': kappa,
            'F1': f1,
            'per_class': np.array(per_class)
        }
    
    def train(self, labeled_loader, test_loader, unlabeled_loader=None, save_path=None):
        """
        Full training loop
        
        Returns:
            best metrics dict
        """
        print(f"\nStarting GGCL training for {self.epochs} epochs")
        print(f"  Lambda contrastive: {self.lambda_contra}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Use unlabeled: {self.use_unlabel}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            # Training
            train_loss = self.train_epoch(labeled_loader, unlabeled_loader, epoch)
            
            # Evaluation
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                metrics = self.evaluate(test_loader)
                
                print(f"\n[Epoch {epoch+1}/{self.epochs}]")
                print(f"  Loss: {train_loss:.4f}")
                print(f"  OA: {metrics['OA']:.2f}%, AA: {metrics['AA']:.2f}%, "
                      f"Kappa: {metrics['Kappa']:.2f}, F1: {metrics['F1']:.2f}%")
                
                if metrics['OA'] > self.best_oa:
                    self.best_oa = metrics['OA']
                    self.best_metrics = metrics.copy()
                    print(f"  *** New best model! ***")
                    
                    if save_path:
                        torch.save(self.net.state_dict(), save_path)
        
        # Final results
        print("\n" + "=" * 50)
        print("FINAL RESULTS (Best Model)")
        print("=" * 50)
        print(f"OA: {self.best_metrics['OA']:.2f}%")
        print(f"AA: {self.best_metrics['AA']:.2f}%")
        print(f"Kappa: {self.best_metrics['Kappa']:.2f}")
        print(f"F1: {self.best_metrics['F1']:.2f}%")
        
        return self.best_metrics
