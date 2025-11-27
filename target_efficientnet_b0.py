import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from torchsummary import summary
import psutil
import gc
from tqdm.notebook import tqdm

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

def to_cpu_recursive(obj):
    """Recursively convert all CUDA tensors to Python-serializable values"""
    if isinstance(obj, torch.Tensor):
        obj_cpu = obj.cpu()
        if obj_cpu.numel() == 1:
            return obj_cpu.item()
        return obj_cpu.tolist()
    elif isinstance(obj, list):
        return [to_cpu_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_cpu_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, se_ratio=0.25, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_channels == out_channels and stride == 1
        
        expanded_channels = in_channels * expansion_factor
        if expansion_factor != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            )
        else:
            self.expand_conv = nn.Identity()
            
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                     stride=stride, padding=(kernel_size-1)//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        self.se = SEBlock(expanded_channels, reduction_ratio=int(1/se_ratio))
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x * binary_mask / keep_prob
        
    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            if self.drop_connect_rate > 0:
                x = self._drop_connect(x)
            x = x + identity
            
        return x


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(EfficientNetB0, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.blocks_config = [
            [1, 16, 1, 3, 1],
            [6, 24, 2, 3, 2],
            [6, 40, 2, 5, 2],
            [6, 80, 3, 3, 2],
            [6, 112, 3, 5, 1],
            [6, 192, 4, 5, 2],
            [6, 320, 1, 3, 1]
        ]
        
        self.blocks = nn.ModuleList()
        in_channels = 32
        
        for expansion_factor, channels, num_layers, kernel_size, stride in self.blocks_config:
            for i in range(num_layers):
                s = stride if i == 0 else 1
                self.blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=s,
                    expansion_factor=expansion_factor
                ))
                in_channels = channels
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.initial_conv(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr_history': [],
            'epochs': [],
            'batch_times': [],
            'epoch_times': [],
            'train_f1_scores': [],
            'val_f1_scores': [],
            'train_precision': [],
            'train_recall': [],
            'val_precision': [],
            'val_recall': [],
            'class_accuracies': {i: [] for i in range(10)},
            'memory_usage': [],
            'gradient_norms': [],
            'weight_norms': []
        }
        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_loss': float('inf'),
            'best_train_acc': 0.0,
            'best_epoch': 0,
            'best_f1_score': 0.0
        }
        self.epoch_start_time = None
        self.batch_start_time = None
        
    def start_epoch(self):
        self.epoch_start_time = time.time()
        
    def end_epoch(self):
        if self.epoch_start_time is not None:
            self.metrics['epoch_times'].append(time.time() - self.epoch_start_time)
        
    def start_batch(self):
        self.batch_start_time = time.time()
        
    def end_batch(self):
        if self.batch_start_time is not None:
            self.metrics['batch_times'].append(time.time() - self.batch_start_time)
    
    def update_lr(self, optimizer):
        self.metrics['lr_history'].append(optimizer.param_groups[0]['lr'])
        
    def update_memory_usage(self):
        self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / (1024 * 1024))
        
    def update_gradient_stats(self, model):
        total_grad_norm = 0.0
        total_weight_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm(2).item() ** 2
            total_weight_norm += param.norm(2).item() ** 2
        
        self.metrics['gradient_norms'].append(np.sqrt(total_grad_norm))
        self.metrics['weight_norms'].append(np.sqrt(total_weight_norm))
    
    def calculate_f1_score(self, y_true, y_pred):
        f1_scores = []
        precisions = []
        recalls = []
        
        for class_idx in range(10):
            true_positives = np.sum((y_true == class_idx) & (y_pred == class_idx))
            false_positives = np.sum((y_true != class_idx) & (y_pred == class_idx))
            false_negatives = np.sum((y_true == class_idx) & (y_pred != class_idx))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            
        return np.mean(f1_scores), np.mean(precisions), np.mean(recalls)
    
    def calculate_class_accuracy(self, y_true, y_pred):
        class_accuracies = {}
        for class_idx in range(10):
            mask = (y_true == class_idx)
            if np.sum(mask) > 0:
                class_accuracies[class_idx] = np.mean(y_pred[mask] == class_idx)
            else:
                class_accuracies[class_idx] = 0.0
        return class_accuracies
    
    def update_train_metrics(self, loss, accuracy, y_true, y_pred):
        self.metrics['train_loss'].append(loss)
        self.metrics['train_acc'].append(accuracy)
        
        f1, precision, recall = self.calculate_f1_score(y_true, y_pred)
        self.metrics['train_f1_scores'].append(f1)
        self.metrics['train_precision'].append(precision)
        self.metrics['train_recall'].append(recall)
        
        class_accuracies = self.calculate_class_accuracy(y_true, y_pred)
        for class_idx, acc in class_accuracies.items():
            self.metrics['class_accuracies'][class_idx].append(acc)
        
        if accuracy > self.best_metrics['best_train_acc']:
            self.best_metrics['best_train_acc'] = accuracy
    
    def update_val_metrics(self, loss, accuracy, y_true, y_pred, epoch):
        self.metrics['val_loss'].append(loss)
        self.metrics['val_acc'].append(accuracy)
        self.metrics['epochs'].append(epoch)
        
        f1, precision, recall = self.calculate_f1_score(y_true, y_pred)
        self.metrics['val_f1_scores'].append(f1)
        self.metrics['val_precision'].append(precision)
        self.metrics['val_recall'].append(recall)
        
        if accuracy > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = accuracy
            self.best_metrics['best_epoch'] = epoch
            
        if loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = loss
            
        if f1 > self.best_metrics['best_f1_score']:
            self.best_metrics['best_f1_score'] = f1
    

    def get_summary(self):
        cpu_metrics = to_cpu_recursive(self.metrics)
        cpu_best_metrics = to_cpu_recursive(self.best_metrics)
        
        summary = {
            'best_val_acc': cpu_best_metrics['best_val_acc'],
            'best_val_loss': cpu_best_metrics['best_val_loss'],
            'best_train_acc': cpu_best_metrics['best_train_acc'],
            'best_epoch': cpu_best_metrics['best_epoch'],
            'best_f1_score': cpu_best_metrics['best_f1_score'],
            'final_val_acc': cpu_metrics['val_acc'][-1] if cpu_metrics['val_acc'] else None,
            'final_train_acc': cpu_metrics['train_acc'][-1] if cpu_metrics['train_acc'] else None,
            'final_val_loss': cpu_metrics['val_loss'][-1] if cpu_metrics['val_loss'] else None,
            'final_train_loss': cpu_metrics['train_loss'][-1] if cpu_metrics['train_loss'] else None,
            'avg_epoch_time': np.mean(cpu_metrics['epoch_times']) if cpu_metrics['epoch_times'] else None,
            'avg_batch_time': np.mean(cpu_metrics['batch_times']) if cpu_metrics['batch_times'] else None,
            'total_training_time': sum(cpu_metrics['epoch_times']) if cpu_metrics['epoch_times'] else None,
            'final_learning_rate': cpu_metrics['lr_history'][-1] if cpu_metrics['lr_history'] else None,
            'peak_memory_usage': max(cpu_metrics['memory_usage']) if cpu_metrics['memory_usage'] else None
        }
        return summary
    
    def save_metrics(self, path):
        cpu_metrics = to_cpu_recursive(self.metrics)
        cpu_best_metrics = to_cpu_recursive(self.best_metrics)
        
        serializable_metrics = cpu_metrics.copy()
        if 'class_accuracies' in serializable_metrics:
            serializable_metrics['class_accuracies'] = {
                str(k): v for k, v in serializable_metrics['class_accuracies'].items()
            }
        
        def ensure_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, list):
                return [ensure_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: ensure_serializable(v) for k, v in obj.items()}
            else:
                return obj
        
        serializable_metrics = ensure_serializable(serializable_metrics)
        serializable_best_metrics = ensure_serializable(cpu_best_metrics)
        
        summary = to_cpu_recursive(self.get_summary())
        summary = ensure_serializable(summary)
        
        serializable_data = {
            'metrics': serializable_metrics,
            'best_metrics': serializable_best_metrics,
            'summary': summary
        }
        
        with open(path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        print(f"Metrics saved to {path}")

    def to_dataframe(self):
        cpu_metrics = to_cpu_recursive(self.metrics)
        
        df_metrics = pd.DataFrame({
            'epoch': cpu_metrics['epochs'],
            'train_loss': cpu_metrics['train_loss'],
            'val_loss': cpu_metrics['val_loss'],
            'train_acc': cpu_metrics['train_acc'],
            'val_acc': cpu_metrics['val_acc'],
            'learning_rate': cpu_metrics['lr_history'],
            'train_f1': cpu_metrics['train_f1_scores'],
            'val_f1': cpu_metrics['val_f1_scores'],
            'train_precision': cpu_metrics['train_precision'],
            'val_precision': cpu_metrics['val_precision'],
            'train_recall': cpu_metrics['train_recall'],
            'val_recall': cpu_metrics['val_recall'],
            'epoch_time': cpu_metrics['epoch_times'],
        })
        
        for class_idx in range(10):
            if len(cpu_metrics['class_accuracies'][class_idx]) > 0:
                df_metrics[f'class_{class_idx}_acc'] = cpu_metrics['class_accuracies'][class_idx]
                    
        return df_metrics


class EfficientNetB0Trainer:
    def __init__(self, train_loader, val_loader, test_loader, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 run_name=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model = EfficientNetB0(num_classes=10).to(device)
        
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"efficientnet_b0_run_{timestamp}"
        else:
            self.run_name = run_name
            
        self.target_model_dir = os.path.join('target_model_2') 
        self.target_result_dir = os.path.join('target model 2 training result') 
        os.makedirs(self.target_model_dir, exist_ok=True)
        os.makedirs(self.target_result_dir, exist_ok=True)
        
        self.metrics_tracker = MetricsTracker()
        
        self.test_predictions = None
        self.test_true_labels = None
        self.test_probabilities = None
        
    def train(self, epochs=200, lr=0.05, weight_decay=1e-4, patience=20, 
              scheduler_type='cosine', save_every=5, mixup_alpha=0.2):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(self.train_loader))
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        config = {
            "model": "EfficientNet B0",
            "dataset": "CIFAR-10",
            "epochs": epochs,
            "batch_size": next(iter(self.train_loader))[0].shape[0],
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "scheduler": scheduler_type,
            "patience": patience if scheduler_type == 'plateau' else None,
            "optimizer": "AdamW",
            "mixup_alpha": mixup_alpha,
            "device": str(self.device),
            "run_name": self.run_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(self.target_result_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        input_size = next(iter(self.train_loader))[0][0].shape
        try:
            model_summary = summary(self.model, input_size=input_size, device=str(self.device))
            with open(os.path.join(self.target_result_dir, 'model_architecture.txt'), 'w') as f:
                f.write(str(self.model))
        except Exception as e:
            print(f"Could not generate model summary: {e}")
        
        total_start_time = time.time()
        
        epoch_bar = tqdm(range(epochs), desc="Training Progress")
        
        for epoch in epoch_bar:
            self.metrics_tracker.start_epoch()
            
            train_loss, train_acc, train_true, train_pred = self._train_epoch(epoch, optimizer, criterion, mixup_alpha)
            
            val_loss, val_acc, val_true, val_pred = self.validate()
            
            self.metrics_tracker.update_train_metrics(train_loss, train_acc, train_true, train_pred)
            self.metrics_tracker.update_val_metrics(val_loss, val_acc, val_true, val_pred, epoch)
            self.metrics_tracker.update_lr(optimizer)
            self.metrics_tracker.update_memory_usage()
            self.metrics_tracker.end_epoch()
            
            epoch_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )
            
            if scheduler_type == 'plateau':
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_acc)
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            else:
                scheduler.step()            
            
            if (epoch + 1) % save_every == 0:
                self.save_model(f'efficientnet_b0_epoch_{epoch+1}.pth')
            
            if val_acc >= self.metrics_tracker.best_metrics['best_val_acc']:
                self.save_model('efficientnet_b0_best.pth')
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            metrics_df = self.metrics_tracker.to_dataframe()
            metrics_df.to_csv(os.path.join(self.target_result_dir, 'training_metrics.csv'), index=False)
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.save_training_plots()
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        total_time = time.time() - total_start_time
        print(f'Training completed in {total_time/60:.2f} minutes')
        
        self.metrics_tracker.save_metrics(os.path.join(self.target_result_dir, 'all_metrics.json'))
        
        self.save_training_plots()
        
        self.load_model('efficientnet_b0_best.pth')
        test_metrics = self.test()
        print(f'Test Accuracy: {test_metrics["accuracy"]:.2f}%')
        print(f'Test Loss: {test_metrics["loss"]:.4f}')
        print(f'Test F1 Score: {test_metrics["f1_score"]:.4f}')
        
        with open(os.path.join(self.target_result_dir, 'test_results.json'), 'w') as f:
            for k, v in test_metrics.items():
                if isinstance(v, (np.ndarray, np.generic)):
                    test_metrics[k] = v.tolist()
                elif isinstance(v, torch.Tensor):
                    test_metrics[k] = v.tolist()
            json.dump(test_metrics, f, indent=4)
            
        self.analyze_test_results()
        
        return self.model
    
    def _train_epoch(self, epoch, optimizer, criterion, mixup_alpha=0):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        
        batch_progress = tqdm(self.train_loader, leave=False, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (inputs, targets) in enumerate(batch_progress):
            self.metrics_tracker.start_batch()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if mixup_alpha > 0:
                inputs, targets_a, targets_b, lam = self._mixup_data(inputs, targets, mixup_alpha)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self._mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            batch_total = targets.size(0)
            
            if mixup_alpha > 0:
                batch_correct = (lam * predicted.eq(targets_a).sum().float() + 
                                (1 - lam) * predicted.eq(targets_b).sum().float())
                dominant_targets = targets_a if lam > 0.5 else targets_b
                all_targets.extend(dominant_targets.cpu().numpy())
            else:
                batch_correct = predicted.eq(targets).sum().item()
                all_targets.extend(targets.cpu().numpy())
                
            total += batch_total
            correct += batch_correct
            all_predictions.extend(predicted.cpu().numpy())
            
            batch_progress.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total if not mixup_alpha > 0 else 'N/A'
            })
            
            self.metrics_tracker.end_batch()
            self.metrics_tracker.update_gradient_stats(self.model)
            
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc, np.array(all_targets), np.array(all_predictions)
    
    def _mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, np.array(all_targets), np.array(all_predictions)
    
    def test(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                probabilities = F.softmax(outputs, dim=1)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        self.test_predictions = np.array(all_predictions)
        self.test_true_labels = np.array(all_true_labels)
        self.test_probabilities = np.array(all_probabilities)
        
        f1, precision, recall = self.metrics_tracker.calculate_f1_score(
            self.test_true_labels, self.test_predictions
        )
        
        class_accuracies = self.metrics_tracker.calculate_class_accuracy(
            self.test_true_labels, self.test_predictions
        )
        
        conf_matrix = confusion_matrix(self.test_true_labels, self.test_predictions)
        
        report = classification_report(self.test_true_labels, self.test_predictions, 
                                      target_names=CIFAR10_CLASSES, output_dict=True)
        
        test_metrics = {
            "accuracy": test_acc,
            "loss": test_loss,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "class_accuracies": class_accuracies,
            "confusion_matrix": conf_matrix,
            "classification_report": report
        }
        test_metrics = to_cpu_recursive(test_metrics)
        return test_metrics
    
    def analyze_test_results(self):
        if self.test_predictions is None or self.test_true_labels is None:
            _ = self.test()
            
        test_analysis_dir = os.path.join(self.target_result_dir, 'test_analysis')
        os.makedirs(test_analysis_dir, exist_ok=True)
        
        self.plot_confusion_matrix(os.path.join(test_analysis_dir, 'confusion_matrix.png'))
        
        self.plot_per_class_metrics(os.path.join(test_analysis_dir, 'per_class_accuracy.png'))
        
        self.plot_roc_curves(os.path.join(test_analysis_dir, 'roc_curves.png'))
        
        self.plot_precision_recall_curves(os.path.join(test_analysis_dir, 'precision_recall_curves.png'))
        
        self.analyze_prediction_confidence(os.path.join(test_analysis_dir, 'prediction_analysis.json'))
    
    def save_model(self, filename='efficientnet_b0.pth'):
        model_path = os.path.join(self.target_model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics_tracker.metrics,
            'best_metrics': self.metrics_tracker.best_metrics,
        }, model_path)
        
        if filename == 'efficientnet_b0_best.pth':
            checkpoint_path = os.path.join(self.target_model_dir, 'checkpoint.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'metrics': self.metrics_tracker.metrics,
                'best_metrics': self.metrics_tracker.best_metrics,
                'run_name': self.run_name,
            }, checkpoint_path)
    
    def load_model(self, filename='efficientnet_b0_best.pth'):
        model_path = os.path.join(self.target_model_dir, filename)
        checkpoint = torch.load(model_path)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        if 'best_metrics' in checkpoint:
            self.metrics_tracker.best_metrics = checkpoint['best_metrics']


    def save_training_plots(self):
        cpu_metrics = to_cpu_recursive(self.metrics_tracker.metrics)
        
        plots_dir = os.path.join(self.target_result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(cpu_metrics['train_loss'], label='Train Loss')
        plt.plot(cpu_metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(cpu_metrics['train_acc'], label='Train Accuracy')
        plt.plot(cpu_metrics['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(cpu_metrics['lr_history'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(cpu_metrics['train_f1_scores'], label='Train F1')
        plt.plot(cpu_metrics['val_f1_scores'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('F1 Score Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_metrics.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(cpu_metrics['train_precision'], label='Train Precision')
        plt.plot(cpu_metrics['val_precision'], label='Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.title('Precision Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(cpu_metrics['train_recall'], label='Train Recall')
        plt.plot(cpu_metrics['val_recall'], label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Recall Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'precision_recall.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(15, 10))
        for class_idx in range(10):
            if len(cpu_metrics['class_accuracies'][class_idx]) > 0:
                plt.plot(
                    cpu_metrics['class_accuracies'][class_idx],
                    label=f'Class {class_idx} ({CIFAR10_CLASSES[class_idx]})'
                )
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Per-Class Accuracy Evolution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'per_class_accuracy.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(cpu_metrics['gradient_norms'])
        plt.xlabel('Batch')
        plt.ylabel('Gradient L2 Norm')
        plt.title('Gradient Norm Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(cpu_metrics['weight_norms'])
        plt.xlabel('Batch')
        plt.ylabel('Weight L2 Norm')
        plt.title('Weight Norm Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'gradient_weight_norms.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cpu_metrics['memory_usage'])
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Evolution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'memory_usage.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(cpu_metrics['epoch_times'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Epoch Execution Times')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        batch_times = cpu_metrics['batch_times']
        if len(batch_times) > 100:
            step = len(batch_times) // 100
            sampled_batch_times = batch_times[::step]
            plt.plot(range(0, len(batch_times), step), sampled_batch_times)
        else:
            plt.plot(batch_times)
        plt.xlabel('Batch')
        plt.ylabel('Time (s)')
        plt.title('Batch Execution Times (Sampled)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'timing_analysis.png'), dpi=300)
        plt.close()

    def plot_confusion_matrix(self, save_path):
        if self.test_true_labels is None or self.test_predictions is None:
            print("Test results not available. Run test() first.")
            return
            
        test_true_labels = to_cpu_recursive(self.test_true_labels)
        test_predictions = to_cpu_recursive(self.test_predictions)
            
        plt.figure(figsize=(12, 10))
        conf_matrix = confusion_matrix(test_true_labels, test_predictions)
        
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()    
    
    def plot_per_class_metrics(self, save_path):
        if self.test_true_labels is None or self.test_predictions is None:
            print("Test results not available. Run test() first.")
            return
            
        test_true_labels = to_cpu_recursive(self.test_true_labels)
        test_predictions = to_cpu_recursive(self.test_predictions)
            
        report = classification_report(
            test_true_labels, 
            test_predictions, 
            target_names=CIFAR10_CLASSES,
            output_dict=True
        )    
        
        classes = []
        precision = []
        recall = []
        f1_score_vals = []
        
        for i, class_name in enumerate(CIFAR10_CLASSES):
            if class_name in report:
                classes.append(class_name)
                precision.append(report[class_name]['precision'])
                recall.append(report[class_name]['recall'])
                f1_score_vals.append(report[class_name]['f1-score'])
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1_score_vals, width, label='F1 Score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def plot_roc_curves(self, save_path):
        if self.test_true_labels is None or self.test_probabilities is None:
            print("Test results not available. Run test() first.")
            return
            
        test_true_labels = to_cpu_recursive(self.test_true_labels)
        test_probabilities = to_cpu_recursive(self.test_probabilities)
        
        if not isinstance(test_true_labels, np.ndarray):
            test_true_labels = np.array(test_true_labels)
        if not isinstance(test_probabilities, np.ndarray):
            test_probabilities = np.array(test_probabilities)
        
        plt.figure(figsize=(15, 10))
        
        for i in range(len(CIFAR10_CLASSES)):
            y_true = np.array(test_true_labels == i, dtype=int)
            y_score = test_probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, 
                tpr, 
                lw=2, 
                label=f'ROC curve for {CIFAR10_CLASSES[i]} (area = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()    
    
    def plot_precision_recall_curves(self, save_path):
        if self.test_true_labels is None or self.test_probabilities is None:
            print("Test results not available. Run test() first.")
            return
            
        test_true_labels = to_cpu_recursive(self.test_true_labels)
        test_probabilities = to_cpu_recursive(self.test_probabilities)
        
        if not isinstance(test_true_labels, np.ndarray):
            test_true_labels = np.array(test_true_labels)
        if not isinstance(test_probabilities, np.ndarray):
            test_probabilities = np.array(test_probabilities)
        
        plt.figure(figsize=(15, 10))
        
        for i in range(len(CIFAR10_CLASSES)):
            y_true = np.array(test_true_labels == i, dtype=int)
            y_score = test_probabilities[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)
            
            plt.plot(
                recall, 
                precision, 
                lw=2, 
                label=f'PR curve for {CIFAR10_CLASSES[i]} (AP = {avg_precision:.2f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def analyze_prediction_confidence(self, save_path):
        if self.test_true_labels is None or self.test_predictions is None or self.test_probabilities is None:
            print("Test results not available. Run test() first.")
            return
            
        test_true_labels = to_cpu_recursive(self.test_true_labels)
        test_predictions = to_cpu_recursive(self.test_predictions)
        test_probabilities = to_cpu_recursive(self.test_probabilities)
        
        if not isinstance(test_true_labels, np.ndarray):
            test_true_labels = np.array(test_true_labels)
        if not isinstance(test_predictions, np.ndarray):
            test_predictions = np.array(test_predictions)
        if not isinstance(test_probabilities, np.ndarray):
            test_probabilities = np.array(test_probabilities)
        
        predicted_probs = np.array([
            test_probabilities[i][pred] if isinstance(test_probabilities[i], list) else test_probabilities[i, pred]
            for i, pred in enumerate(test_predictions)
        ])
        
        correct_mask = (test_predictions == test_true_labels)
        incorrect_mask = ~correct_mask
        
        if np.any(correct_mask):
            correct_indices = np.argsort(predicted_probs[correct_mask])[::-1]
            correct_indices = np.arange(len(correct_mask))[correct_mask][correct_indices]
            correct_indices = correct_indices[:10]
        else:
            correct_indices = []
        
        if np.any(incorrect_mask):
            incorrect_indices = np.argsort(predicted_probs[incorrect_mask])[::-1]
            incorrect_indices = np.arange(len(incorrect_mask))[incorrect_mask][incorrect_indices]
            incorrect_indices = incorrect_indices[:10]
        else:
            incorrect_indices = []
        
        confidence_analysis = {
            "most_confident_correct": [],
            "most_confident_incorrect": []
        }
        
        for idx in correct_indices:
            confidence_analysis["most_confident_correct"].append({
                "true_label": int(test_true_labels[idx]),
                "true_class_name": CIFAR10_CLASSES[test_true_labels[idx]],
                "predicted_label": int(test_predictions[idx]),
                "predicted_class_name": CIFAR10_CLASSES[test_predictions[idx]],
                "confidence": float(predicted_probs[idx]),
                "all_probabilities": [float(p) for p in (test_probabilities[idx] if isinstance(test_probabilities[idx], list) else test_probabilities[idx])]
            })
        
        for idx in incorrect_indices:
            confidence_analysis["most_confident_incorrect"].append({
                "true_label": int(test_true_labels[idx]),
                "true_class_name": CIFAR10_CLASSES[test_true_labels[idx]],
                "predicted_label": int(test_predictions[idx]),
                "predicted_class_name": CIFAR10_CLASSES[test_predictions[idx]],
                "confidence": float(predicted_probs[idx]),
                "all_probabilities": [float(p) for p in (test_probabilities[idx] if isinstance(test_probabilities[idx], list) else test_probabilities[idx])]
            })
        
        with open(save_path, 'w') as f:
            json.dump(confidence_analysis, f, indent=4)


def train_robust_efficientnet_b0(train_loader, val_loader, test_loader, run_name=None, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainer = EfficientNetB0Trainer(train_loader, val_loader, test_loader, device, run_name)
    
    config = {
        'epochs': 200,
        'lr': 0.01,
        'weight_decay': 1e-4,
        'patience': 10,
        'scheduler_type': 'cosine',
        'save_every': 5,
        'mixup_alpha': 0.2
    }
    
    config.update(kwargs)
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    trained_model = trainer.train(**config)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {trainer.metrics_tracker.best_metrics['best_val_acc']:.2f}%")
    print(f"Model saved at: {os.path.join(trainer.target_model_dir, 'efficientnet_b0_best.pth')}")
    print(f"Training results saved at: {trainer.target_result_dir}")
    
    return trained_model


def train_target_model2_efficientnet_b0(train_loader, val_loader, test_loader, run_name, **kwargs):
    model = train_robust_efficientnet_b0(train_loader, val_loader, test_loader, run_name, **kwargs)
    return model


def visualize_model_predictions(model, test_loader, num_images=25, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(5, 5, images_so_far)
            ax.axis('off')
            ax.set_title(f'Pred: {CIFAR10_CLASSES[preds[j]]}\nTrue: {CIFAR10_CLASSES[labels[j]]}')
            
            inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            
            inp = np.clip(inp, 0, 1)
            
            ax.imshow(inp)
            
            if images_so_far == num_images:
                plt.tight_layout()
                return fig
                
    plt.tight_layout()
    return fig


def generate_training_report(result_dir):
    metrics_path = os.path.join(result_dir, 'all_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    test_path = os.path.join(result_dir, 'test_results.json')
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    df_metrics = pd.DataFrame({
        'epoch': list(range(1, len(metrics_data['metrics']['train_loss']) + 1)),
        'train_loss': metrics_data['metrics']['train_loss'],
        'val_loss': metrics_data['metrics']['val_loss'],
        'train_acc': metrics_data['metrics']['train_acc'],
        'val_acc': metrics_data['metrics']['val_acc'],
    })
    
    report = {
        'summary': metrics_data['summary'],
        'test_results': test_data,
        'metrics_df': df_metrics
    }
    
    return report