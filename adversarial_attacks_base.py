import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchattacks
from torchattacks import (
    FGSM, PGD, PGDL2, DeepFool, CW, AutoAttack,
    TPGD, FFGSM, MIFGSM, APGD, Square
)
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import math

class AdversarialAttackEvaluator:
    def __init__(self, model, device=None, normalize_fn=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize_fn = normalize_fn
        self.model.to(self.device)
        self.model.eval()
        self.available_attacks = {
            'fgsm': FGSM,
            'pgd': PGD,
            'pgdl2': PGDL2,
            'deepfool': DeepFool,
            'cw': CW,
            'autoattack': AutoAttack,
            'tpgd': TPGD,
            'ffgsm': FFGSM,
            'mifgsm': MIFGSM,
            'apgd': APGD,
            'square': Square
        }
        self.results = {}
        self.attack_samples = {}
        self.execution_times = {}
        self.perceptibility_metrics = {}
    
    def _preprocess_inputs(self, inputs):
        """Preprocess inputs if normalization function is provided."""
        if self.normalize_fn is not None:
            return self.normalize_fn(inputs)
        return inputs
    
    def evaluate_clean_accuracy(self, dataloader, num_samples=None):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating clean accuracy"):
                if num_samples is not None and total >= num_samples:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                images = self._preprocess_inputs(images)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        self.results['clean'] = {
            'accuracy': accuracy,
            'num_samples': total
        }
        print(f"Clean accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy
    
    def create_attack(self, attack_name, **kwargs):
        if attack_name.lower() not in self.available_attacks:
            raise ValueError(f"Attack {attack_name} not supported. Available attacks: {list(self.available_attacks.keys())}")
        attack_class = self.available_attacks[attack_name.lower()]
        return attack_class(self.model, **kwargs)
    
    def compute_perceptibility_metrics(self, clean_img, adv_img):
        if isinstance(clean_img, torch.Tensor):
            clean_np = clean_img.detach().cpu().numpy()
            adv_np = adv_img.detach().cpu().numpy()
        else:
            clean_np = clean_img
            adv_np = adv_img
        if clean_np.shape[0] == 3 or clean_np.shape[0] == 1:
            clean_np = np.transpose(clean_np, (1, 2, 0))
            adv_np = np.transpose(adv_np, (1, 2, 0))
        clean_np = np.clip(clean_np, 0, 1)
        adv_np = np.clip(adv_np, 0, 1)
        perturbation = adv_np - clean_np
        perturbation_flat = perturbation.flatten()
        l0_norm = np.sum(np.abs(perturbation_flat) > 1e-5)
        l1_norm = np.sum(np.abs(perturbation_flat))
        l2_norm = np.linalg.norm(perturbation_flat)
        linf_norm = np.max(np.abs(perturbation_flat))
        clean_uint8 = (clean_np * 255).astype(np.uint8)
        adv_uint8 = (adv_np * 255).astype(np.uint8)
        is_grayscale = False
        if len(clean_np.shape) == 2 or (len(clean_np.shape) == 3 and clean_np.shape[2] == 1):
            is_grayscale = True
            if len(clean_np.shape) == 3:
                clean_uint8 = clean_uint8.squeeze(-1)
                adv_uint8 = adv_uint8.squeeze(-1)
        ssim_value = None
        for win_size in [7, 5, 3]:
            if min(clean_uint8.shape[:2]) <= win_size:
                continue
            try:
                if is_grayscale:
                    ssim_value = ssim(clean_uint8, adv_uint8, 
                                     win_size=win_size, 
                                     data_range=255,
                                     multichannel=False)
                else:
                    ssim_value = ssim(clean_uint8, adv_uint8, 
                                     win_size=win_size,
                                     data_range=255,
                                     channel_axis=-1)
                if ssim_value is not None and not np.isnan(ssim_value):
                    break
            except Exception as e:
                continue
        if ssim_value is None or np.isnan(ssim_value):
            if is_grayscale:
                clean_flat = clean_np.flatten()
                adv_flat = adv_np.flatten()
            else:
                clean_flat = np.mean(clean_np, axis=2).flatten()
                adv_flat = np.mean(adv_np, axis=2).flatten()
            try:
                corr_coef = np.corrcoef(clean_flat, adv_flat)[0, 1]
                ssim_value = max(0, corr_coef)
            except:
                ssim_value = 0.0
        try:
            psnr_value = psnr(clean_uint8, adv_uint8, data_range=255)
            if np.isinf(psnr_value) or np.isnan(psnr_value):
                psnr_value = 100.0 if np.array_equal(clean_uint8, adv_uint8) else 0.0
        except:
            mse = np.mean((clean_np - adv_np) ** 2)
            if mse == 0:
                psnr_value = 100.0
            else:
                psnr_value = 10 * np.log10((1.0 ** 2) / mse)
        mse = np.mean((clean_np - adv_np) ** 2)
        pcp = (l0_norm / np.prod(clean_np.shape)) * 100
        try:
            if len(perturbation.shape) == 3:
                tv_norm = 0
                for c in range(perturbation.shape[2]):
                    tv_h = np.sum(np.abs(perturbation[1:, :, c] - perturbation[:-1, :, c]))
                    tv_w = np.sum(np.abs(perturbation[:, 1:, c] - perturbation[:, :-1, c]))
                    tv_norm += (tv_h + tv_w) / (perturbation.shape[0] * perturbation.shape[1])
                tv_norm /= perturbation.shape[2]
            else:
                tv_h = np.sum(np.abs(perturbation[1:, :] - perturbation[:-1, :]))
                tv_w = np.sum(np.abs(perturbation[:, 1:] - perturbation[:, :-1]))
                tv_norm = (tv_h + tv_w) / (perturbation.shape[0] * perturbation.shape[1])
        except:
            tv_norm = 0.0
        try:
            if len(clean_np.shape) == 3 and clean_np.shape[2] == 3:
                color_shift = np.mean(np.abs(perturbation), axis=(0, 1))
            else:
                color_shift = [np.mean(np.abs(perturbation))]
        except:
            color_shift = [0.0] * (3 if len(clean_np.shape) == 3 and clean_np.shape[2] == 3 else 1)
        metrics = {
            'l0_norm': float(l0_norm),
            'l0_percent': float(pcp),
            'l1_norm': float(l1_norm),
            'l2_norm': float(l2_norm),
            'linf_norm': float(linf_norm),
            'mse': float(mse),
            'ssim': float(ssim_value),
            'psnr': float(psnr_value),
            'tv_norm': float(tv_norm)
        }
        if len(color_shift) == 3:
            metrics['color_shift_r'] = float(color_shift[0])
            metrics['color_shift_g'] = float(color_shift[1])
            metrics['color_shift_b'] = float(color_shift[2])
        else:
            metrics['color_shift'] = float(color_shift[0])
        return metrics
    
    def evaluate_attack(self, attack, dataloader, attack_name=None, num_samples=None, save_samples=False, num_samples_to_save=5):
        if isinstance(attack, str):
            attack_name = attack
            attack = self.create_attack(attack)
        elif attack_name is None:
            attack_name = attack.__class__.__name__
        correct = 0
        total = 0
        saved_samples = []
        metrics_sum = None
        start_time = time.time()
        for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"Evaluating {attack_name}")):
            if num_samples is not None and total >= num_samples:
                break
            images, labels = images.to(self.device), labels.to(self.device)
            images = self._preprocess_inputs(images)
            adv_images = attack(images, labels)
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if i % max(1, len(dataloader) // 10) == 0:
                for j in range(min(5, images.size(0))):
                    metrics = self.compute_perceptibility_metrics(
                        images[j].detach().cpu(), 
                        adv_images[j].detach().cpu()
                    )
                    if metrics_sum is None:
                        metrics_sum = {k: v for k, v in metrics.items()}
                        metrics_count = 1
                    else:
                        for k, v in metrics.items():
                            if not np.isnan(v):
                                metrics_sum[k] += v
                        metrics_count += 1
            if save_samples and len(saved_samples) < num_samples_to_save:
                for j in range(min(images.size(0), num_samples_to_save - len(saved_samples))):
                    sample_metrics = self.compute_perceptibility_metrics(
                        images[j].detach().cpu(),
                        adv_images[j].detach().cpu()
                    )
                    saved_samples.append({
                        'clean': images[j].detach().cpu(),
                        'adversarial': adv_images[j].detach().cpu(),
                        'true_label': labels[j].item(),
                        'clean_pred': torch.max(self.model(images[j].unsqueeze(0)), 1)[1].item(),
                        'adv_pred': predicted[j].item(),
                        'metrics': sample_metrics
                    })
        end_time = time.time()
        execution_time = end_time - start_time
        avg_metrics = {k: v/metrics_count for k, v in metrics_sum.items()} if metrics_sum else {}
        accuracy = 100 * correct / total
        self.results[attack_name] = {
            'accuracy': accuracy,
            'num_samples': total
        }
        self.execution_times[attack_name] = execution_time
        self.perceptibility_metrics[attack_name] = avg_metrics
        if save_samples:
            self.attack_samples[attack_name] = saved_samples
        print(f"\n{attack_name} attack results:")
        print(f"  Accuracy under attack: {accuracy:.2f}% ({correct}/{total})")
        print(f"  Robustness (accuracy drop): {self.results['clean']['accuracy'] - accuracy:.2f}%")
        print(f"  Execution time: {execution_time:.2f} seconds")
        if avg_metrics:
            print("\n  Perceptibility Metrics (average):")
            print(f"    L0 norm: {avg_metrics['l0_norm']:.1f} pixels ({avg_metrics['l0_percent']:.2f}% of image)")
            print(f"    L1 norm: {avg_metrics['l1_norm']:.4f}")
            print(f"    L2 norm: {avg_metrics['l2_norm']:.4f}")
            print(f"    Linf norm: {avg_metrics['linf_norm']:.4f}")
            print(f"    SSIM: {avg_metrics['ssim']:.4f} (1.0 = identical)")
            print(f"    PSNR: {avg_metrics['psnr']:.2f} dB")
            print(f"    MSE: {avg_metrics['mse']:.6f}")
            print(f"    TV norm: {avg_metrics['tv_norm']:.6f}")
        return accuracy
    
    def run_standard_attacks(self, dataloader, num_samples=None, save_samples=True):
        print("Starting comprehensive adversarial attack evaluation...")
        if 'clean' not in self.results:
            self.evaluate_clean_accuracy(dataloader, num_samples)
        fgsm = self.create_attack('fgsm', eps=8/255)
        self.evaluate_attack(fgsm, dataloader, num_samples=num_samples, save_samples=save_samples)
        pgd = self.create_attack('pgd', eps=8/255, alpha=2/255, steps=10)
        self.evaluate_attack(pgd, dataloader, num_samples=num_samples, save_samples=save_samples)
        deepfool = self.create_attack('deepfool', steps=50)
        self.evaluate_attack(deepfool, dataloader, num_samples=num_samples, save_samples=save_samples)
        cw = self.create_attack('cw', c=1, lr=0.01, steps=100)
        self.evaluate_attack(cw, dataloader, num_samples=min(100, num_samples if num_samples else float('inf')), save_samples=save_samples)
        autoattack = self.create_attack('autoattack', eps=8/255, version='standard')
        self.evaluate_attack(autoattack, dataloader, num_samples=min(100, num_samples if num_samples else float('inf')), save_samples=save_samples)
        print("Completed standard attack suite.")
        return self.results
    
    def run_custom_attacks(self, dataloader, attacks_config, num_samples=None, save_samples=True):
        print("Starting custom adversarial attack evaluation...")
        if 'clean' not in self.results:
            self.evaluate_clean_accuracy(dataloader, num_samples)
        for attack_config in attacks_config:
            name = attack_config['name']
            params = attack_config.get('params', {})
            samples = attack_config.get('num_samples', num_samples)
            attack = self.create_attack(name, **params)
            self.evaluate_attack(attack, dataloader, attack_name=f"{name}_{hash(str(params))%1000}", 
                                num_samples=samples, save_samples=save_samples)
        print("Completed custom attack suite.")
        return self.results
    
    def visualize_adversarial_examples(self, attack_name, num_examples=5):
        """Visualize adversarial examples for a specific attack with perceptibility metrics."""
        if attack_name not in self.attack_samples:
            print(f"No saved samples for attack: {attack_name}")
            return
        samples = self.attack_samples[attack_name][:num_examples]
        fig, axes = plt.subplots(3, len(samples), figsize=(len(samples) * 3, 9))
        for i, sample in enumerate(samples):
            clean_img = sample['clean'].permute(1, 2, 0).numpy()
            clean_img = np.clip(clean_img, 0, 1)
            axes[0, i].imshow(clean_img)
            axes[0, i].set_title(f"Original\nTrue: {sample['true_label']}\nPred: {sample['clean_pred']}")
            axes[0, i].axis('off')
            adv_img = sample['adversarial'].permute(1, 2, 0).numpy()
            adv_img = np.clip(adv_img, 0, 1)
            axes[1, i].imshow(adv_img)
            axes[1, i].set_title(f"Adversarial\nTrue: {sample['true_label']}\nPred: {sample['adv_pred']}")
            axes[1, i].axis('off')
            clean = sample['clean'].permute(1, 2, 0).numpy()
            adv = sample['adversarial'].permute(1, 2, 0).numpy()
            perturbation = np.abs(adv - clean)
            perturbation = perturbation / perturbation.max() if perturbation.max() > 0 else perturbation
            axes[2, i].imshow(perturbation)
            if 'metrics' in sample:
                metrics = sample['metrics']
                axes[2, i].set_title(f"Perturbation\nL2: {metrics['l2_norm']:.3f}, L∞: {metrics['linf_norm']:.3f}\n"
                                   f"SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.1f}dB")
            else:
                axes[2, i].set_title(f"Perturbation\nL2: {np.linalg.norm(perturbation.flatten()):.4f}")
            axes[2, i].axis('off')
        plt.suptitle(f"Adversarial Examples: {attack_name}")
        plt.tight_layout()
        plt.show()
        if 'metrics' in samples[0]:
            metrics_data = []
            metrics_columns = ['Sample', 'L0 (pixels)', 'L0 (%)', 'L1', 'L2', 'L∞', 'SSIM', 'PSNR (dB)', 'MSE']
            for i, sample in enumerate(samples):
                metrics = sample['metrics']
                metrics_data.append([
                    f"Sample {i+1}",
                    f"{metrics['l0_norm']:.0f}",
                    f"{metrics['l0_percent']:.2f}%",
                    f"{metrics['l1_norm']:.4f}",
                    f"{metrics['l2_norm']:.4f}",
                    f"{metrics['linf_norm']:.4f}",
                    f"{metrics['ssim']:.4f}",
                    f"{metrics['psnr']:.2f}",
                    f"{metrics['mse']:.6f}"
                ])
            fig, ax = plt.subplots(figsize=(12, len(samples) * 0.5 + 1))
            ax.axis('off')
            table = ax.table(
                cellText=metrics_data,
                colLabels=metrics_columns,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            for j, cell in enumerate(table._cells[(0, j)] for j in range(len(metrics_columns))):
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
            plt.suptitle(f"Perceptibility Metrics: {attack_name}")
            plt.tight_layout()
            plt.show()
    
    def visualize_all_adversarial_examples(self, num_examples=3):
        for attack_name in self.attack_samples:
            self.visualize_adversarial_examples(attack_name, num_examples)
    
    def plot_accuracy_comparison(self):
        if not self.results:
            print("No results to plot.")
            return
        attack_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in attack_names]
        sorted_indices = np.argsort(accuracies)
        sorted_names = [attack_names[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]
        plt.figure(figsize=(10, 6))
        bars = plt.barh(sorted_names, sorted_accuracies, color='skyblue')
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f"{sorted_accuracies[i]:.2f}%", va='center')
        plt.xlabel('Accuracy (%)')
        plt.title('Model Accuracy Under Different Attacks')
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.show()
    
    def plot_execution_times(self):
        if not self.execution_times:
            print("No execution times to plot.")
            return
        attack_names = list(self.execution_times.keys())
        times = [self.execution_times[name] for name in attack_names]
        sorted_indices = np.argsort(times)
        sorted_names = [attack_names[i] for i in sorted_indices]
        sorted_times = [times[i] for i in sorted_indices]
        plt.figure(figsize=(10, 6))
        bars = plt.barh(sorted_names, sorted_times, color='salmon')
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{sorted_times[i]:.2f}s", va='center')
        plt.xlabel('Execution Time (seconds)')
        plt.title('Attack Execution Times')
        plt.tight_layout()
        plt.show()
    
    def generate_confusion_matrix(self, attack_name, dataloader, num_samples=None):
        """Generate confusion matrix for a specific attack."""
        if isinstance(attack_name, str) and attack_name not in self.available_attacks and attack_name != 'clean':
            print(f"Attack {attack_name} not found.")
            return
        y_true = []
        y_pred = []
        if attack_name == 'clean':
            with torch.no_grad():
                for images, labels in tqdm(dataloader, desc="Generating clean confusion matrix"):
                    if num_samples is not None and len(y_true) >= num_samples:
                        break
                    images, labels = images.to(self.device), labels.to(self.device)
                    images = self._preprocess_inputs(images)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    if num_samples is not None and len(y_true) >= num_samples:
                        y_true = y_true[:num_samples]
                        y_pred = y_pred[:num_samples]
                        break
        else:
            if isinstance(attack_name, str):
                attack = self.create_attack(attack_name)
            else:
                attack = attack_name
                attack_name = attack.__class__.__name__
            for images, labels in tqdm(dataloader, desc=f"Generating {attack_name} confusion matrix"):
                if num_samples is not None and len(y_true) >= num_samples:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                images = self._preprocess_inputs(images)
                adv_images = attack(images, labels)
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                if num_samples is not None and len(y_true) >= num_samples:
                    y_true = y_true[:num_samples]
                    y_pred = y_pred[:num_samples]
                    break
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {attack_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        print(f"Classification Report: {attack_name}")
        print(classification_report(y_true, y_pred))
        return cm
    
    def plot_perceptibility_metrics(self):
        if not self.perceptibility_metrics:
            print("No perceptibility metrics to plot.")
            return
        attack_names = [name for name in self.perceptibility_metrics.keys() if name != 'clean']
        if not attack_names:
            print("No attack perceptibility metrics to plot.")
            return
        metric_sets = [
            {
                'title': 'Distance Metrics (L1, L2, Linf)',
                'metrics': ['l1_norm', 'l2_norm', 'linf_norm'],
                'labels': ['L1 Norm', 'L2 Norm', 'L∞ Norm'],
                'colors': ['#1f77b4', '#ff7f0e', '#2ca02c']
            },
            {
                'title': 'Visual Quality Metrics',
                'metrics': ['ssim', 'psnr', 'mse'],
                'labels': ['SSIM (higher=better)', 'PSNR (dB, higher=better)', 'MSE (lower=better)'],
                'colors': ['#d62728', '#9467bd', '#8c564b']
            },
            {
                'title': 'Pixel Change Metrics',
                'metrics': ['l0_norm', 'l0_percent', 'tv_norm'],
                'labels': ['L0 (pixels changed)', 'Changed Pixels (%)', 'TV Norm (smoothness)'],
                'colors': ['#e377c2', '#7f7f7f', '#bcbd22']
            }
        ]
        for metric_set in metric_sets:
            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.25
            index = np.arange(len(attack_names))
            for i, (metric, label, color) in enumerate(zip(metric_set['metrics'], metric_set['labels'], metric_set['colors'])):
                values = []
                for attack in attack_names:
                    if metric in self.perceptibility_metrics[attack]:
                        values.append(self.perceptibility_metrics[attack][metric])
                    else:
                        values.append(np.nan)
                position = index + (i - 1) * bar_width
                if metric == 'ssim':
                    ax.bar(position, values, bar_width, label=label, color=color, alpha=0.7)
                elif metric == 'psnr':
                    ax.bar(position, [v/50 for v in values], bar_width, label=f"{label} (/50)", color=color, alpha=0.7)
                elif metric == 'mse':
                    ax.bar(position, [v*1000 for v in values], bar_width, label=f"{label} (×1000)", color=color, alpha=0.7)
                elif metric == 'l0_percent':
                    ax.bar(position, values, bar_width, label=label, color=color, alpha=0.7)
                else:
                    max_val = max([v for v in values if not np.isnan(v)]) if values else 1
                    normalized = [v/max_val for v in values]
                    ax.bar(position, normalized, bar_width, label=f"{label} (normalized)", color=color, alpha=0.7)
            ax.set_xlabel('Attack')
            ax.set_ylabel('Metric Value')
            ax.set_title(metric_set['title'])
            ax.set_xticks(index)
            ax.set_xticklabels(attack_names, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            plt.show()
    
    def generate_comprehensive_report(self):
        if not self.results:
            print("No results to report.")
            return
        print("\n" + "="*80)
        print("COMPREHENSIVE ADVERSARIAL ROBUSTNESS REPORT")
        print("="*80)
        if 'clean' in self.results:
            print(f"\nClean Accuracy: {self.results['clean']['accuracy']:.2f}%")
        print("\nAccuracy Under Different Attacks:")
        for attack, result in self.results.items():
            if attack == 'clean':
                continue
            print(f"  {attack}: {result['accuracy']:.2f}% (Drop: {self.results['clean']['accuracy'] - result['accuracy']:.2f}%)")
        if self.execution_times:
            print("\nAttack Execution Times:")
            for attack, time in self.execution_times.items():
                print(f"  {attack}: {time:.2f} seconds")
        if self.perceptibility_metrics:
            print("\nPerceptibility Metrics (averages):")
            metric_rows = []
            metric_names = []
            for attack, metrics in self.perceptibility_metrics.items():
                if attack != 'clean':
                    for metric in metrics:
                        if metric not in metric_names:
                            metric_names.append(metric)
            distance_metrics = ['l0_norm', 'l0_percent', 'l1_norm', 'l2_norm', 'linf_norm']
            quality_metrics = ['ssim', 'psnr', 'mse']
            other_metrics = [m for m in metric_names if m not in distance_metrics and m not in quality_metrics]
            sorted_metrics = distance_metrics + quality_metrics + other_metrics
            metric_names = [m for m in sorted_metrics if m in metric_names]
            header = "Attack".ljust(20)
            for metric in metric_names:
                if metric == 'l0_norm':
                    header += "L0 Norm".ljust(15)
                elif metric == 'l0_percent':
                    header += "L0 (%)".ljust(15)
                elif metric == 'l1_norm':
                    header += "L1 Norm".ljust(15)
                elif metric == 'l2_norm':
                    header += "L2 Norm".ljust(15) 
                elif metric == 'linf_norm':
                    header += "L∞ Norm".ljust(15)
                elif metric == 'ssim':
                    header += "SSIM".ljust(15)
                elif metric == 'psnr':
                    header += "PSNR (dB)".ljust(15)
                elif metric == 'mse':
                    header += "MSE".ljust(15)
                elif metric == 'tv_norm':
                    header += "TV Norm".ljust(15)
                else:
                    header += metric.ljust(15)
            print("  " + header)
            print("  " + "-" * len(header))
            for attack, metrics in self.perceptibility_metrics.items():
                if attack == 'clean':
                    continue
                row = attack.ljust(20)
                for metric in metric_names:
                    if metric in metrics:
                        value = metrics[metric]
                        if metric in ['l0_percent', 'ssim']:
                            row += f"{value:.4f}".ljust(15)
                        elif metric in ['psnr']:
                            row += f"{value:.2f}".ljust(15)
                        elif metric in ['mse', 'tv_norm']:
                            row += f"{value:.6f}".ljust(15)
                        else:
                            row += f"{value:.4f}".ljust(15)
                    else:
                        row += "N/A".ljust(15)
                print("  " + row)
        attack_accuracies = [result['accuracy'] for name, result in self.results.items() if name != 'clean']
        if attack_accuracies:
            print("\nSummary Statistics:")
            print(f"  Average accuracy under attack: {np.mean(attack_accuracies):.2f}%")
            print(f"  Worst-case accuracy: {np.min(attack_accuracies):.2f}%")
            print(f"  Best-case accuracy: {np.max(attack_accuracies):.2f}%")
            print(f"  Standard deviation: {np.std(attack_accuracies):.2f}%")
        if 'clean' in self.results and attack_accuracies:
            robustness_score = np.min(attack_accuracies) / self.results['clean']['accuracy']
            print(f"\nRobustness Score: {robustness_score:.4f}")
            if robustness_score > 0.8:
                robustness_level = "Excellent"
            elif robustness_score > 0.6:
                robustness_level = "Good"
            elif robustness_score > 0.4:
                robustness_level = "Moderate"
            elif robustness_score > 0.2:
                robustness_level = "Poor"
            else:
                robustness_level = "Very Poor"
            print(f"Robustness Level: {robustness_level}")
        print("\n" + "="*80)
        report_data = []
        for attack, result in self.results.items():
            row_data = {
                'Attack': attack,
                'Accuracy': result['accuracy'],
                'Accuracy Drop': self.results['clean']['accuracy'] - result['accuracy'] if attack != 'clean' else 0,
                'Execution Time': self.execution_times.get(attack, float('nan')),
                'Samples': result['num_samples']
            }
            if attack in self.perceptibility_metrics:
                for metric, value in self.perceptibility_metrics[attack].items():
                    row_data[metric] = value
            report_data.append(row_data)
        report_df = pd.DataFrame(report_data)
        display(report_df)
        self.plot_accuracy_comparison()
        self.plot_execution_times()
        if self.perceptibility_metrics:
            self.plot_perceptibility_metrics()
        return report_df
    
    def save_report_to_csv(self, filename="adversarial_robustness_report.csv"):
        if not self.results:
            print("No results to save.")
            return
        report_data = []
        for attack, result in self.results.items():
            row_data = {
                'Attack': attack,
                'Accuracy': result['accuracy'],
                'Accuracy Drop': self.results['clean']['accuracy'] - result['accuracy'] if attack != 'clean' else 0,
                'Execution Time': self.execution_times.get(attack, float('nan')),
                'Samples': result['num_samples']
            }
            if attack in self.perceptibility_metrics:
                for metric, value in self.perceptibility_metrics[attack].items():
                    row_data[metric] = value
            report_data.append(row_data)
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(filename, index=False)
        txt_filename = filename.replace('.csv', '_summary.txt')
        with open(txt_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE ADVERSARIAL ROBUSTNESS REPORT\n")
            f.write("="*80 + "\n\n")
            if 'clean' in self.results:
                f.write(f"Clean Accuracy: {self.results['clean']['accuracy']:.2f}%\n\n")
            f.write("Accuracy Under Different Attacks:\n")
            for attack, result in self.results.items():
                if attack == 'clean':
                    continue
                f.write(f"  {attack}: {result['accuracy']:.2f}% (Drop: {self.results['clean']['accuracy'] - result['accuracy']:.2f}%)\n")
            f.write("\nAttack Execution Times:\n")
            for attack, time in self.execution_times.items():
                f.write(f"  {attack}: {time:.2f} seconds\n")
            if self.perceptibility_metrics:
                f.write("\nPerceptibility Metrics (averages):\n")
                for attack, metrics in self.perceptibility_metrics.items():
                    if attack == 'clean':
                        continue
                    f.write(f"  {attack}:\n")
                    if 'l0_norm' in metrics:
                        f.write(f"    L0 norm: {metrics['l0_norm']:.1f} pixels ({metrics['l0_percent']:.2f}% of image)\n")
                    if 'l1_norm' in metrics:
                        f.write(f"    L1 norm: {metrics['l1_norm']:.4f}\n")
                    if 'l2_norm' in metrics:
                        f.write(f"    L2 norm: {metrics['l2_norm']:.4f}\n")
                    if 'linf_norm' in metrics:
                        f.write(f"    Linf norm: {metrics['linf_norm']:.4f}\n")
                    if 'ssim' in metrics:
                        f.write(f"    SSIM: {metrics['ssim']:.4f}\n")
                    if 'psnr' in metrics:
                        f.write(f"    PSNR: {metrics['psnr']:.2f} dB\n")
                    if 'mse' in metrics:
                        f.write(f"    MSE: {metrics['mse']:.6f}\n")
                    if 'tv_norm' in metrics:
                        f.write(f"    TV norm: {metrics['tv_norm']:.6f}\n")
                    f.write("\n")
            attack_accuracies = [result['accuracy'] for name, result in self.results.items() if name != 'clean']
            if attack_accuracies:
                f.write("\nSummary Statistics:\n")
                f.write(f"  Average accuracy under attack: {np.mean(attack_accuracies):.2f}%\n")
                f.write(f"  Worst-case accuracy: {np.min(attack_accuracies):.2f}%\n")
                f.write(f"  Best-case accuracy: {np.max(attack_accuracies):.2f}%\n")
                f.write(f"  Standard deviation: {np.std(attack_accuracies):.2f}%\n")
            if 'clean' in self.results and attack_accuracies:
                robustness_score = np.min(attack_accuracies) / self.results['clean']['accuracy']
                f.write(f"\nRobustness Score: {robustness_score:.4f}\n")
                if robustness_score > 0.8:
                    robustness_level = "Excellent"
                elif robustness_score > 0.6:
                    robustness_level = "Good"
                elif robustness_score > 0.4:
                    robustness_level = "Moderate"
                elif robustness_score > 0.2:
                    robustness_level = "Poor"
                else:
                    robustness_level = "Very Poor"
                f.write(f"Robustness Level: {robustness_level}\n")
        print(f"Report saved to {filename}")
        print(f"Human-readable summary saved to {txt_filename}")
        return report_df
def run_adversarial_evaluation(model, test_loader, num_samples=1000, save_results=True):
    evaluator = AdversarialAttackEvaluator(model)
    evaluator.evaluate_clean_accuracy(test_loader, num_samples)
    evaluator.run_standard_attacks(test_loader, num_samples)
    report_df = evaluator.generate_comprehensive_report()
    evaluator.visualize_all_adversarial_examples()
    if save_results:
        evaluator.save_report_to_csv()
    return evaluator
if __name__ == "__main__":
    import torch
    import torchvision.models as models
    from torchvision import datasets, transforms
    model = models.resnet18(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    evaluator = run_adversarial_evaluation(model, test_loader, num_samples=100)