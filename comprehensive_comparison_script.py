import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class QuantumEnsembleAttackWrapper:
    def __init__(self, ensemble_manager, target_class=None):
        self.ensemble_manager = ensemble_manager
        self.target_class = target_class
        self.name = "QuantumEnsemble"
        if target_class is not None:
            self.name += f"_Target{target_class}"
    
    def __call__(self, images, labels=None):
        self.ensemble_manager.eval()
        with torch.no_grad():
            adversarial_images, perturbation, _, _ = self.ensemble_manager(images, self.target_class)
        return adversarial_images


def tensor_to_serializable(obj):
    import torch
    import numpy as np
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [tensor_to_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except:
            return "UnserializableObject"


def compute_perceptibility_metrics(clean_img, adv_img):
    import torch
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
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
            with np.errstate(divide='ignore', invalid='ignore'):
                if is_grayscale:
                    ssim_value = ssim(
                        clean_uint8,
                        adv_uint8,
                        win_size=win_size,
                        data_range=255,
                        multichannel=False
                    )
                else:
                    ssim_value = ssim(
                        clean_uint8,
                        adv_uint8,
                        win_size=win_size,
                        data_range=255,
                        channel_axis=-1
                    )
            if ssim_value is not None and not np.isnan(ssim_value):
                break
        except Exception as e:
            continue
    if ssim_value is None or np.isnan(ssim_value):
        try:
            if is_grayscale:
                clean_flat = clean_np.flatten()
                adv_flat = adv_np.flatten()
            else:
                clean_flat = np.mean(clean_np, axis=2).flatten()
                adv_flat = np.mean(adv_np, axis=2).flatten()
            corr_coef = np.corrcoef(clean_flat, adv_flat)[0, 1]
            ssim_value = max(0, corr_coef)
        except:
            ssim_value = 1.0 if np.array_equal(clean_np, adv_np) else 0.9
    mse = np.mean((clean_np - adv_np) ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        if mse < 1e-10:
            psnr_value = 100.0
        else:
            try:
                psnr_value = psnr(clean_uint8, adv_uint8, data_range=255)
                if np.isinf(psnr_value) or np.isnan(psnr_value):
                    psnr_value = 10 * np.log10((255.0 ** 2) / max(mse * (255.0 ** 2), 1e-10))
            except:
                psnr_value = 10 * np.log10((255.0 ** 2) / max(mse * (255.0 ** 2), 1e-10))
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


def evaluate_standard_attacks(target_model, test_loader, num_samples=100, save_dir="standard_attacks_results"):
    import os
    import time
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    import torchattacks
    os.makedirs(save_dir, exist_ok=True)
    device = next(target_model.parameters()).device
    attacks = {
        'fgsm_eps4': torchattacks.FGSM(target_model, eps=4/255),
        'fgsm_eps8': torchattacks.FGSM(target_model, eps=8/255),
        'pgd_10steps': torchattacks.PGD(target_model, eps=8/255, alpha=2/255, steps=10),
        'pgd_40steps': torchattacks.PGD(target_model, eps=8/255, alpha=2/255, steps=40),
        'deepfool': torchattacks.DeepFool(target_model, steps=50),
    }
    results = {'clean': {}}
    print("Evaluating clean accuracy...")
    target_model.eval()
    correct = 0
    total = 0
    clean_samples = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            if total >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = target_model(images)
            _, predicted = torch.max(outputs, 1)
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()
            if len(clean_samples) < 20 and batch_idx % 5 == 0:
                for j in range(min(batch_size, 3)):
                    clean_samples.append({
                        'image': images[j].detach().cpu(),
                        'label': labels[j].item(),
                        'prediction': predicted[j].item()
                    })
    clean_accuracy = 100.0 * correct / total
    results['clean']['accuracy'] = clean_accuracy
    results['clean']['num_samples'] = total
    results['clean']['images'] = clean_samples
    print(f"Clean accuracy: {clean_accuracy:.2f}% ({correct}/{total})")
    for attack_name, attack in attacks.items():
        print(f"Evaluating {attack_name}...")
        results[attack_name] = {
            'accuracy': 0,
            'num_samples': 0,
            'execution_time': 0,
            'original_predictions': [],
            'adversarial_predictions': [],
            'perceptibility_metrics': {}
        }
        correct = 0
        total = 0
        perceptibility_metrics_sum = None
        metrics_count = 0
        samples = []
        start_time = time.time()
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            if total >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            with torch.no_grad():
                outputs = target_model(images)
                _, original_preds = torch.max(outputs, 1)
                results[attack_name]['original_predictions'].extend(original_preds.cpu().numpy().tolist())
            adv_images = attack(images, labels)
            with torch.no_grad():
                adv_outputs = target_model(adv_images)
                _, adv_preds = torch.max(adv_outputs, 1)
                results[attack_name]['adversarial_predictions'].extend(adv_preds.cpu().numpy().tolist())
                correct += (adv_preds == labels).sum().item()
                total += batch_size
            if total % 50 == 0 or batch_idx == 0:
                for j in range(min(3, batch_size)):
                    original_img = images[j].detach().cpu()
                    adv_img = adv_images[j].detach().cpu()
                    metrics = compute_perceptibility_metrics(original_img, adv_img)
                    if perceptibility_metrics_sum is None:
                        perceptibility_metrics_sum = {k: v for k, v in metrics.items()}
                        metrics_count = 1
                    else:
                        for k, v in metrics.items():
                            if not np.isnan(v):
                                perceptibility_metrics_sum[k] += v
                        metrics_count += 1
                    if len(samples) < 5:
                        samples.append({
                            'clean': original_img,
                            'adversarial': adv_img,
                            'true_label': labels[j].item(),
                            'original_pred': original_preds[j].item(),
                            'adv_pred': adv_preds[j].item(),
                            'metrics': metrics
                        })
        execution_time = time.time() - start_time
        accuracy = 100.0 * correct / total
        results[attack_name]['accuracy'] = accuracy
        results[attack_name]['num_samples'] = total
        results[attack_name]['execution_time'] = execution_time
        if perceptibility_metrics_sum:
            avg_metrics = {k: v / metrics_count for k, v in perceptibility_metrics_sum.items()}
            results[attack_name]['perceptibility_metrics'] = avg_metrics
        print(f"{attack_name} results:")
        print(f"  Accuracy under attack: {accuracy:.2f}% ({correct}/{total})")
        print(f"  Robustness (accuracy drop): {clean_accuracy - accuracy:.2f}%")
        print(f"  Execution time: {execution_time:.2f} seconds")
        if perceptibility_metrics_sum:
            print("\n  Perceptibility Metrics (average):")
            print(f"    L0 (changed pixels): {avg_metrics['l0_percent']:.2f}%")
            print(f"    L2 norm: {avg_metrics['l2_norm']:.6f}")
            print(f"    Linf norm: {avg_metrics['linf_norm']:.6f}")
            print(f"    SSIM: {avg_metrics['ssim']:.4f}")
            print(f"    PSNR: {avg_metrics['psnr']:.2f} dB")
        if samples:
            fig, axes = plt.subplots(3, len(samples), figsize=(len(samples) * 3, 9))
            for i, sample in enumerate(samples):
                clean_img = sample['clean'].permute(1, 2, 0).numpy()
                clean_img = np.clip(clean_img, 0, 1)
                axes[0, i].imshow(clean_img)
                axes[0, i].set_title(f"Original\nTrue: {sample['true_label']}\nPred: {sample['original_pred']}")
                axes[0, i].axis('off')
                adv_img = sample['adversarial'].permute(1, 2, 0).numpy()
                adv_img = np.clip(adv_img, 0, 1)
                axes[1, i].imshow(adv_img)
                axes[1, i].set_title(f"Adversarial\nPred: {sample['adv_pred']}")
                axes[1, i].axis('off')
                pert = np.abs(adv_img - clean_img)
                pert = pert / pert.max() if pert.max() > 0 else pert
                axes[2, i].imshow(pert)
                axes[2, i].set_title(f"Perturbation\nL2: {sample['metrics']['l2_norm']:.6f}\nSSIM: {sample['metrics']['ssim']:.4f}")
                axes[2, i].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{attack_name}_samples.png"))
            plt.close(fig)
    return results


def evaluate_quantum_ensemble(
    ensemble_manager,
    target_model,
    test_loader,
    num_samples=100,
    target_class=None,
    save_dir="quantum_ensemble_results"
):
    import os
    import time
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    os.makedirs(save_dir, exist_ok=True)
    device = next(ensemble_manager.parameters()).device
    attack_name = "QuantumEnsemble"
    if target_class is not None:
        attack_name += f"_Target{target_class}"
    ensemble_manager.eval()
    target_model.eval()
    results = {
        'accuracy': 0,
        'num_samples': 0,
        'execution_time': 0,
        'attack_success_rate': 0,
        'perceptibility_metrics': {},
        'class_performance': {}
    }
    for i in range(10):
        results['class_performance'][i] = {'total': 0, 'success': 0}
    correct = 0
    total = 0
    success_count = 0
    perceptibility_metrics_sum = None
    metrics_count = 0
    samples = []
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f"Evaluating {attack_name}")):
            if total >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            original_outputs = target_model(images)
            original_preds = torch.argmax(original_outputs, 1)
            adversarial_images, perturbation, member_outputs, _ = ensemble_manager(images, target_class)
            perturbation_size = torch.norm(perturbation, p=2, dim=(1, 2, 3)).mean().item()
            if perturbation_size < 1e-6:
                print(f"\nWarning: Very small perturbation detected ({perturbation_size:.8f}). This might indicate issues with the attack.")
            adversarial_outputs = target_model(adversarial_images)
            adversarial_preds = torch.argmax(adversarial_outputs, 1)
            if target_class is None:
                success = (original_preds != adversarial_preds)
            else:
                target_tensor = torch.full_like(adversarial_preds, target_class)
                success = (adversarial_preds == target_tensor)
            success_count += success.sum().item()
            correct += (adversarial_preds == labels).sum().item()
            total += batch_size
            for i in range(batch_size):
                label = labels[i].item()
                results['class_performance'][label]['total'] += 1
                if success[i].item():
                    results['class_performance'][label]['success'] += 1
            if batch_idx % 5 == 0 or batch_idx == 0:
                for j in range(min(3, batch_size)):
                    metrics = compute_perceptibility_metrics(
                        images[j].detach().cpu(),
                        adversarial_images[j].detach().cpu()
                    )
                    if perceptibility_metrics_sum is None:
                        perceptibility_metrics_sum = {k: v for k, v in metrics.items()}
                        metrics_count = 1
                    else:
                        for k, v in metrics.items():
                            if not np.isnan(v):
                                perceptibility_metrics_sum[k] += v
                        metrics_count += 1
                    if len(samples) < 5:
                        samples.append({
                            'clean': images[j].detach().cpu(),
                            'adversarial': adversarial_images[j].detach().cpu(),
                            'perturbation': perturbation[j].detach().cpu(),
                            'true_label': labels[j].item(),
                            'original_pred': original_preds[j].item(),
                            'adv_pred': adversarial_preds[j].item(),
                            'metrics': metrics
                        })
    execution_time = time.time() - start_time
    accuracy = 100.0 * correct / total
    attack_success_rate = 100.0 * success_count / total
    results['accuracy'] = accuracy
    results['num_samples'] = total
    results['execution_time'] = execution_time
    results['attack_success_rate'] = attack_success_rate
    if perceptibility_metrics_sum and metrics_count > 0:
        avg_metrics = {k: v / metrics_count for k, v in perceptibility_metrics_sum.items()}
        results['perceptibility_metrics'] = avg_metrics
    for label in results['class_performance']:
        class_stats = results['class_performance'][label]
        if class_stats['total'] > 0:
            class_stats['success_rate'] = 100.0 * class_stats['success'] / class_stats['total']
        else:
            class_stats['success_rate'] = 0.0
    print(f"\n{attack_name} results:")
    print(f"  Accuracy under attack: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Attack success rate: {attack_success_rate:.2f}%")
    print(f"  Execution time: {execution_time:.2f} seconds")
    if perceptibility_metrics_sum and metrics_count > 0:
        print("\nPerceptibility metrics:")
        print(f"  L0 norm: {avg_metrics['l0_norm']:.2f} pixels")
        print(f"  L0 percent: {avg_metrics['l0_percent']:.4f}%")
        print(f"  L1 norm: {avg_metrics['l1_norm']:.6f}")
        print(f"  L2 norm: {avg_metrics['l2_norm']:.6f}")
        print(f"  Linf norm: {avg_metrics['linf_norm']:.6f}")
        print(f"  SSIM: {avg_metrics['ssim']:.6f}")
        print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"  MSE: {avg_metrics['mse']:.8f}")
        print(f"  TV norm: {avg_metrics['tv_norm']:.6f}")
        if 'color_shift_r' in avg_metrics:
            print(
                f"  Color shift (R,G,B): ({avg_metrics['color_shift_r']:.6f}, "
                f"{avg_metrics['color_shift_g']:.6f}, {avg_metrics['color_shift_b']:.6f})"
            )
        elif 'color_shift' in avg_metrics:
            print(f"  Color shift: {avg_metrics['color_shift']:.6f}")
    if samples:
        for i, sample in enumerate(samples):
            vis_path = os.path.join(save_dir, f"sample_{i}_label_{sample['true_label']}.png")
            try:
                ensemble_manager.visualize_ensemble_attack(
                    sample['clean'],
                    sample['true_label'],
                    filename=vis_path,
                    target_class=target_class
                )
            except Exception as e:
                print(f"Warning: Could not use ensemble's built-in visualization. Using fallback. Error: {e}")
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                clean_img = sample['clean'].permute(1, 2, 0).numpy()
                clean_img = np.clip(clean_img, 0, 1)
                axes[0, 0].imshow(clean_img)
                axes[0, 0].set_title(f"Original\nTrue: {sample['true_label']}\nPred: {sample['original_pred']}")
                axes[0, 0].axis('off')
                adv_img = sample['adversarial'].permute(1, 2, 0).numpy()
                adv_img = np.clip(adv_img, 0, 1)
                axes[0, 1].imshow(adv_img)
                axes[0, 1].set_title(f"Adversarial\nPred: {sample['adv_pred']}")
                axes[0, 1].axis('off')
                pert = sample['perturbation'].permute(1, 2, 0).numpy()
                pert_max = np.max(np.abs(pert))
                if pert_max > 0:
                    pert_scaled = pert / pert_max
                    pert_scaled = np.clip(pert_scaled * 10 + 0.5, 0, 1)
                else:
                    pert_scaled = np.ones_like(pert) * 0.5
                axes[0, 2].imshow(pert_scaled)
                axes[0, 2].set_title(f"Perturbation (×10)\nL2: {sample['metrics']['l2_norm']:.6f}")
                axes[0, 2].axis('off')
                metrics_text = "\n".join([
                    f"L2 norm: {sample['metrics']['l2_norm']:.6f}",
                    f"L0 %: {sample['metrics']['l0_percent']:.4f}%",
                    f"Linf norm: {sample['metrics']['linf_norm']:.6f}",
                    f"MSE: {sample['metrics']['mse']:.8f}",
                    f"SSIM: {sample['metrics']['ssim']:.6f}",
                    f"PSNR: {sample['metrics']['psnr']:.2f} dB"
                ])
                axes[1, 0].text(
                    0.5,
                    0.5,
                    metrics_text,
                    ha='center',
                    va='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                axes[1, 0].axis('off')
                diff = np.abs(adv_img - clean_img)
                diff_max = np.max(diff)
                if diff_max > 0:
                    diff = diff / diff_max
                axes[1, 1].imshow(diff)
                axes[1, 1].set_title("Pixel Differences\n(normalized)")
                axes[1, 1].axis('off')
                attack_info = "\n".join([
                    f"Attack: {attack_name}",
                    f"Original Pred: {sample['original_pred']}",
                    f"Adversarial Pred: {sample['adv_pred']}",
                    f"Success: {'Yes' if sample['original_pred'] != sample['adv_pred'] else 'No'}"
                ])
                axes[1, 2].text(
                    0.5,
                    0.5,
                    attack_info,
                    ha='center',
                    va='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                axes[1, 2].axis('off')
                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close(fig)
    plt.figure(figsize=(10, 6))
    class_labels = list(results['class_performance'].keys())
    success_rates = [results['class_performance'][c]['success_rate'] for c in class_labels]
    plt.bar(class_labels, success_rates, color='royalblue')
    plt.xlabel('Class')
    plt.ylabel('Attack Success Rate (%)')
    plt.title(f'{attack_name} - Success Rate by Class')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    for i, rate in enumerate(success_rates):
        plt.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{attack_name}_class_success_rates.png"))
    plt.close()
    return results


def create_comparison_report(standard_results, quantum_results, save_dir="comparison_results"):
    os.makedirs(save_dir, exist_ok=True)
    comparison_data = []
    if 'clean' in standard_results:
        comparison_data.append({
            'Attack': 'Clean',
            'Accuracy': standard_results['clean']['accuracy'],
            'Accuracy Drop': 0.0,
            'Attack Success Rate': 0.0,
            'Execution Time': 0.0,
            'L2 Norm': 0.0,
            'L0 Percent': 0.0,
            'SSIM': 1.0,
            'PSNR': 100.0
        })
        clean_accuracy = standard_results['clean']['accuracy']
    else:
        clean_accuracy = 100.0
    for attack_name, attack_results in standard_results.items():
        if attack_name == 'clean':
            continue
        accuracy = attack_results['accuracy']
        accuracy_drop = clean_accuracy - accuracy
        execution_time = attack_results.get('execution_time', 0.0)
        perceptibility_metrics = attack_results.get('perceptibility_metrics', {})
        l2_norm = perceptibility_metrics.get('l2_norm', 0.0)
        l0_percent = perceptibility_metrics.get('l0_percent', 0.0)
        ssim_value = perceptibility_metrics.get('ssim', 1.0)
        psnr_value = perceptibility_metrics.get('psnr', 100.0)
        comparison_data.append({
            'Attack': attack_name,
            'Accuracy': accuracy,
            'Accuracy Drop': accuracy_drop,
            'Attack Success Rate': accuracy_drop,
            'Execution Time': execution_time,
            'L2 Norm': l2_norm,
            'L0 Percent': l0_percent,
            'SSIM': ssim_value,
            'PSNR': psnr_value
        })
    for attack_name, attack_results in quantum_results.items():
        quantum_suffix = attack_name if isinstance(attack_name, str) else ""
        accuracy = attack_results['accuracy']
        accuracy_drop = clean_accuracy - accuracy
        attack_success_rate = attack_results.get('attack_success_rate', accuracy_drop)
        execution_time = attack_results.get('execution_time', 0.0)
        perceptibility_metrics = attack_results.get('perceptibility_metrics', {})
        l2_norm = perceptibility_metrics.get('l2_norm', 0.0)
        l0_percent = perceptibility_metrics.get('l0_percent', 0.0)
        ssim_value = perceptibility_metrics.get('ssim', 1.0)
        psnr_value = perceptibility_metrics.get('psnr', 100.0)
        comparison_data.append({
            'Attack': f"QuantumEnsemble{quantum_suffix}",
            'Accuracy': accuracy,
            'Accuracy Drop': accuracy_drop,
            'Attack Success Rate': attack_success_rate,
            'Execution Time': execution_time,
            'L2 Norm': l2_norm,
            'L0 Percent': l0_percent,
            'SSIM': ssim_value,
            'PSNR': psnr_value
        })
    df = pd.DataFrame(comparison_data)
    df.to_csv(os.path.join(save_dir, "attacks_comparison.csv"), index=False)
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('Accuracy Drop', ascending=False)
    bars = plt.barh(
        df_sorted['Attack'],
        df_sorted['Accuracy Drop'],
        color=[
            'firebrick' if 'QuantumEnsemble' in attack else 'royalblue'
            for attack in df_sorted['Attack']
        ]
    )
    plt.xlabel('Accuracy Drop (%)')
    plt.title('Attack Effectiveness (Higher is Better)')
    plt.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{df_sorted['Accuracy Drop'].iloc[i]:.1f}%",
            va='center'
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_drop_comparison.png"))
    plt.close()
    plt.figure(figsize=(12, 8))
    df_sorted = df.sort_values('L2 Norm', ascending=True)
    df_sorted = df_sorted[df_sorted['Attack'] != 'Clean']
    bars = plt.barh(
        df_sorted['Attack'],
        df_sorted['L2 Norm'],
        color=[
            'firebrick' if 'QuantumEnsemble' in attack else 'royalblue'
            for attack in df_sorted['Attack']
        ]
    )
    plt.xlabel('L2 Norm (Lower is Better)')
    plt.title('Perturbation Magnitude')
    plt.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{df_sorted['L2 Norm'].iloc[i]:.6f}",
            va='center'
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "l2_norm_comparison.png"))
    plt.close()
    plt.figure(figsize=(10, 8))
    df_attacks = df[df['Attack'] != 'Clean']
    for i, row in df_attacks.iterrows():
        marker = 'X' if 'QuantumEnsemble' in row['Attack'] else 'o'
        color = 'firebrick' if 'QuantumEnsemble' in row['Attack'] else 'royalblue'
        size = 150 if 'QuantumEnsemble' in row['Attack'] else 80
        plt.scatter(
            row['L2 Norm'],
            row['Accuracy Drop'],
            label=row['Attack'],
            marker=marker,
            color=color,
            s=size
        )
        plt.annotate(
            row['Attack'],
            (row['L2 Norm'], row['Accuracy Drop']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    plt.xlabel('L2 Norm (Lower is Better)')
    plt.ylabel('Accuracy Drop (Higher is Better)')
    plt.title('Attack Effectiveness vs. Perturbation Size')
    plt.grid(True, alpha=0.3)
    if df_attacks['L2 Norm'].min() < 0.001:
        plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "effectiveness_vs_perturbation.png"))
    plt.close()
    plt.figure(figsize=(10, 10))
    selected_attacks = ['fgsm_eps8', 'pgd_10steps', 'deepfool']
    if 'QuantumEnsemble' in df['Attack'].values:
        selected_attacks.append('QuantumEnsemble')
    df_selected = df[df['Attack'].isin(selected_attacks)]
    metrics = ['Accuracy Drop', 'SSIM', 'PSNR', 'L2 Norm', 'L0 Percent']
    df_norm = pd.DataFrame()
    df_norm['Attack'] = df_selected['Attack']
    for metric in metrics:
        if metric in ['L2 Norm', 'L0 Percent']:
            if df_selected[metric].max() != df_selected[metric].min():
                df_norm[metric] = 1 - (
                    (df_selected[metric] - df_selected[metric].min())
                    / (df_selected[metric].max() - df_selected[metric].min())
                )
            else:
                df_norm[metric] = 1.0
        else:
            if df_selected[metric].max() != df_selected[metric].min():
                df_norm[metric] = (
                    df_selected[metric] - df_selected[metric].min()
                ) / (df_selected[metric].max() - df_selected[metric].min())
            else:
                df_norm[metric] = 1.0
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    for i, attack in enumerate(df_norm['Attack']):
        values = df_norm.loc[df_norm['Attack'] == attack, metrics].values.flatten().tolist()
        values += values[:1]
        color = 'firebrick' if 'QuantumEnsemble' in attack else 'royalblue'
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=attack, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Multi-Metric Attack Comparison", size=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "multi_metric_comparison.png"))
    plt.close()
    plt.figure(figsize=(10, 8))
    df_attacks = df[df['Attack'] != 'Clean']
    for i, row in df_attacks.iterrows():
        marker = 'X' if 'QuantumEnsemble' in row['Attack'] else 'o'
        color = 'firebrick' if 'QuantumEnsemble' in row['Attack'] else 'royalblue'
        size = 150 if 'QuantumEnsemble' in row['Attack'] else 80
        plt.scatter(
            row['SSIM'],
            row['PSNR'],
            label=row['Attack'],
            marker=marker,
            color=color,
            s=size
        )
        plt.annotate(
            row['Attack'],
            (row['SSIM'], row['PSNR']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    plt.xlabel('SSIM (Higher is Better)')
    plt.ylabel('PSNR (Higher is Better)')
    plt.title('Image Quality Metrics Comparison')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.9, 1.0 if df_attacks['SSIM'].max() <= 1.0 else 1.01)
    plt.ylim(df_attacks['PSNR'].min() * 0.9, df_attacks['PSNR'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quality_metrics_comparison.png"))
    plt.close()
    plt.figure(figsize=(14, 7))
    plt.axis('off')
    table_data = df[['Attack', 'Accuracy Drop', 'L2 Norm', 'SSIM', 'PSNR']].copy()
    table_data.columns = ['Attack', 'Effectiveness (%)', 'L2 Norm', 'SSIM', 'PSNR (dB)']
    table_data['Effectiveness (%)'] = table_data['Effectiveness (%)'].map('{:.2f}'.format)
    table_data['L2 Norm'] = table_data['L2 Norm'].map('{:.6f}'.format)
    table_data['SSIM'] = table_data['SSIM'].map('{:.4f}'.format)
    table_data['PSNR (dB)'] = table_data['PSNR (dB)'].map('{:.2f}'.format)
    table = plt.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    for idx, row in enumerate(table_data['Attack']):
        if 'QuantumEnsemble' in row:
            for j in range(len(table_data.columns)):
                cell = table._cells[(idx + 1, j)]
                cell.set_facecolor('mistyrose')
    for j in range(len(table_data.columns)):
        cell = table._cells[(0, j)]
        cell.set_facecolor('lightsteelblue')
        cell.set_text_props(weight='bold')
    plt.title('Adversarial Attacks Comparison Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_summary_table.png"))
    plt.close()
    print(f"Comparison report and visualizations saved to {save_dir}")
    return df


def run_comprehensive_evaluation(
    target_model,
    ensemble_manager,
    test_loader,
    num_samples=100,
    target_class=None,
    base_dir="adversarial_evaluation_results"
):
    os.makedirs(base_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Starting comprehensive adversarial attack evaluation on {num_samples} samples")
    print("=" * 80)
    print("\n1. Evaluating standard adversarial attacks...")
    standard_results = evaluate_standard_attacks(
        target_model=target_model,
        test_loader=test_loader,
        num_samples=num_samples,
        save_dir=os.path.join(base_dir, "standard_attacks")
    )
    print("\n2. Evaluating quantum ensemble attack (non-targeted)...")
    quantum_results_non_targeted = evaluate_quantum_ensemble(
        ensemble_manager=ensemble_manager,
        target_model=target_model,
        test_loader=test_loader,
        num_samples=num_samples,
        target_class=None,
        save_dir=os.path.join(base_dir, "quantum_non_targeted")
    )
    targeted_results = {}
    if target_class is not None:
        print(f"\n3. Evaluating quantum ensemble attack (targeted to class {target_class})...")
        quantum_results_targeted = evaluate_quantum_ensemble(
            ensemble_manager=ensemble_manager,
            target_model=target_model,
            test_loader=test_loader,
            num_samples=num_samples,
            target_class=target_class,
            save_dir=os.path.join(base_dir, "quantum_targeted")
        )
        targeted_results['quantum_targeted'] = quantum_results_targeted
    print("\n4. Creating comparison report for non-targeted attacks...")
    comparison_df_non_targeted = create_comparison_report(
        standard_results=standard_results,
        quantum_results={'': quantum_results_non_targeted},
        save_dir=os.path.join(base_dir, "comparison_non_targeted")
    )
    if target_class is not None:
        print(f"\n5. Creating comparison report for targeted attacks (class {target_class})...")
        comparison_df_targeted = create_comparison_report(
            standard_results=standard_results,
            quantum_results={f'_Target{target_class}': quantum_results_targeted},
            save_dir=os.path.join(base_dir, "comparison_targeted")
        )
    print("\n6. Creating comprehensive summary...")
    summary = {
        'standard_attacks': standard_results,
        'quantum_non_targeted': quantum_results_non_targeted
    }
    if target_class is not None:
        summary['quantum_targeted'] = quantum_results_targeted
    serializable_summary = tensor_to_serializable(summary)
    with open(os.path.join(base_dir, "comprehensive_summary.json"), 'w') as f:
        json.dump(serializable_summary, f, indent=2)
        json_summary = {}
        for k, v in summary.items():
            if isinstance(v, dict):
                json_summary[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        json_summary[k][sub_k] = {}
                        for sub_sub_k, sub_sub_v in sub_v.items():
                            if isinstance(sub_sub_v, dict):
                                json_summary[k][sub_k][sub_sub_k] = {
                                    sub_sub_sub_k: float(sub_sub_sub_v) if isinstance(
                                        sub_sub_sub_v,
                                        (int, float, np.number)
                                    ) else sub_sub_sub_v
                                    for sub_sub_sub_k, sub_sub_sub_v in sub_sub_v.items()
                                }
                            elif isinstance(sub_sub_v, np.ndarray):
                                json_summary[k][sub_k][sub_sub_k] = sub_sub_v.tolist()
                            elif isinstance(sub_sub_v, (int, float, np.number)):
                                json_summary[k][sub_k][sub_sub_k] = float(sub_sub_v)
                            else:
                                json_summary[k][sub_k][sub_sub_k] = sub_sub_v
                    elif isinstance(sub_v, np.ndarray):
                        json_summary[k][sub_k] = sub_v.tolist()
                    elif isinstance(sub_v, (int, float, np.number)):
                        json_summary[k][sub_k] = float(sub_v)
                    else:
                        json_summary[k][sub_k] = sub_v
            elif isinstance(v, np.ndarray):
                json_summary[k] = v.tolist()
            elif isinstance(v, (int, float, np.number)):
                json_summary[k] = float(v)
            else:
                json_summary[k] = v
        json.dump(json_summary, f, indent=2)
    print("\n" + "=" * 80)
    print(f"Comprehensive evaluation completed. Results saved to {base_dir}")
    print("=" * 80)
    return summary


def save_metrics_to_json(metrics, filename):
    import json
    serializable_metrics = tensor_to_serializable(metrics)
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
