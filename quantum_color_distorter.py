import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random


class QuantumCircuitCache:
    def __init__(self, circuit_func, cache_size=100):
        self.circuit_func = circuit_func
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        self.member_id = 0
    
    def __call__(self, features, weights):
        features_flat = features.detach().cpu().numpy().flatten()
        features_rounded = np.round(features_flat, 6)
        features_key = hash(str(features_rounded))
        
        weights_flat = weights.detach().cpu().flatten()
        weights_key = []
        
        if weights_flat.numel() > 0:
            indices = torch.linspace(0, weights_flat.numel() - 1, min(10, weights_flat.numel())).long()
            for idx in indices:
                weights_key.append(float(weights_flat[idx]))
        
        cache_key = (self.member_id, features_key, tuple(weights_key))
        
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        self.misses += 1
        result = self.circuit_func(features, weights)
        
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = result
        return result


def rgb_to_hsv(rgb):
    rgb = torch.clamp(rgb, 0, 1)
    
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    max_rgb, _ = torch.max(rgb, dim=-1)
    min_rgb, _ = torch.min(rgb, dim=-1)
    diff = max_rgb - min_rgb
    
    h = torch.zeros_like(max_rgb)
    
    r_max_mask = max_rgb == r
    g_max_mask = max_rgb == g
    b_max_mask = max_rgb == b
    
    h[r_max_mask] = (60 * ((g[r_max_mask] - b[r_max_mask]) / (diff[r_max_mask] + 1e-6)) + 360) % 360
    h[g_max_mask] = (60 * ((b[g_max_mask] - r[g_max_mask]) / (diff[g_max_mask] + 1e-6)) + 120) % 360
    h[b_max_mask] = (60 * ((r[b_max_mask] - g[b_max_mask]) / (diff[b_max_mask] + 1e-6)) + 240) % 360
    
    h = h / 360.0
    
    s = torch.zeros_like(max_rgb)
    non_zero_mask = max_rgb != 0
    s[non_zero_mask] = diff[non_zero_mask] / max_rgb[non_zero_mask]
    
    v = max_rgb
    
    hsv = torch.stack([h, s, v], dim=-1)
    
    return hsv


def rgb_to_lab(rgb):
    rgb = torch.clamp(rgb, 0, 1)
    
    rgb_to_xyz_matrix = torch.tensor(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        device=rgb.device,
    )
    
    mask = rgb > 0.04045
    rgb_linear = torch.zeros_like(rgb)
    rgb_linear[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb_linear[~mask] = rgb[~mask] / 12.92
    
    xyz = torch.tensordot(rgb_linear, rgb_to_xyz_matrix.t(), dims=1)
    
    xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device)
    xyz = xyz / xyz_ref
    
    mask = xyz > 0.008856
    xyz_f = torch.zeros_like(xyz)
    xyz_f[mask] = torch.pow(xyz[mask], 1 / 3)
    xyz_f[~mask] = 7.787 * xyz[~mask] + 16 / 116
    
    x, y, z = xyz_f[..., 0], xyz_f[..., 1], xyz_f[..., 2]
    
    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    lab = torch.stack([l, a, b], dim=-1)
    
    return lab


class QuantumColorDistorter(nn.Module):
    def __init__(self, n_qubits=3, n_layers=4, epsilon=0.1, device="cpu", seed=None):
        super(QuantumColorDistorter, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
                    
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epsilon = epsilon
        self.device = device
        
        self.n_features = 12
        
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        
        self.q_weights = nn.Parameter(0.01 * torch.randn(self.n_layers, self.n_qubits, 3))
        
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.channel_bias = nn.Parameter(torch.zeros(3))
        
        self.channel_amplifiers = nn.Parameter(torch.ones(3))
        
        q_circuit_func = qml.QNode(
            self.quantum_circuit,
            self.q_device,
            interface="torch",
            diff_method="adjoint",
        )
        
        self.q_circuit = QuantumCircuitCache(q_circuit_func, cache_size=200)
        
        self.feature_mean = nn.Parameter(torch.zeros(self.n_features), requires_grad=False)
        self.feature_std = nn.Parameter(torch.ones(self.n_features), requires_grad=False)
        
        self.calibration_samples = 0
        self.max_calibration_samples = 100
    
    def extract_color_features(self, images):
        batch_size = images.shape[0]
        features = torch.zeros(batch_size, self.n_features, device=self.device)
        
        for b in range(batch_size):
            img = images[b].reshape(3, -1)
            
            rgb_means = torch.mean(img, dim=1)
            
            centered_img = img - rgb_means.unsqueeze(1)
            
            cov_matrix = torch.mm(centered_img, centered_img.t()) / (img.shape[1] - 1 + 1e-8)
            
            cov_flat = cov_matrix.flatten()
            
            feature_vector = torch.cat([rgb_means, cov_flat])
            features[b] = feature_vector
        
        if self.calibration_samples < self.max_calibration_samples:
            with torch.no_grad():
                batch_mean = features.mean(dim=0)
                batch_std = features.std(dim=0, unbiased=False) if features.size(0) > 1 else torch.ones_like(features[0])
                batch_std = torch.clamp(batch_std, min=1e-6)
                
                if self.calibration_samples == 0:
                    self.feature_mean.copy_(batch_mean)
                    self.feature_std.copy_(batch_std)
                else:
                    alpha = 1.0 / (self.calibration_samples + 1)
                    self.feature_mean.copy_((1 - alpha) * self.feature_mean + alpha * batch_mean)
                    self.feature_std.copy_((1 - alpha) * self.feature_std + alpha * batch_std)
                
                self.calibration_samples += 1
        
        feature_mean = self.feature_mean.to(self.device)
        feature_std = self.feature_std.to(self.device)
        
        normalized_features = (features - feature_mean) / feature_std
        
        return normalized_features
    
    def quantum_circuit(self, features, weights):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            
            feature_idx = i * 4
            if feature_idx < len(features):
                qml.RZ(features[feature_idx] * np.pi, wires=i)
            
            feature_idx = i * 4 + 1
            if feature_idx < len(features):
                qml.RX(features[feature_idx] * np.pi, wires=i)
        
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(weights[l, i, 0], wires=i)
            
            qml.SWAP(wires=[0, 1])
            if self.n_qubits > 2:
                qml.SWAP(wires=[1, 2])
            
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for i in range(self.n_qubits):
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def generate_color_perturbation(self, q_output, original_shape):
        batch_size, channels, height, width = original_shape
        perturbation = torch.zeros(original_shape, device=self.device)
        
        for b in range(batch_size):
            rgb_noise = torch.zeros(3, height, width, device=self.device)
            
            for c in range(3):
                q_val = q_output[b, c % self.n_qubits]
                
                x_coords = torch.linspace(-1, 1, width, device=self.device)
                y_coords = torch.linspace(-1, 1, height, device=self.device)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
                
                angle = q_val.item() * np.pi
                gradient = torch.sin(
                    x_grid * torch.cos(torch.tensor(angle)) + y_grid * torch.sin(torch.tensor(angle))
                )
                
                rgb_noise[c] = gradient * self.channel_amplifiers[c]
            
            mixed_noise = torch.zeros_like(rgb_noise)
            for c_out in range(3):
                for c_in in range(3):
                    mixed_noise[c_out] += rgb_noise[c_in] * self.color_matrix[c_out, c_in]
                mixed_noise[c_out] += self.channel_bias[c_out]
            
            perturbation[b] = mixed_noise
        
        perturbation = self.epsilon * torch.tanh(perturbation)
        
        return perturbation
    
    def forward(self, images):
        batch_size = images.shape[0]
        original_shape = images.shape
        
        features = self.extract_color_features(images)
        
        q_outputs = torch.zeros((batch_size, self.n_qubits), device=self.device)
        
        q_outs = []
        for i in range(batch_size):
            q_out = self.q_circuit(features[i], self.q_weights)
            q_outs.append(torch.tensor(q_out, device=self.device))
        
        q_outputs = torch.stack(q_outs)
        
        perturbation = self.generate_color_perturbation(q_outputs, original_shape)
        
        adversarial_images = torch.clamp(images + perturbation, 0, 1)
        
        return perturbation, adversarial_images
    
    def visualize_perturbation(self, image, perturbation, adversarial, filename=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        with torch.no_grad():
            img_np = image.detach().permute(1, 2, 0).cpu().numpy()
            pert_np = perturbation.detach().permute(1, 2, 0).cpu().numpy()
            adv_np = adversarial.detach().permute(1, 2, 0).cpu().numpy()
        
        pert_amplified = np.clip(pert_np * 10, -1, 1)
        
        img_tensor = image.detach().permute(1, 2, 0)
        adv_tensor = adversarial.detach().permute(1, 2, 0)
        
        with torch.no_grad():
            img_lab_tensor = rgb_to_lab(img_tensor)
            adv_lab_tensor = rgb_to_lab(adv_tensor)
            
            delta_e_tensor = torch.sqrt(torch.sum((img_lab_tensor - adv_lab_tensor) ** 2, dim=-1))
            
            img_lab = img_lab_tensor.cpu().numpy()
            adv_lab = adv_lab_tensor.cpu().numpy()
            delta_e = delta_e_tensor.cpu().numpy()
        
        axes[0, 0].imshow(np.clip(img_np, 0, 1))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(np.clip(pert_amplified + 0.5, 0, 1))
        axes[0, 1].set_title("Color Perturbation (10x)")
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(np.clip(adv_np, 0, 1))
        axes[0, 2].set_title("Adversarial Image")
        axes[0, 2].axis("off")
        
        axes[1, 0].hist(img_np[:, :, 0].flatten(), bins=30, color="r", alpha=0.5, range=(0, 1))
        axes[1, 0].hist(img_np[:, :, 1].flatten(), bins=30, color="g", alpha=0.5, range=(0, 1))
        axes[1, 0].hist(img_np[:, :, 2].flatten(), bins=30, color="b", alpha=0.5, range=(0, 1))
        axes[1, 0].set_title("Original RGB Histogram")
        
        axes[1, 1].hist(adv_np[:, :, 0].flatten(), bins=30, color="r", alpha=0.5, range=(0, 1))
        axes[1, 1].hist(adv_np[:, :, 1].flatten(), bins=30, color="g", alpha=0.5, range=(0, 1))
        axes[1, 1].hist(adv_np[:, :, 2].flatten(), bins=30, color="b", alpha=0.5, range=(0, 1))
        axes[1, 1].set_title("Adversarial RGB Histogram")
        
        im = axes[1, 2].imshow(delta_e, cmap="viridis")
        axes[1, 2].set_title("Color Difference (ΔE)")
        axes[1, 2].axis("off")
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            
        plt.close(fig)
        del img_np, pert_np, adv_np, img_lab, adv_lab, delta_e
        del img_tensor, adv_tensor, img_lab_tensor, adv_lab_tensor, delta_e_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def visualize_colorspaces(self, image, adversarial, filename=None):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        with torch.no_grad():
            img_tensor = image.detach().permute(1, 2, 0)
            adv_tensor = adversarial.detach().permute(1, 2, 0)
            
            img_tensor = torch.clamp(img_tensor, 0, 1)
            adv_tensor = torch.clamp(adv_tensor, 0, 1)
            
            diff_rgb_tensor = torch.abs(adv_tensor - img_tensor)
            
            img_hsv_tensor = rgb_to_hsv(img_tensor)
            adv_hsv_tensor = rgb_to_hsv(adv_tensor)
            
            diff_h_tensor = torch.abs(torch.fmod(adv_hsv_tensor[..., 0] - img_hsv_tensor[..., 0] + 0.5, 1) - 0.5)
            
            img_lab_tensor = rgb_to_lab(img_tensor)
            adv_lab_tensor = rgb_to_lab(adv_tensor)
            
            img_np = img_tensor.cpu().numpy()
            adv_np = adv_tensor.cpu().numpy()
            diff_rgb = diff_rgb_tensor.cpu().numpy()
            
            img_hsv = img_hsv_tensor.cpu().numpy()
            adv_hsv = adv_hsv_tensor.cpu().numpy()
            diff_h = diff_h_tensor.cpu().numpy()
            
            img_lab = img_lab_tensor.cpu().numpy()
            adv_lab = adv_lab_tensor.cpu().numpy()
            
            diff_a = np.abs(adv_lab[..., 1] - img_lab[..., 1]) / 100
        
        def normalize_lab_channel(channel, min_val, max_val):
            return (channel - min_val) / (max_val - min_val)
        
        img_a = normalize_lab_channel(img_lab[..., 1], -100, 100)
        adv_a = normalize_lab_channel(adv_lab[..., 1], -100, 100)
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original (RGB)")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(adv_np)
        axes[0, 1].set_title("Adversarial (RGB)")
        axes[0, 1].axis("off")
        
        im_rgb = axes[0, 2].imshow(diff_rgb * 5)
        axes[0, 2].set_title("Difference (RGB) × 5")
        axes[0, 2].axis("off")
        plt.colorbar(im_rgb, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        axes[1, 0].imshow(img_hsv[..., 0], cmap="hsv")
        axes[1, 0].set_title("Original (Hue)")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(adv_hsv[..., 0], cmap="hsv")
        axes[1, 1].set_title("Adversarial (Hue)")
        axes[1, 1].axis("off")
        
        im_h = axes[1, 2].imshow(diff_h * 5, cmap="viridis")
        axes[1, 2].set_title("Difference (Hue) × 5")
        axes[1, 2].axis("off")
        plt.colorbar(im_h, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        axes[2, 0].imshow(img_a, cmap="RdYlGn_r")
        axes[2, 0].set_title("Original (a* channel)")
        axes[2, 0].axis("off")
        
        axes[2, 1].imshow(adv_a, cmap="RdYlGn_r")
        axes[2, 1].set_title("Adversarial (a* channel)")
        axes[2, 1].axis("off")
        
        im_a = axes[2, 2].imshow(diff_a * 5, cmap="viridis")
        axes[2, 2].set_title("Difference (a* channel) × 5")
        axes[2, 2].axis("off")
        plt.colorbar(im_a, ax=axes[2, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            
        plt.close(fig)
        del img_np, adv_np, diff_rgb, img_hsv, adv_hsv, diff_h, img_lab, adv_lab, diff_a
        del img_tensor, adv_tensor, diff_rgb_tensor, img_hsv_tensor, adv_hsv_tensor, diff_h_tensor, img_lab_tensor, adv_lab_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def test_quantum_color_distorter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_image = torch.rand(1, 3, 32, 32, device=device)
    
    qcd = QuantumColorDistorter(n_qubits=3, n_layers=4, epsilon=0.1, device=device)
    qcd.to(device)
    
    perturbation, adversarial_image = qcd(test_image)
    
    qcd.visualize_perturbation(
        test_image[0],
        perturbation[0],
        adversarial_image[0],
        "quantum_color_distorter_output.png",
    )
    
    qcd.visualize_colorspaces(
        test_image[0],
        adversarial_image[0],
        "quantum_color_spaces_analysis.png",
    )
    
    if hasattr(qcd.q_circuit, "hits"):
        print(f"Cache statistics - Hits: {qcd.q_circuit.hits}, Misses: {qcd.q_circuit.misses}")
        print(
            f"Cache hit rate: {qcd.q_circuit.hits/(qcd.q_circuit.hits+qcd.q_circuit.misses)*100:.2f}%"
        )
    
    print(
        f"Perturbation stats: Min={perturbation.min().item():.4f}, Max={perturbation.max().item():.4f}"
    )
    print(f"Perturbation L2 norm: {torch.norm(perturbation).item():.4f}")
    
    return qcd, test_image, perturbation, adversarial_image


if __name__ == "__main__":
    qcd, test_image, perturbation, adversarial_image = test_quantum_color_distorter()
