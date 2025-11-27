import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from math import pi
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


def pytorch_gaussian_blur(image, sigma=1.0):
    kernel_size = max(3, int(2 * int(4 * sigma + 0.5) + 1))
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    is_single_channel = False
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
        is_single_channel = True
    
    blurred = image.unsqueeze(0)
    
    if hasattr(F, 'gaussian_blur'):
        blurred = F.gaussian_blur(
            blurred,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma]
        )
    else:
        padding = kernel_size // 2
        
        x = torch.arange(-padding, padding + 1, dtype=torch.float32, device=image.device)
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).expand(blurred.size(1), 1, 1, kernel_size)
        blurred = F.conv2d(
            F.pad(blurred, (padding, padding, 0, 0), mode='reflect'),
            kernel_h,
            groups=blurred.size(1)
        )
        
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).expand(blurred.size(1), 1, kernel_size, 1)
        blurred = F.conv2d(
            F.pad(blurred, (0, 0, padding, padding), mode='reflect'),
            kernel_v,
            groups=blurred.size(1)
        )
    
    blurred = blurred.squeeze(0)
    
    if is_single_channel:
        blurred = blurred.squeeze(0)
    
    return blurred


def pytorch_gabor_kernel(frequency, theta, sigma_x, sigma_y, n_stds=3, size=None):
    theta = torch.tensor(theta, dtype=torch.float32)
    frequency = torch.tensor(frequency, dtype=torch.float32)

    if size is None:
        height = int(sigma_y * n_stds) * 2 + 1
        width = int(sigma_x * n_stds) * 2 + 1
    else:
        height, width = size
    
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1
    
    y, x = torch.meshgrid(
        torch.arange(-(height // 2), height // 2 + 1, dtype=torch.float32),
        torch.arange(-(width // 2), width // 2 + 1, dtype=torch.float32),
        indexing='ij'
    )
    
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
    
    envelope = torch.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    )
    kernel = envelope * torch.cos(2 * torch.pi * frequency * x_theta)
    
    kernel = kernel / torch.max(torch.abs(kernel))
    
    return kernel


def apply_gabor_filters(image, kernels, device="cpu"):
    channels, height, width = image.shape
    responses = []
    
    for kernel in kernels:
        kernel_tensor = kernel.to(device).unsqueeze(0).unsqueeze(0)
        
        channel_responses = []
        for c in range(channels):
            img_channel = image[c:c+1].unsqueeze(0)
            
            padded = F.pad(
                img_channel,
                (kernel.shape[1]//2, kernel.shape[1]//2,
                 kernel.shape[0]//2, kernel.shape[0]//2),
                mode='circular'
            )
            filtered = F.conv2d(padded, kernel_tensor)
            
            response = torch.mean(torch.abs(filtered))
            channel_responses.append(response.item())
        
        responses.append(np.mean(channel_responses))
    
    return responses


class QuantumTextureAttacker(nn.Module):
    def __init__(self, n_qubits=4, n_layers=5, epsilon=0.1, scales=3, orientations=4, device="cpu", seed=None):
        super(QuantumTextureAttacker, self).__init__()
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
        self.scales = scales
        self.orientations = orientations
        
        self.n_features = scales * orientations
        
        self._initialize_gabor_kernels()
        
        self.q_device = qml.device("default.qubit", wires=self.n_qubits)
        
        self.n_conv_layers = 2
        self.n_pooling_layers = 3
        
        self.q_weights = nn.Parameter(0.01 * torch.randn(self.n_layers, self.n_qubits, 3))
        
        self.channel_weights = nn.Parameter(torch.ones(3, self.n_features) / self.n_features)
        
        self.output_scale = nn.Parameter(torch.tensor([0.1]))
        
        q_circuit_func = qml.QNode(
            self.quantum_circuit,
            self.q_device,
            interface="torch",
            diff_method="adjoint"
        )
        
        self.q_circuit = QuantumCircuitCache(q_circuit_func, cache_size=200)
        
        self.feature_mean = nn.Parameter(torch.zeros(self.n_features), requires_grad=False)
        self.feature_std = nn.Parameter(torch.ones(self.n_features), requires_grad=False)
        
        self.calibration_samples = 0
        self.max_calibration_samples = 100
    
    def _initialize_gabor_kernels(self):
        kernels = []
        pytorch_gabor_kernels = []
        
        for scale in range(1, self.scales + 1):
            sigma = 3.0 * scale
            frequency = 0.15 / scale
            
            for theta in range(self.orientations):
                angle = theta * np.pi / self.orientations
                
                kernel = pytorch_gabor_kernel(
                    frequency=frequency,
                    theta=angle,
                    sigma_x=sigma,
                    sigma_y=sigma,
                    n_stds=3,
                    size=(16, 16)
                )
                
                pytorch_gabor_kernels.append(kernel)
                
                kernel_np = kernel.cpu().numpy()
                kernels.append(kernel_np)
        
        self.gabor_kernels = kernels
        self.pytorch_gabor_kernels = pytorch_gabor_kernels
    
    def extract_texture_features(self, images):
        batch_size = images.shape[0]
        features = torch.zeros(batch_size, self.n_features, device=self.device)
        
        for b in range(batch_size):
            img = images[b]
            responses = apply_gabor_filters(img, self.pytorch_gabor_kernels, self.device)
            features[b] = torch.tensor(responses, device=self.device)
        
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
            feature_idx = i % len(features)
            qml.RX(features[feature_idx], wires=i)
            qml.RY(features[(feature_idx + 1) % len(features)], wires=i)
        
        for l in range(self.n_conv_layers):
            layer_idx = l
            
            for i in range(self.n_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
            
            for i in range(self.n_qubits):
                qml.CZ(wires=[i, (i + 1) % self.n_qubits])
        
        for l in range(self.n_pooling_layers):
            layer_idx = l + self.n_conv_layers
            
            for i in range(self.n_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
            
            target = l % self.n_qubits
            for i in range(self.n_qubits):
                if i != target:
                    qml.CNOT(wires=[i, target])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def generate_texture_perturbation(self, q_output, original_shape):
        batch_size, channels, height, width = original_shape
        perturbation = torch.zeros(original_shape, device=self.device)
        
        for b in range(batch_size):
            for c in range(3):
                texture_noise = torch.randn(height, width, device=self.device)
                
                for i, q_val in enumerate(q_output[b]):
                    freq = (i + 1) * pi / 8
                    phase = q_val.item() * pi
                    
                    x_coords = torch.linspace(0, freq * 2, width, device=self.device)
                    y_coords = torch.linspace(0, freq * 2, height, device=self.device)
                    
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    modulation = torch.sin(x_grid + phase) * torch.cos(y_grid + phase * 0.5)
                    
                    texture_noise += self.channel_weights[c, i % self.n_features] * modulation
                
                smoothed_noise = pytorch_gaussian_blur(texture_noise, sigma=1.0)
                
                perturbation[b, c] = smoothed_noise * self.output_scale
        
        perturbation = self.epsilon * torch.tanh(perturbation)
        
        return perturbation
    
    def forward(self, images):
        batch_size = images.shape[0]
        original_shape = images.shape
        
        features = self.extract_texture_features(images)
        
        q_outputs = torch.zeros((batch_size, self.n_qubits), device=self.device)
        
        q_outs = []
        for i in range(batch_size):
            q_out = self.q_circuit(features[i], self.q_weights)
            q_outs.append(torch.tensor(q_out, device=self.device))

        q_outputs = torch.stack(q_outs)
        
        perturbation = self.generate_texture_perturbation(q_outputs, original_shape)
        
        adversarial_images = torch.clamp(images + perturbation, 0, 1)
        
        return perturbation, adversarial_images
    
    def visualize_perturbation(self, image, perturbation, adversarial, filename=None):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        with torch.no_grad():
            img_np = image.detach().permute(1, 2, 0).cpu().numpy()
            pert_np = perturbation.detach().permute(1, 2, 0).cpu().numpy()
            adv_np = adversarial.detach().permute(1, 2, 0).cpu().numpy()
        
        pert_amplified = np.clip(pert_np * 10, -1, 1)
        
        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(np.clip(pert_np + 0.5, 0, 1))
        axes[1].set_title("Texture Perturbation")
        axes[1].axis("off")
        
        axes[2].imshow(np.clip(pert_amplified + 0.5, 0, 1))
        axes[2].set_title("Perturbation (10x)")
        axes[2].axis("off")
        
        axes[3].imshow(np.clip(adv_np, 0, 1))
        axes[3].set_title("Adversarial Image")
        axes[3].axis("off")
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        plt.close(fig)
        del img_np, pert_np, adv_np, pert_amplified
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def visualize_gabor_responses(self, image, filename=None):
        with torch.no_grad():
            img_np = image.detach().cpu().numpy()
        
        n_rows = self.scales
        n_cols = self.orientations + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        
        for i in range(n_rows):
            axes[i, 0].imshow(np.transpose(img_np, (1, 2, 0)))
            axes[i, 0].set_title("Original" if i == 0 else "")
            axes[i, 0].axis("off")
        
        kernel_idx = 0
        for scale in range(self.scales):
            for orient in range(self.orientations):
                kernel = self.gabor_kernels[kernel_idx]
                kernel_idx += 1
                
                response = np.zeros((32, 32, 3))
                for c in range(3):
                    import scipy.ndimage as ndi
                    filtered = ndi.convolve(img_np[c], kernel, mode='wrap')
                    response[:, :, c] = np.abs(filtered)
                
                response = np.clip(response / response.max(), 0, 1)
                
                ax = axes[scale, orient + 1]
                ax.imshow(response)
                
                if scale == 0:
                    ax.set_title(f"θ={orient*45}°")
                if orient == 0:
                    ax.set_ylabel(f"Scale {scale+1}", rotation=90, size='large')
                
                ax.axis("off")
        
        plt.suptitle("Gabor Filter Responses at Different Scales and Orientations", fontsize=16)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        plt.close(fig)
        del img_np, response
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def test_quantum_texture_attacker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_image = torch.rand(1, 3, 32, 32, device=device)
    
    qta = QuantumTextureAttacker(n_qubits=4, n_layers=5, epsilon=0.1, device=device)
    qta.to(device)
    
    perturbation, adversarial_image = qta(test_image)
    
    qta.visualize_perturbation(
        test_image[0], 
        perturbation[0], 
        adversarial_image[0], 
        "quantum_texture_attacker_output.png"
    )
    
    qta.visualize_gabor_responses(
        test_image[0],
        "quantum_texture_gabor_responses.png"
    )
    
    if hasattr(qta.q_circuit, 'hits'):
        print(f"Cache statistics - Hits: {qta.q_circuit.hits}, Misses: {qta.q_circuit.misses}")
        print(f"Cache hit rate: {qta.q_circuit.hits/(qta.q_circuit.hits+qta.q_circuit.misses)*100:.2f}%")
    
    print(f"Perturbation stats: Min={perturbation.min().item():.4f}, Max={perturbation.max().item():.4f}")
    print(f"Perturbation L2 norm: {torch.norm(perturbation).item():.4f}")
    
    return qta, test_image, perturbation, adversarial_image


if __name__ == "__main__":
    qta, test_image, perturbation, adversarial_image = test_quantum_texture_attacker()
