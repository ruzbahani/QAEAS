import os
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gc
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


class QuantumFocalAttacker(nn.Module):
    """
    Quantum ensemble member specialized in attacking sensitive regions.
    This member focuses on identifying and perturbing the most important areas
    of an image for classification by the target model, using Grad-CAM to locate
    regions of high activation importance.
    """

    def __init__(self, target_model, n_qubits=3, n_layers=6, n_focal_regions=6,
                 epsilon=0.1, device="cpu", seed=None):
        """
        Initialize the QuantumFocalAttacker.

        Args:
            target_model (nn.Module): Target classification model to attack
            n_qubits (int): Number of qubits to use (default: 3)
            n_layers (int): Number of circuit layers (default: 6)
            n_focal_regions (int): Number of focal regions to target (default: 6)
            epsilon (float): Maximum perturbation magnitude (default: 0.1)
            device (str): Device to run classical computations on (default: "cpu")
        """
        super(QuantumFocalAttacker, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.target_model = target_model
        self.target_model.eval()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_focal_regions = n_focal_regions
        self.epsilon = epsilon
        self.device = device

        self.q_device = qml.device("default.qubit", wires=self.n_qubits)

        self.n_encoder_layers = 3
        self.n_adversarial_layers = 3

        self.q_weights = nn.Parameter(0.01 * torch.randn(2, self.n_layers, self.n_qubits, 3))

        self.region_importance = nn.Parameter(torch.ones(self.n_focal_regions) / self.n_focal_regions)
        self.focus_sharpness = nn.Parameter(torch.tensor([5.0]))

        self.channel_weights = nn.Parameter(torch.ones(3) / 3)

        q_circuit_func = qml.QNode(
            self.quantum_circuit,
            self.q_device,
            interface="torch",
            diff_method="adjoint"
        )

        self.q_circuit = QuantumCircuitCache(q_circuit_func, cache_size=200)

        self.gradients = None
        self.activations = None
        self.hooks_registered = False

        self._register_hooks()

        self.clustering_cache = {}
        self.clustering_frequency = 10

        self.debug_stats = {
            'clustering_failures': 0,
            'low_patch_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        if hasattr(self.target_model, 'layer3') and not self.hooks_registered:
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()

            def forward_hook(module, input, output):
                self.activations = output.detach()

            try:
                last_conv_layer = self.target_model.layer3[-1].conv2
                self.backward_hook = last_conv_layer.register_full_backward_hook(backward_hook)
                self.forward_hook = last_conv_layer.register_forward_hook(forward_hook)
                self.hooks_registered = True
            except Exception as e:
                print(f"Warning: Could not register hooks on target model: {e}")
                print("Make sure your model has a structure similar to ResNet")

    @torch.no_grad()
    def compute_gradcam(self, images, target_class=None):
        """
        Compute Grad-CAM activation maps for the given images.

        Args:
            images (torch.Tensor): Input images [batch_size, 3, 32, 32]
            target_class (int, optional): Target class to attack, if None will use predicted class

        Returns:
            torch.Tensor: Grad-CAM maps [batch_size, 32, 32]
        """
        batch_size = images.shape[0]

        self.target_model.eval()
        if not self.hooks_registered:
            self._register_hooks()
            if not self.hooks_registered:
                print("Warning: Could not register hooks. Using random activation maps.")
                random_maps = torch.rand(batch_size, 32, 32, device=self.device)
                return random_maps

        with torch.enable_grad():
            images.requires_grad_(True)
            outputs = self.target_model(images)

            if target_class is None:
                target_class = outputs.argmax(dim=1)
            elif isinstance(target_class, int):
                target_class = torch.tensor([target_class] * batch_size, device=self.device)

            loss = 0
            for i in range(batch_size):
                loss += outputs[i, target_class[i]]

            self.target_model.zero_grad()
            loss.backward(retain_graph=True)

            gradcam_maps = []

            for i in range(batch_size):
                if self.gradients is None or self.activations is None:
                    gradcam_maps.append(torch.rand(32, 32, device=self.device))
                    continue

                grads = self.gradients[i]
                acts = self.activations[i]

                weights = torch.mean(grads, dim=(1, 2))

                cam = torch.zeros(acts.shape[1:], device=self.device)
                for j, w in enumerate(weights):
                    cam += w * acts[j]

                cam = F.relu(cam)
                if cam.max() > 1e-10:
                    cam = cam / cam.max()

                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=(32, 32),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

                noise = torch.randn_like(cam) * 0.05
                cam = torch.clamp(cam + noise, 0, 1)

                gradcam_maps.append(cam)

            images.requires_grad_(False)

        return torch.stack(gradcam_maps)

    @torch.no_grad()
    def extract_focal_features(self, images, gradcam_maps):
        """
        Extract features from focal regions identified by Grad-CAM.

        Args:
            images (torch.Tensor): Input images [batch_size, 3, 32, 32]
            gradcam_maps (torch.Tensor): Grad-CAM maps [batch_size, 32, 32]

        Returns:
            torch.Tensor: Focal features [batch_size, n_features]
        """
        batch_size = images.shape[0]
        n_features = self.n_focal_regions * 2
        features = torch.zeros(batch_size, n_features, device=self.device)

        for b in range(batch_size):
            cache_key = b % self.clustering_frequency
            if cache_key in self.clustering_cache:
                features[b] = self.clustering_cache[cache_key].clone()
                self.debug_stats['cache_hits'] += 1
                continue

            self.debug_stats['cache_misses'] += 1

            img = images[b]
            cam = gradcam_maps[b]

            patches = []
            importance = []

            for patch_size in [3, 5, 7]:
                stride = max(1, patch_size // 2)
                for i in range(0, 32 - patch_size + 1, stride):
                    for j in range(0, 32 - patch_size + 1, stride):
                        img_patch = img[:, i:i + patch_size, j:j + patch_size]
                        patch_importance = torch.mean(cam[i:i + patch_size, j:j + patch_size])

                        avg_color = torch.mean(img_patch.reshape(3, -1), dim=1)

                        patch_features = torch.cat([
                            avg_color,
                            torch.tensor([i / 32, j / 32, patch_size / 32], device=self.device)
                        ])

                        patches.append(patch_features)
                        importance.append(patch_importance)

            if len(patches) >= self.n_focal_regions:
                patches = torch.stack(patches)
                importance = torch.stack(importance)

                weights = importance / (torch.sum(importance) + 1e-6)
                weighted_patches = patches * weights.unsqueeze(1)

                try:
                    max_samples = 1024
                    if len(weighted_patches) > max_samples:
                        indices = torch.randperm(len(weighted_patches))[:max_samples]
                        data_for_clustering = weighted_patches[indices].cpu().numpy()
                    else:
                        data_for_clustering = weighted_patches.cpu().numpy()

                    scaler = StandardScaler()
                    data_for_clustering = scaler.fit_transform(data_for_clustering)

                    n_clusters = min(self.n_focal_regions, len(data_for_clustering))

                    kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        batch_size=6144,
                        n_init=1,
                        random_state=42
                    )

                    cluster_labels = kmeans.fit_predict(data_for_clustering)

                    if len(weighted_patches) > max_samples:
                        full_cluster_labels = np.zeros(len(weighted_patches), dtype=np.int32)
                        centers = kmeans.cluster_centers_

                        remaining_indices = np.setdiff1d(np.arange(len(weighted_patches)), indices)
                        for idx in remaining_indices:
                            sample = scaler.transform(weighted_patches[idx].cpu().numpy().reshape(1, -1))
                            distances = np.linalg.norm(centers - sample, axis=1)
                            closest_cluster = np.argmin(distances)
                            full_cluster_labels[idx] = closest_cluster

                        full_cluster_labels[indices] = cluster_labels
                        cluster_labels = full_cluster_labels

                    for i in range(n_clusters):
                        cluster_indices = torch.tensor(np.where(cluster_labels == i)[0], device=self.device)

                        if len(cluster_indices) > 0:
                            cluster_importance = torch.mean(importance[cluster_indices])
                            cluster_avg_features = torch.mean(patches[cluster_indices], dim=0)

                            idx = i * 2
                            if idx < n_features:
                                features[b, idx] = cluster_importance
                                features[b, idx + 1] = torch.mean(cluster_avg_features[:3])

                    remaining_features = n_features - (n_clusters * 2)
                    if remaining_features > 0:
                        start_idx = n_clusters * 2
                        for i in range(remaining_features):
                            features[b, start_idx + i] = features[b, i % (n_clusters * 2)]

                    self.clustering_cache[cache_key] = features[b].clone()

                except Exception as e:
                    self.debug_stats['clustering_failures'] += 1
                    print(f"Warning: Clustering failed: {e}. Using fallback features.")

                    avg_rgb = torch.mean(img.reshape(3, -1), dim=1)
                    avg_importance = torch.mean(cam)

                    for i in range(n_features // 2):
                        features[b, i * 2] = avg_importance + torch.randn(1, device=self.device) * 0.1
                        features[b, i * 2 + 1] = avg_rgb[i % 3] + torch.randn(1, device=self.device) * 0.1

                    self.clustering_cache[cache_key] = features[b].clone()
            else:
                self.debug_stats['low_patch_count'] += 1

                avg_rgb = torch.mean(img.reshape(3, -1), dim=1)
                avg_importance = torch.mean(cam)

                for i in range(n_features // 2):
                    features[b, i * 2] = avg_importance + torch.randn(1, device=self.device) * 0.1
                    features[b, i * 2 + 1] = avg_rgb[i % 3] + torch.randn(1, device=self.device) * 0.1

                self.clustering_cache[cache_key] = features[b].clone()

        features = torch.clamp(features, -1, 1)

        return features

    def quantum_circuit(self, features, weights):
        """
        Variational Quantum Adversarial Network circuit.

        Args:
            features (torch.Tensor): Input features [n_features]
            weights (torch.Tensor): Circuit parameters [2, n_layers, n_qubits, 3]

        Returns:
            list: Expectation values from the circuit [n_qubits]
        """
        for l in range(self.n_encoder_layers):
            if l == 0:
                for i in range(self.n_qubits):
                    idx1 = (i * 2) % len(features)
                    idx2 = (i * 2 + 1) % len(features)

                    f1 = qml.math.clip(features[idx1], -1, 1)
                    f2 = qml.math.clip(features[idx2], -1, 1)

                    qml.Hadamard(wires=i)
                    qml.RY(f1 * np.pi, wires=i)
                    qml.RZ(f2 * np.pi, wires=i)

            for i in range(self.n_qubits):
                qml.RX(weights[0, l, i, 0], wires=i)
                qml.RY(weights[0, l, i, 1], wires=i)
                qml.RZ(weights[0, l, i, 2], wires=i)

            for i in range(self.n_qubits - 1):
                qml.CRot(
                    weights[0, l, i, 0],
                    weights[0, l, i, 1],
                    weights[0, l, i, 2],
                    wires=[i, (i + 1) % self.n_qubits]
                )

        for l in range(self.n_adversarial_layers):
            idx = l + self.n_encoder_layers
            if idx < self.n_layers:
                qml.MultiRZ(weights[1, l, 0, 0], wires=range(self.n_qubits))

                for i in range(self.n_qubits):
                    qml.RX(weights[1, l, i, 1], wires=i)
                    qml.RY(weights[1, l, i, 2], wires=i)

                for i in range(self.n_qubits - 1):
                    qml.ctrl(qml.Hadamard, control=i)(wires=i + 1)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def generate_focal_perturbation(self, q_output, gradcam_maps, original_shape):
        """
        Generate focal perturbations targeting sensitive regions.

        Args:
            q_output (torch.Tensor): Quantum circuit output [batch_size, n_qubits]
            gradcam_maps (torch.Tensor): Grad-CAM maps [batch_size, 32, 32]
            original_shape (tuple): Shape of original images [batch_size, 3, 32, 32]

        Returns:
            torch.Tensor: Perturbation with the same shape as original images
        """
        batch_size, channels, height, width = original_shape
        perturbation = torch.zeros(original_shape, device=self.device)

        for b in range(batch_size):
            cam = gradcam_maps[b]

            smoothed_cam = TF.gaussian_blur(
                cam.unsqueeze(0).unsqueeze(0),
                kernel_size=5,
                sigma=1.0
            ).squeeze()

            sharpness = self.focus_sharpness.item()
            focal_mask = torch.pow(smoothed_cam, sharpness)

            focal_mask = focal_mask / (focal_mask.max() + 1e-10)

            for c in range(channels):
                q_val = q_output[b, c % self.n_qubits]

                pattern = torch.zeros((height, width), device=self.device)

                freq = 0.2 + 0.3 * abs(q_val.item())
                phase = q_val.item() * np.pi

                x_coords = torch.linspace(0, width * freq, width, device=self.device)
                y_coords = torch.linspace(0, height * freq, height, device=self.device)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

                if c == 0:
                    dist = torch.sqrt((x_grid - width // 2) ** 2 + (y_grid - height // 2) ** 2)
                    pattern = torch.sin(dist * 0.5 + phase)
                elif c == 1:
                    pattern = torch.sin(x_grid + phase) * torch.sin(y_grid + phase)
                else:
                    pattern = torch.sin((x_grid + y_grid) * 0.5 + phase)

                pattern = pattern * self.channel_weights[c]

                masked_pattern = pattern * focal_mask

                perturbation[b, c] = masked_pattern

        perturbation = self.epsilon * torch.tanh(perturbation)

        return perturbation

    def forward(self, images, target_class=None):
        """
        Forward pass that generates focal adversarial perturbations.

        Args:
            images (torch.Tensor): Input images [batch_size, 3, 32, 32]
            target_class (int, optional): Target class to attack, if None will use predicted class

        Returns:
            tuple: (perturbation, adversarial_images, gradcam_maps)
        """
        batch_size = images.shape[0]
        original_shape = images.shape

        gradcam_maps = self.compute_gradcam(images, target_class)

        features = self.extract_focal_features(images, gradcam_maps)

        q_outputs = torch.zeros((batch_size, self.n_qubits), device=self.device)

        q_outs = []
        for i in range(batch_size):
            q_out = self.q_circuit(features[i], self.q_weights)
            q_outs.append(torch.tensor(q_out, device=self.device))

        q_outputs = torch.stack(q_outs)

        perturbation = self.generate_focal_perturbation(q_outputs, gradcam_maps, original_shape)

        adversarial_images = torch.clamp(images + perturbation, 0, 1)

        del features, q_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return perturbation, adversarial_images, gradcam_maps

    @torch.no_grad()
    def visualize_perturbation(self, image, perturbation, adversarial, gradcam_map, filename=None):
        """
        Visualize the focal attack with Grad-CAM and perturbation.

        Args:
            image (torch.Tensor): Original image [3, 32, 32]
            perturbation (torch.Tensor): Perturbation [3, 32, 32]
            adversarial (torch.Tensor): Adversarial image [3, 32, 32]
            gradcam_map (torch.Tensor): Grad-CAM map [32, 32]
            filename (str, optional): If provided, save the visualization to this file
        """
        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
        pert_np = perturbation.detach().cpu().permute(1, 2, 0).numpy()
        adv_np = adversarial.detach().cpu().permute(1, 2, 0).numpy()
        cam_np = gradcam_map.detach().cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        pert_amplified = np.clip(pert_np * 10, -1, 1)

        axes[0, 0].imshow(np.clip(img_np, 0, 1))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cam_np, cmap='jet', alpha=0.7)
        axes[0, 1].imshow(np.clip(img_np, 0, 1), alpha=0.3)
        axes[0, 1].set_title("Grad-CAM Heatmap")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(np.clip(pert_amplified + 0.5, 0, 1))
        axes[0, 2].set_title("Focal Perturbation (10x)")
        axes[0, 2].axis("off")

        masked_pert = pert_amplified * cam_np[:, :, np.newaxis]

        focal_regions_vis = np.zeros_like(img_np)
        threshold = np.percentile(cam_np, 80)
        focal_mask = cam_np > threshold

        for c in range(3):
            focal_regions_vis[:, :, c] = np.where(focal_mask, cam_np, 0)

        axes[1, 0].imshow(focal_regions_vis, cmap='jet')
        axes[1, 0].set_title("Focal Regions")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(np.clip(masked_pert + 0.5, 0, 1))
        axes[1, 1].set_title("Masked Perturbation")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(np.clip(adv_np, 0, 1))
        axes[1, 2].set_title("Adversarial Image")
        axes[1, 2].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

        del img_np, pert_np, adv_np, cam_np, pert_amplified, masked_pert, focal_regions_vis
        gc.collect()

    @torch.no_grad()
    def visualize_focal_clusters(self, image, gradcam_map, filename=None):
        """
        Visualize the focal clusters identified in the image.

        Args:
            image (torch.Tensor): Original image [3, 32, 32]
            gradcam_map (torch.Tensor): Grad-CAM map [32, 32]
            filename (str, optional): If provided, save the visualization to this file
        """
        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
        cam_np = gradcam_map.detach().cpu().numpy()

        cam_np = np.clip(cam_np + np.random.normal(0, 0.05, cam_np.shape), 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].imshow(cam_np, cmap='jet', alpha=0.5)
        axes[0].set_title("Image with Grad-CAM")
        axes[0].axis("off")

        try:
            flat_img = img_np.reshape(-1, 3)
            flat_cam = cam_np.flatten()

            weighted_data = flat_img * flat_cam[:, np.newaxis]

            pca = PCA(n_components=3)
            data_pca = pca.fit_transform(weighted_data)

            n_clusters = min(self.n_focal_regions, len(weighted_data))

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=6144,
                n_init=1,
                random_state=42
            )

            cluster_labels = kmeans.fit_predict(data_pca)

            cluster_img = np.zeros_like(img_np)

            cluster_colors = []
            for i in range(n_clusters):
                hue = i / n_clusters
                cluster_colors.append(plt.cm.hsv(hue)[:3])

            labels_2d = cluster_labels.reshape(32, 32)

            for i in range(n_clusters):
                mask = labels_2d == i
                color = cluster_colors[i]
                for c in range(3):
                    cluster_img[:, :, c] = np.where(mask, color[c], cluster_img[:, :, c])

            axes[1].imshow(cluster_img)
            axes[1].set_title(f"Focal Clusters (K={n_clusters})")
            axes[1].axis("off")

        except Exception as e:
            axes[1].text(
                0.5, 0.5,
                f"Clustering failed: {str(e)}",
                ha='center',
                va='center',
                transform=axes[1].transAxes
            )
            axes[1].axis('off')

        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

        del img_np, cam_np
        gc.collect()

    def get_debug_info(self):
        """
        Return debug information about the model.
        """
        cache_info = {
            'cache_hits': self.q_circuit.hits,
            'cache_misses': self.q_circuit.misses,
            'cache_hit_ratio': self.q_circuit.hits / (self.q_circuit.hits + self.q_circuit.misses + 1e-10)
        }

        debug_info = {
            **self.debug_stats,
            'cache_info': cache_info,
            'clustering_cache_size': len(self.clustering_cache)
        }

        return debug_info
