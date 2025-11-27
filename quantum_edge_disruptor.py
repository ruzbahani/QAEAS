import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pywt
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


def pytorch_haar_wavelet(image):
    """
    اعمال تبدیل موجک Haar با استفاده از PyTorch

    Args:
        image (torch.Tensor): تصویر ورودی [channels, height, width]

    Returns:
        tuple: ضرایب تبدیل موجک (approximation, (horizontal, vertical, diagonal))
    """
    c, h, w = image.shape
    device = image.device

    ll_filter = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device)
    lh_filter = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], device=device)
    hl_filter = torch.tensor([[0.5, -0.5], [0.5, -0.5]], device=device)
    hh_filter = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=device)

    ll_filter = ll_filter.expand(1, 1, 2, 2)
    lh_filter = lh_filter.expand(1, 1, 2, 2)
    hl_filter = hl_filter.expand(1, 1, 2, 2)
    hh_filter = hh_filter.expand(1, 1, 2, 2)

    pad_h = 0 if h % 2 == 0 else 1
    pad_w = 0 if w % 2 == 0 else 1

    if pad_h != 0 or pad_w != 0:
        image = F.pad(image, (0, pad_w, 0, pad_h))

    approx = torch.zeros((c, h // 2, w // 2), device=device)
    horizontal = torch.zeros((c, h // 2, w // 2), device=device)
    vertical = torch.zeros((c, h // 2, w // 2), device=device)
    diagonal = torch.zeros((c, h // 2, w // 2), device=device)

    for i in range(c):
        img_channel = image[i:i + 1].unsqueeze(0)
        approx[i] = F.conv2d(img_channel, ll_filter, stride=2)[0, 0]
        horizontal[i] = F.conv2d(img_channel, lh_filter, stride=2)[0, 0]
        vertical[i] = F.conv2d(img_channel, hl_filter, stride=2)[0, 0]
        diagonal[i] = F.conv2d(img_channel, hh_filter, stride=2)[0, 0]

    return approx, (horizontal, vertical, diagonal)


def pytorch_sobel_filter(image):
    """
    اعمال فیلتر Sobel با PyTorch برای تشخیص لبه

    Args:
        image (torch.Tensor): تصویر ورودی [channels, height, width]

    Returns:
        torch.Tensor: نقشه لبه [height, width]
    """
    device = image.device
    channels, height, width = image.shape

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=torch.float32,
        device=device,
    )

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    edge_maps = []

    for c in range(channels):
        img_channel = image[c:c + 1].unsqueeze(0)
        padded = F.pad(img_channel, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, sobel_x)
        grad_y = F.conv2d(padded, sobel_y)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)[0, 0]
        edge_maps.append(magnitude)

    combined_edges = torch.stack(edge_maps).mean(dim=0)
    return combined_edges


def pytorch_gaussian_blur(image, sigma=0.5):
    """
    اعمال فیلتر گوسی با PyTorch

    Args:
        image (torch.Tensor): تصویر ورودی [height, width] یا [channels, height, width]
        sigma (float): انحراف معیار فیلتر گوسی

    Returns:
        torch.Tensor: تصویر بلور شده با همان ابعاد ورودی
    """
    kernel_size = max(3, int(2 * int(4 * sigma + 0.5) + 1))

    if kernel_size % 2 == 0:
        kernel_size += 1

    is_single_channel = False
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
        is_single_channel = True

    blurred = image.unsqueeze(0)

    if hasattr(F, "gaussian_blur"):
        blurred = F.gaussian_blur(
            blurred,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma],
        )
    else:
        padding = kernel_size // 2

        x = torch.arange(-padding, padding + 1, dtype=torch.float32, device=image.device)
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_h = kernel_1d.view(1, 1, 1, kernel_size).expand(blurred.size(1), 1, 1, kernel_size)
        blurred = F.conv2d(
            F.pad(blurred, (padding, padding, 0, 0), mode="reflect"),
            kernel_h,
            groups=blurred.size(1),
        )

        kernel_v = kernel_1d.view(1, 1, kernel_size, 1).expand(blurred.size(1), 1, kernel_size, 1)
        blurred = F.conv2d(
            F.pad(blurred, (0, 0, padding, padding), mode="reflect"),
            kernel_v,
            groups=blurred.size(1),
        )

    blurred = blurred.squeeze(0)

    if is_single_channel:
        blurred = blurred.squeeze(0)

    return blurred


class QuantumEdgeDisruptor(nn.Module):
    """
    Quantum ensemble member specialized in edge and detail disruption.
    This member focuses on attacking high-frequency components in images,
    specifically disrupting edges and boundaries that are crucial for object recognition.
    بهینه‌سازی شده برای GPU با کاهش انتقال داده بین CPU و GPU.
    """

    def __init__(self, n_qubits=3, n_layers=4, epsilon=0.1, device="cpu", seed=None):
        """
        Initialize the QuantumEdgeDisruptor.

        Args:
            n_qubits (int): Number of qubits to use (default: 3)
            n_layers (int): Number of circuit layers (default: 4)
            epsilon (float): Maximum perturbation magnitude (default: 0.1)
            device (str): Device to run classical computations on (default: "cpu")
        """
        super(QuantumEdgeDisruptor, self).__init__()
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

        self.n_features = 8

        self.q_device = qml.device("default.qubit", wires=self.n_qubits)

        self.n_entangling_layers = 3
        self.n_refinement_layers = 1

        self.q_weights = nn.Parameter(0.01 * torch.randn(self.n_layers, self.n_qubits, 3))

        self.edge_threshold = nn.Parameter(torch.tensor([0.1]))
        self.edge_weight = nn.Parameter(torch.tensor([1.0]))

        self.detail_weights = nn.Parameter(0.1 * torch.ones(3, self.n_features))

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

    def extract_wavelet_features(self, images):
        """
        Extract high-frequency wavelet features from images.
        نسخه بهینه‌شده با استفاده از PyTorch wavelet.

        Args:
            images (torch.Tensor): Input images [batch_size, 3, 32, 32]

        Returns:
            torch.Tensor: Wavelet features [batch_size, n_features]
        """
        batch_size = images.shape[0]
        features = torch.zeros(batch_size, self.n_features, device=self.device)

        for b in range(batch_size):
            img = images[b]

            _, (horizontal, vertical, diagonal) = pytorch_haar_wavelet(img)

            feature_vector = []

            for c in range(3):
                h_energy = torch.mean(torch.abs(horizontal[c]))
                v_energy = torch.mean(torch.abs(vertical[c]))
                d_energy = torch.mean(torch.abs(diagonal[c]))

                channel_features = [h_energy.item(), v_energy.item(), d_energy.item()]
                feature_vector.extend(channel_features[: min(len(channel_features), self.n_features)])

            if len(feature_vector) >= self.n_features:
                feature_vector = feature_vector[: self.n_features]
            else:
                feature_vector.extend([0] * (self.n_features - len(feature_vector)))

            features[b] = torch.tensor(feature_vector, device=self.device)

        if self.calibration_samples < self.max_calibration_samples:
            with torch.no_grad():
                batch_mean = features.mean(dim=0)
                batch_std = features.std(dim=0, unbiased=False) if features.size(0) > 1 else torch.ones_like(
                    features[0]
                )
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
        """
        Quantum Circuit Born Machine architecture.

        Args:
            features (torch.Tensor): Input features [n_features]
            weights (torch.Tensor): Circuit parameters [n_layers, n_qubits, 3]

        Returns:
            list: Expectation values from the circuit [n_qubits]
        """
        for i in range(self.n_qubits):
            feature_idx = i % len(features)
            qml.SX(wires=i)
            qml.RZ(features[feature_idx] * np.pi, wires=i)
            qml.SX(wires=i)

        for l in range(self.n_entangling_layers):
            for i in range(self.n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)

            for i in range(self.n_qubits):
                qml.CZ(wires=[i, (i + 1) % self.n_qubits])

            for i in range(self.n_qubits):
                qml.T(wires=i)

        l = self.n_entangling_layers
        if l < self.n_layers:
            for i in range(self.n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def detect_edges(self, image):
        """
        Detect edges in the image using Sobel filters.
        نسخه بهینه‌شده با استفاده از PyTorch.

        Args:
            image (torch.Tensor): Input image [3, 32, 32]

        Returns:
            torch.Tensor: Edge map [32, 32]
        """
        combined_edges = pytorch_sobel_filter(image)
        threshold_value = self.edge_threshold.item()
        edge_map = (combined_edges > threshold_value).float()
        return edge_map

    def generate_edge_perturbation(self, q_output, edge_map, original_shape):
        """
        Generate edge-focused perturbations from quantum outputs.
        نسخه بهینه‌شده با استفاده از PyTorch.

        Args:
            q_output (torch.Tensor): Quantum circuit output [batch_size, n_qubits]
            edge_map (torch.Tensor): Edge maps [batch_size, 32, 32]
            original_shape (tuple): Shape of original images [batch_size, 3, 32, 32]

        Returns:
            torch.Tensor: Perturbation with the same shape as original images
        """
        batch_size, channels, height, width = original_shape
        perturbation = torch.zeros(original_shape, device=self.device)

        for b in range(batch_size):
            edges = edge_map[b]

            for c in range(channels):
                channel_pert = torch.zeros((height, width), device=self.device)

                for i, q_val in enumerate(q_output[b]):
                    scaled_q_val = q_val.item() * self.detail_weights[c, i % self.n_features]

                    x_coords = torch.linspace(-1, 1, width, device=self.device)
                    y_coords = torch.linspace(-1, 1, height, device=self.device)
                    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

                    freq = (i + 1) * 4.0
                    wavelet = torch.sin(x_grid * freq) * torch.sin(y_grid * freq)
                    phase_shift = np.pi * q_val.item()
                    wavelet_phase = torch.sin(x_grid * freq + phase_shift) * torch.sin(
                        y_grid * freq + phase_shift
                    )

                    pattern = (wavelet + wavelet_phase) * 0.5

                    channel_pert += edges * pattern * scaled_q_val

                smoothed = pytorch_gaussian_blur(channel_pert, sigma=0.5)
                perturbation[b, c] = smoothed * self.edge_weight

        perturbation = self.epsilon * torch.tanh(perturbation)

        return perturbation

    def forward(self, images):
        """
        Forward pass that generates edge-disrupting adversarial perturbations.
        نسخه بهینه‌شده با کش و پیش‌تخصیص.

        Args:
            images (torch.Tensor): Input images [batch_size, 3, 32, 32]

        Returns:
            tuple: (perturbation, adversarial_images)
        """
        batch_size = images.shape[0]
        original_shape = images.shape

        features = self.extract_wavelet_features(images)

        edge_maps = torch.stack([self.detect_edges(images[i]) for i in range(batch_size)])

        q_outputs = torch.zeros((batch_size, self.n_qubits), device=self.device)

        q_outs = []
        for i in range(batch_size):
            q_out = self.q_circuit(features[i], self.q_weights)
            q_outs.append(torch.tensor(q_out, device=self.device))

        q_outputs = torch.stack(q_outs)

        perturbation = self.generate_edge_perturbation(q_outputs, edge_maps, original_shape)

        adversarial_images = torch.clamp(images + perturbation, 0, 1)

        return perturbation, adversarial_images

    def visualize_perturbation(self, image, perturbation, adversarial, filename=None):
        """
        Visualize the original image, perturbation, and adversarial image.

        Args:
            image (torch.Tensor): Original image [3, 32, 32]
            perturbation (torch.Tensor): Perturbation [3, 32, 32]
            adversarial (torch.Tensor): Adversarial image [3, 32, 32]
            filename (str, optional): If provided, save the visualization to this file
        """
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        with torch.no_grad():
            img_np = image.detach().permute(1, 2, 0).cpu().numpy()
            pert_np = perturbation.detach().permute(1, 2, 0).cpu().numpy()
            adv_np = adversarial.detach().permute(1, 2, 0).cpu().numpy()
            edge_map_tensor = self.detect_edges(image)
            edge_map = edge_map_tensor.detach().cpu().numpy()

        pert_amplified = np.clip(pert_np * 10, -1, 1)

        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(edge_map, cmap="gray")
        axes[1].set_title("Edge Map")
        axes[1].axis("off")

        axes[2].imshow(np.clip(pert_np + 0.5, 0, 1))
        axes[2].set_title("Edge Perturbation")
        axes[2].axis("off")

        axes[3].imshow(np.clip(pert_amplified + 0.5, 0, 1))
        axes[3].set_title("Perturbation (10x)")
        axes[3].axis("off")

        axes[4].imshow(np.clip(adv_np, 0, 1))
        axes[4].set_title("Adversarial Image")
        axes[4].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.close(fig)
        del img_np, pert_np, adv_np, edge_map, edge_map_tensor, pert_amplified
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def visualize_wavelet_decomposition(self, image, filename=None):
        """
        Visualize the wavelet decomposition of the image.

        Args:
            image (torch.Tensor): Input image [3, 32, 32]
            filename (str, optional): If provided, save the visualization to this file
        """
        img_np = image.detach().cpu().numpy()

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        for c in range(3):
            coeffs = pywt.wavedec2(img_np[c], "haar", level=1)

            approx, (horizontal, vertical, diagonal) = coeffs

            def normalize_for_display(coeffs_):
                return np.clip(
                    (coeffs_ - coeffs_.min()) / (coeffs_.max() - coeffs_.min() + 1e-10),
                    0,
                    1,
                )

            axes[c, 0].imshow(normalize_for_display(approx), cmap="gray")
            axes[c, 0].set_title(f"Channel {c + 1}: Approximation")
            axes[c, 0].axis("off")

            axes[c, 1].imshow(normalize_for_display(horizontal), cmap="gray")
            axes[c, 1].set_title(f"Channel {c + 1}: Horizontal")
            axes[c, 1].axis("off")

            axes[c, 2].imshow(normalize_for_display(vertical), cmap="gray")
            axes[c, 2].set_title(f"Channel {c + 1}: Vertical")
            axes[c, 2].axis("off")

            axes[c, 3].imshow(normalize_for_display(diagonal), cmap="gray")
            axes[c, 3].set_title(f"Channel {c + 1}: Diagonal")
            axes[c, 3].axis("off")

        plt.suptitle("Haar Wavelet Decomposition", fontsize=16)
        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.close(fig)
        del img_np
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def test_quantum_edge_disruptor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_image = torch.rand(1, 3, 32, 32, device=device)

    qed = QuantumEdgeDisruptor(n_qubits=3, n_layers=4, epsilon=0.1, device=device)
    qed.to(device)

    perturbation, adversarial_image = qed(test_image)

    qed.visualize_perturbation(
        test_image[0],
        perturbation[0],
        adversarial_image[0],
        "quantum_edge_disruptor_output.png",
    )

    qed.visualize_wavelet_decomposition(
        test_image[0],
        "quantum_wavelet_decomposition.png",
    )

    if hasattr(qed.q_circuit, "hits"):
        print(f"Cache statistics - Hits: {qed.q_circuit.hits}, Misses: {qed.q_circuit.misses}")
        print(
            f"Cache hit rate: {qed.q_circuit.hits / (qed.q_circuit.hits + qed.q_circuit.misses) * 100:.2f}%"
        )

    print(f"Perturbation stats: Min={perturbation.min().item():.4f}, Max={perturbation.max().item():.4f}")
    print(f"Perturbation L2 norm: {torch.norm(perturbation).item():.4f}")

    return qed, test_image, perturbation, adversarial_image


if __name__ == "__main__":
    qed, test_image, perturbation, adversarial_image = test_quantum_edge_disruptor()
