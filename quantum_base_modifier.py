import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import hashlib


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
        features_bytes = features_rounded.tobytes()
        features_hash = hashlib.md5(features_bytes).hexdigest()

        weights_flat = weights.detach().cpu().numpy().flatten()
        if len(weights_flat) > 20:
            indices = np.linspace(0, len(weights_flat) - 1, 20, dtype=int)
            weights_samples = weights_flat[indices]
        else:
            weights_samples = weights_flat

        weights_rounded = np.round(weights_samples, 6)
        weights_bytes = weights_rounded.tobytes()
        weights_hash = hashlib.md5(weights_bytes).hexdigest()

        cache_key = (self.member_id, features_hash, weights_hash)

        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1
        result = self.circuit_func(features, weights)

        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)

        self.cache[cache_key] = result
        return result


def torch_dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]

    k = torch.arange(N, device=x.device).float()
    n = k.unsqueeze(1)
    dct_mat = torch.cos(torch.pi * (2 * n + 1) * k / (2 * N))

    if norm == "ortho":
        dct_mat[:, 0] *= 1.0 / np.sqrt(2)
        dct_mat *= np.sqrt(2.0 / N)

    result = torch.matmul(dct_mat, x)
    return result


def torch_dct_2d(x):
    x = torch_dct(x)
    x = x.transpose(-2, -1)
    x = torch_dct(x)
    x = x.transpose(-2, -1)
    return x


class QuantumBaseModifier(nn.Module):
    def __init__(self, n_qubits=4, n_layers=6, epsilon=0.1, device="cpu", seed=None):
        super(QuantumBaseModifier, self).__init__()
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
        self.n_features = 16

        self.q_device = qml.device("default.qubit", wires=self.n_qubits, shots=None)

        self.n_qft_layers = 2
        self.n_ent_layers = 4

        q_circuit_func = qml.QNode(
            self.quantum_circuit,
            self.q_device,
            interface="torch",
            diff_method="parameter-shift",
        )

        self.q_circuit = QuantumCircuitCache(q_circuit_func, cache_size=200)

        weight_scale = 0.1 / np.sqrt(n_qubits * n_layers)
        self.q_weights = nn.Parameter(
            weight_scale * torch.randn(2, self.n_layers, self.n_qubits, 3)
        )

        self.scale_factors = nn.Parameter(0.05 * torch.ones(3))

        self.encoding_scale = nn.Parameter(torch.ones(self.n_features))

        self.feature_mean = nn.Parameter(
            torch.zeros(self.n_features), requires_grad=False
        )
        self.feature_std = nn.Parameter(
            torch.ones(self.n_features), requires_grad=False
        )

        self.freq_weights = nn.Parameter(torch.zeros(4, 4))

        self.calibration_samples = 0
        self.max_calibration_samples = 100

        self._create_dct_matrices()

    def _create_dct_matrices(self):
        N = 32
        k = torch.arange(N, device=self.device).float()
        n = k.unsqueeze(1)
        self.dct_mat_x = torch.cos(torch.pi * (2 * n + 1) * k / (2 * N))

        k = torch.arange(N, device=self.device).float()
        n = k.unsqueeze(1)
        self.dct_mat_y = torch.cos(torch.pi * (2 * n + 1) * k / (2 * N))

        self.dct_mat_x *= np.sqrt(2.0 / N)
        self.dct_mat_y *= np.sqrt(2.0 / N)
        self.dct_mat_x[:, 0] *= 1.0 / np.sqrt(2)
        self.dct_mat_y[:, 0] *= 1.0 / np.sqrt(2)

    def extract_low_freq_features(self, image):
        batch_size = image.shape[0]
        features = torch.zeros(batch_size, self.n_features, device=self.device)

        for ali in range(batch_size):
            img = image[ali]
            dct_features = torch.zeros((32, 32), device=self.device)

            for c in range(3):
                channel_img = img[c]
                dct_x = torch.matmul(self.dct_mat_x, channel_img)
                dct_xy = torch.matmul(dct_x, self.dct_mat_y.T)

                channel_weight = 1.0 if c == 0 else (0.8 if c == 1 else 0.6)
                dct_features += dct_xy * channel_weight / 2.4

            feature_idx = 0
            low_freq = torch.zeros(self.n_features, device=self.device)

            for diag in range(8):
                for j in range(min(diag + 1, 4)):
                    i = diag - j
                    if i < 4 and feature_idx < self.n_features:
                        low_freq[feature_idx] = dct_features[i, j]
                        feature_idx += 1

            features[ali] = low_freq

        if self.calibration_samples < self.max_calibration_samples:
            with torch.no_grad():
                batch_mean = features.mean(dim=0)
                batch_std = (
                    features.std(dim=0, unbiased=False)
                    if features.size(0) > 1
                    else torch.ones_like(features[0])
                )
                batch_std = torch.clamp(batch_std, min=1e-6)

                if self.calibration_samples == 0:
                    self.feature_mean.data = batch_mean.to(self.device)
                    self.feature_std.data = batch_std.to(self.device)
                else:
                    alpha = 0.1
                    self.feature_mean.data = (
                        (1 - alpha) * self.feature_mean + alpha * batch_mean
                    ).to(self.device)
                    self.feature_std.data = (
                        (1 - alpha) * self.feature_std + alpha * batch_std
                    ).to(self.device)

                self.calibration_samples += 1

        feature_mean = self.feature_mean.to(self.device)
        feature_std = self.feature_std.to(self.device)

        normalized_features = (features - feature_mean) / feature_std

        scaled_features = normalized_features * self.encoding_scale

        return scaled_features

    def quantum_circuit(self, features, weights):
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)
            qml.RZ(
                features[i + self.n_qubits]
                if i + self.n_qubits < features.shape[0]
                else 0.0,
                wires=i,
            )
            if i + 2 * self.n_qubits < features.shape[0]:
                qml.RX(features[i + 2 * self.n_qubits], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])

        for layer in range(self.n_layers):
            layer_idx = layer % (self.n_qft_layers + self.n_ent_layers)

            if layer_idx < self.n_qft_layers:
                for i in range(self.n_qubits):
                    qml.RX(weights[0, layer % self.n_qft_layers, i, 0], wires=i)
                    qml.RY(weights[0, layer % self.n_qft_layers, i, 1], wires=i)
                    qml.RZ(weights[0, layer % self.n_qft_layers, i, 2], wires=i)

                qml.QFT(wires=range(self.n_qubits))

                for i in range(self.n_qubits):
                    phase_angle = (
                        weights[0, layer % self.n_qft_layers, i, 0] % (2 * np.pi)
                    )
                    qml.PhaseShift(phase_angle, wires=i)

                qml.adjoint(qml.QFT)(wires=range(self.n_qubits))

            else:
                ent_layer_idx = layer_idx - self.n_qft_layers

                for i in range(self.n_qubits):
                    qml.RX(
                        weights[1, ent_layer_idx % self.n_ent_layers, i, 0], wires=i
                    )
                    qml.RY(
                        weights[1, ent_layer_idx % self.n_ent_layers, i, 1], wires=i
                    )
                    qml.RZ(
                        weights[1, ent_layer_idx % self.n_ent_layers, i, 2], wires=i
                    )

                if ent_layer_idx % 2 == 0:
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                else:
                    for i in range(self.n_qubits):
                        target = (i + 2) % self.n_qubits
                        qml.CNOT(wires=[i, target])

                if self.n_qubits >= 3:
                    control1 = ent_layer_idx % self.n_qubits
                    control2 = (control1 + 1) % self.n_qubits
                    target = (control2 + 1) % self.n_qubits
                    qml.Toffoli(wires=[control1, control2, target])

        expectations = []
        for i in range(self.n_qubits):
            expectations.append(qml.expval(qml.PauliX(i)))
            expectations.append(qml.expval(qml.PauliY(i)))
            expectations.append(qml.expval(qml.PauliZ(i)))

        return expectations

    def process_quantum_output(self, q_output, original_shape):
        batch_size, channels, height, width = original_shape
        perturbation = torch.zeros(original_shape, device=self.device)

        for b in range(batch_size):
            freq_pattern = torch.zeros((height, width), device=self.device)
            q_idx = 0

            for i in range(4):
                for j in range(4):
                    idx_group = q_idx % (self.n_qubits * 3)
                    value_x = q_output[b, idx_group]
                    value_y = q_output[
                        b, (idx_group + 1) % (self.n_qubits * 3)
                    ]
                    value_z = q_output[
                        b, (idx_group + 2) % (self.n_qubits * 3)
                    ]
                    combined_value = (value_x + value_y + value_z) / 3.0
                    freq_pattern[i, j] = combined_value * (
                        1.0 + self.freq_weights[i, j]
                    )
                    q_idx += 1

            for i in range(4):
                for j in range(4):
                    freq_pattern[height - i - 1, j] = freq_pattern[i, j] * 0.5
                    freq_pattern[i, width - j - 1] = freq_pattern[i, j] * 0.5
                    freq_pattern[height - i - 1, width - j - 1] = (
                        freq_pattern[i, j] * 0.25
                    )

            spatial_pattern = torch.matmul(
                self.dct_mat_x.T, torch.matmul(freq_pattern, self.dct_mat_y)
            )

            for c in range(channels):
                perturbation[b, c] = spatial_pattern * self.scale_factors[c]

        perturbation = self.epsilon * torch.tanh(perturbation)
        return perturbation

    def forward(self, images):
        batch_size = images.shape[0]
        original_shape = images.shape

        features = self.extract_low_freq_features(images)

        q_outputs = torch.zeros((batch_size, 3 * self.n_qubits), device=self.device)

        q_outs = []
        for i in range(batch_size):
            q_out = self.q_circuit(features[i], self.q_weights)
            q_outs.append(torch.tensor(q_out, device=self.device))

        q_outputs = torch.stack(q_outs)

        perturbation = self.process_quantum_output(q_outputs, original_shape)

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
        axes[1].set_title("Perturbation")
        axes[1].axis("off")

        axes[2].imshow(np.clip(pert_amplified + 0.5, 0, 1))
        axes[2].set_title("Perturbation (10x)")
        axes[2].axis("off")

        axes[3].imshow(np.clip(adv_np, 0, 1))
        axes[3].set_title("Adversarial Image")
        axes[3].axis("off")

        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.close(fig)
        del img_np, pert_np, adv_np, pert_amplified


def test_quantum_base_modifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_image = torch.rand(1, 3, 32, 32, device=device)

    qbm = QuantumBaseModifier(n_qubits=4, n_layers=6, epsilon=0.1, device=device)
    qbm.to(device)

    perturbation, adversarial_image = qbm(test_image)

    qbm.visualize_perturbation(
        test_image[0],
        perturbation[0],
        adversarial_image[0],
        "quantum_base_modifier_output.png",
    )

    if hasattr(qbm.q_circuit, "hits"):
        print(f"Cache statistics - Hits: {qbm.q_circuit.hits}, Misses: {qbm.q_circuit.misses}")
        print(
            f"Cache hit rate: {qbm.q_circuit.hits / (qbm.q_circuit.hits + qbm.q_circuit.misses) * 100:.2f}%"
        )

    print(
        f"Perturbation stats: Min={perturbation.min().item():.4f}, Max={perturbation.max().item():.4f}"
    )
    print(f"Perturbation L2 norm: {torch.norm(perturbation).item():.4f}")

    return qbm, test_image, perturbation, adversarial_image


if __name__ == "__main__":
    qbm, test_image, perturbation, adversarial_image = test_quantum_base_modifier()
