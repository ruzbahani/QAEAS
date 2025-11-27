# QAEAS: Quantum Adaptive Ensemble Attack System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.32-green.svg)](https://pennylane.ai/)

> **QAEAS: A Quantum Adaptive Ensemble Attack System Against Robust Deep Neural Networks**  
> Ali Mohammadi Ruzbahani, Abbas Yazdinejad, Hadis Karimipour  
> University of Calgary | Under Review

## Overview

QAEAS is a quantum-enhanced adversarial attack framework that exploits near-term quantum computing resources to generate imperceptible adversarial perturbations against robust deep neural networks. The system integrates five specialized Variational Quantum Circuits (VQCs), each targeting distinct visual features (frequency, texture, edges, color, saliency), coordinated through a Hierarchical Adaptive Kalman Filter for dynamic ensemble weighting.

### Key Results

- **Attack Success Rate**: 87.3% (CIFAR-10), 78.9% (CIFAR-100)
- **Perceptual Quality**: SSIM > 0.96, 13.5% lower L2 distortion vs. baselines
- **Transferability**: 68.9% success on unseen architectures
- **NISQ Compatible**: 3-4 qubits, 4-6 circuit layers

## Architecture

QAEAS consists of five quantum adversarial modules:

1. **Quantum Base Modifier (QBM)**: DCT-based frequency domain perturbations with QFT layers
2. **Quantum Texture Attacker (QTA)**: QCNN architecture targeting Gabor texture features
3. **Quantum Edge Disruptor (QED)**: Wavelet-driven edge perturbations using Born Machine circuits
4. **Quantum Color Distorter (QCD)**: RGB statistical encoding with SWAP-based color mixing
5. **Quantum Focal Attacker (QFA)**: Grad-CAM guided attention-based perturbations

Ensemble outputs are adaptively weighted using a two-tier Kalman Filter:
- **Member-level filters**: Track per-class effectiveness for each module
- **Global filter**: Aggregates cross-module performance with adaptive noise scaling

## Installation

### Requirements

```bash
# Core dependencies
Python >= 3.9
PyTorch >= 2.0
PennyLane >= 0.32
CUDA >= 11.8 (for GPU acceleration)
```

### Setup

```bash
# Clone repository
git clone https://github.com/ruzbahani/QAEAS.git
cd QAEAS

# Create virtual environment
python -m venv qaeas_env
source qaeas_env/bin/activate  # On Windows: qaeas_env\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pennylane pennylane-lightning
pip install numpy scipy scikit-learn scikit-image
pip install matplotlib seaborn tqdm pandas
pip install lpips  # For perceptual metrics
```

### Verify Installation

```python
import torch
import pennylane as qml
print(f"PyTorch: {torch.__version__}")
print(f"PennyLane: {qml.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

## Quick Start

### 1. Prepare Dataset

```python
from dataloader import Ali_DataLoader

# Initialize CIFAR-10 dataloader
data_loader = Ali_DataLoader(
    data_dir='./data',
    batch_size=64,
    train_percent=0.80,
    val_percent=0.10,
    test_percent=0.10
)

train_loader, val_loader, test_loader = data_loader.get_dataloaders()
```

### 2. Train Target Model

```python
from target_resnet20 import ResNet20, train_target_model1_resnet20

# Train ResNet-20 on CIFAR-10
model = ResNet20(num_classes=10)
train_target_model1_resnet20(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=300,
    lr=0.001,
    device='cuda'
)
```

### 3. Initialize QAEAS

```python
from quantum_ensemble_manager import QuantumEnsembleManager

# Create quantum ensemble attack system
qaeas = QuantumEnsembleManager(
    target_model=model,
    device='cuda',
    epsilon=8/255,  # L∞ perturbation budget
    run_name='qaeas_cifar10'
)
```

### 4. Generate Adversarial Examples

```python
# Load test batch
images, labels = next(iter(test_loader))
images, labels = images.to('cuda'), labels.to('cuda')

# Generate adversarial perturbations
adversarial_images, perturbation, member_outputs, weights = qaeas(images)

# Evaluate attack success
with torch.no_grad():
    original_preds = torch.argmax(model(images), dim=1)
    adversarial_preds = torch.argmax(model(adversarial_images), dim=1)
    success_rate = (original_preds != adversarial_preds).float().mean()
    
print(f"Attack Success Rate: {success_rate*100:.2f}%")
```

### 5. Train QAEAS with Kalman Adaptation

```python
# Train quantum ensemble with adaptive coordination
train_metrics = qaeas.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    vis_every=10,
    save_every=10,
    target_class=None  # None for untargeted attack
)

# Evaluate on test set
eval_results = qaeas.evaluate_on_dataset(
    test_loader,
    num_samples=10000
)
```

## Project Structure

```
QAEAS/
├── dataloader.py                    # CIFAR-10/100 data loading and preprocessing
├── dataset_analyzer.py              # Dataset visualization and statistics
├── quantum_base_modifier.py         # QBM: DCT-based frequency perturbations
├── quantum_texture_attacker.py      # QTA: Gabor texture-based QCNN
├── quantum_edge_disruptor.py        # QED: Wavelet edge perturbations
├── quantum_color_distorter.py       # QCD: RGB statistical color mixing
├── quantum_focal_attacker.py        # QFA: Grad-CAM attention-driven attacks
├── quantum_ensemble_manager.py      # Kalman-based ensemble coordinator
├── target_resnet20.py               # ResNet-20 target model
├── target_efficientnet_b0.py        # EfficientNet-B0 target model
├── adversarial_attacks_base.py      # Classical baseline attacks (FGSM, PGD, C&W)
├── comprehensive_comparison_script.py # Evaluation against baselines
├── qaeas.py                         # Full QAEAS evaluation pipeline
├── main.ipynb                       # End-to-end workflow notebook
└── README.md
```

## Usage Examples

### Untargeted Attack

```python
# Generate untargeted adversarial examples
qaeas = QuantumEnsembleManager(target_model=model, epsilon=8/255)
adv_images, _, _, _ = qaeas(images, target_class=None)
```

### Targeted Attack

```python
# Force misclassification to class 3
qaeas = QuantumEnsembleManager(target_model=model, epsilon=8/255)
adv_images, _, _, _ = qaeas(images, target_class=3)
```

### Evaluate Transferability

```python
from torchvision.models import vgg16

# Test on surrogate model
surrogate_model = vgg16(pretrained=False, num_classes=10)
surrogate_model.load_state_dict(torch.load('vgg16_cifar10.pth'))

with torch.no_grad():
    transfer_preds = torch.argmax(surrogate_model(adv_images), dim=1)
    transfer_rate = (transfer_preds != labels).float().mean()
    
print(f"Transfer Success Rate: {transfer_rate*100:.2f}%")
```

### Visualize Quantum Module Contributions

```python
# Generate visualization of per-module perturbations
qaeas.visualize_ensemble_attack(
    image=images[0],
    true_label=labels[0].item(),
    save_path='./results/ensemble_visualization.png'
)
```

### Kalman Filter Analysis

```python
# Plot Kalman filter weight evolution
qaeas.visualize_kalman_filter(stage='final')
```

## Evaluation Metrics

QAEAS evaluation includes:

- **Attack Success Rate (ASR)**: Percentage of successful misclassifications
- **L2/L∞ Norms**: Perturbation magnitude constraints
- **SSIM**: Structural similarity (target > 0.96)
- **LPIPS**: Learned perceptual similarity
- **PSNR**: Peak signal-to-noise ratio
- **Transferability**: Cross-architecture success rates
- **Defense Robustness**: Performance under adversarial training

## Reproducibility

All experiments use fixed seeds for reproducibility:

```python
SEEDS = [14, 38, 416, 911, 1369]
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epsilon (ε) | 8/255 | L∞ perturbation budget |
| Batch Size | 32 (train), 64 (eval) | Mini-batch size |
| Learning Rate (quantum) | 0.001 | Quantum circuit parameters |
| Learning Rate (ensemble) | 0.005 | Kalman ensemble weights |
| Epochs | 50 | Training iterations |
| Kalman λ | 0.95 | Temporal decay factor |
| Process Noise (Q) | 0.1 | State transition uncertainty |
| Measurement Noise (R) | 0.05 | Observation uncertainty |

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM, 16GB RAM
- **Recommended**: NVIDIA RTX 3090/4090, 32GB RAM
- **Quantum Simulation**: PennyLane default.qubit backend (CPU/GPU)

**Note**: QAEAS is evaluated using quantum circuit simulation. Deployment on physical NISQ hardware requires additional error mitigation techniques.

## Computational Cost

| Configuration | Time per Image | Overhead vs Classical |
|---------------|----------------|----------------------|
| Full QAEAS (4q/6L) | 11.3 ms | 1.66× |
| Classical Ensemble | 6.8 ms | 1.00× |
| QAEAS (3q/5L) | 8.7 ms | 1.28× |

*Measured on NVIDIA RTX 4090, PyTorch 2.0, PennyLane 0.32*

## Baseline Comparisons

QAEAS is compared against:

- **Classical Attacks**: FGSM, PGD-10, C&W, AutoAttack
- **Classical Ensemble**: 5-module architecture with classical neural networks
- **Target Models**: ResNet-20, EfficientNet-B0, WideResNet-34-10, ViT-S

## Citation

If you use QAEAS in your research, please cite:

```bibtex
@article{mohammadi2025qaeas,
  title={QAEAS: A Quantum Adaptive Ensemble Attack System Against Robust Deep Neural Networks},
  author={Mohammadi Ruzbahani, Ali and Yazdinejad, Abbas and Karimipour, Hadis},
  journal={Under Review},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research conducted at the Smart Cyber-Physical Systems (SCPS) Lab, University of Calgary
- Quantum circuit implementations built on [PennyLane](https://pennylane.ai/)
- Classical deep learning components use [PyTorch](https://pytorch.org/)

## Contact

- **Ali Mohammadi Ruzbahani**: ali.mohammadiruzbaha@ucalgary.ca
- **Project Lead**: Prof. Hadis Karimipour (hadis.karimipour@ucalgary.ca)

## Ethical Considerations

QAEAS is developed for academic research to:
- Assess quantum-enhanced adversarial capabilities under white-box settings
- Inform development of quantum-aware defense mechanisms
- Establish theoretical bounds for post-quantum AI security

This work is **not intended for malicious use**. Adversarial examples should only be generated for authorized testing and research purposes.

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory  
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: PennyLane circuit execution slow  
**Solution**: Enable `pennylane-lightning` for GPU-accelerated simulation

**Issue**: Kalman filter numerical instability  
**Solution**: Eigenvalue clamping is enabled by default; adjust `min_eigenvalue` threshold

**Issue**: Low attack success rate  
**Solution**: Ensure target model is properly trained (>85% clean accuracy on CIFAR-10)

## Future Work

- Deployment on physical NISQ hardware (IBM Quantum, Rigetti)
- Extension to ImageNet with patch-based quantum encoding
- Quantum-aware adversarial training defenses
- Integration with quantum error mitigation techniques

---

**Disclaimer**: This repository contains research code for academic purposes. The quantum circuits are simulated using classical hardware. Physical quantum hardware execution is not currently supported.
