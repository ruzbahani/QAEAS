import os
os.environ['OMP_NUM_THREADS'] = '4'
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import json
import gc
from scipy.ndimage import gaussian_filter
import traceback


class QuantumEnsembleManager(nn.Module):
    def __init__(self, target_model, device="cpu", epsilon=0.1, run_name=None):
        super(QuantumEnsembleManager, self).__init__()
        
        self.target_model = target_model
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.device = device
        self.epsilon = epsilon
        
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"quantum_ensemble_{timestamp}"
        else:
            self.run_name = run_name
            
        self.results_dir = os.path.join('quantum_ensemble_results', self.run_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.debug_stats = {
            'member_failures': [0, 0, 0, 0, 0],
            'kalman_filter_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'gradient_errors': 0
        }
        
        self.cache_enabled = True
        self.member_cache = [{} for _ in range(5)]
        self.cache_size = 50
        self.member_salt = [np.random.randint(10000) for _ in range(5)]
        
        print("Initializing ensemble members...")
        self.initialize_ensemble_members()
        
        print("Initializing Kalman filter...")
        self.initialize_kalman_filter()

        self.global_filter['P_history'] = [self.global_filter['P'].copy()]

        self.ensemble_weights = nn.Parameter(torch.ones(5) / 5)
        
        self.metrics = {
            'attack_success_rate': [],
            'perturbation_norm': [],
            'ensemble_weights': [],
            'member_performance': [[] for _ in range(5)],
            'kalman_gain': [],
            'loss_history': [],
            'train_times': []
        }
        
        self.setup_optimizers()
        
        print(f"Quantum ensemble initialized on {device}")

    def _generate_cache_key(self, member_index, images, target_class=None):
        if not hasattr(self, 'cache_enabled') or not self.cache_enabled:
            return None
            
        b = 0 if images.shape[0] > 0 else 0
        img = images[b]
        
        regions = [
            torch.mean(img[:, :10, :10]).item(),
            torch.mean(img[:, -10:, :10]).item(),
            torch.mean(img[:, :10, -10:]).item(),
            torch.mean(img[:, -10:, -10:]).item(),
            torch.mean(img[:, 11:21, 11:21]).item()
        ]
        
        region_str = "_".join([f"{r:.4f}" for r in regions])
        salt = getattr(self, 'member_salt', [member_index * 1000 + 42])[member_index % len(getattr(self, 'member_salt', [0]))]
        target_class_str = f"_tc_{target_class}" if target_class is not None else ""
        return f"member_{member_index}_salt_{salt}{target_class_str}_{hash(region_str)}"

    def manage_cache(self, enabled=None, clear=False, size=None):
        if enabled is not None:
            self.cache_enabled = enabled
            print(f"Ensemble caching {'enabled' if enabled else 'disabled'}")
        
        if clear:
            for i in range(len(self.member_cache)):
                self.member_cache[i].clear()
            
            for i, member in enumerate(self.ensemble_members):
                if hasattr(member, 'q_circuit') and hasattr(member.q_circuit, 'cache'):
                    member.q_circuit.cache.clear()
                if hasattr(member, 'clustering_cache'):
                    member.clustering_cache.clear()
            
            self.member_salt = [np.random.randint(10000) for _ in range(5)]
            print("All caches cleared and new salts generated")
        
        if size is not None:
            self.cache_size = size
            for member in self.ensemble_members:
                if hasattr(member, 'q_circuit'):
                    member.q_circuit.cache_size = size
            print(f"Cache size set to {size}")
        
        return {
            "enabled": self.cache_enabled,
            "size": self.cache_size,
            "member_salts": self.member_salt
        }
    

    def initialize_ensemble_members(self):
        try:
            from quantum_base_modifier import QuantumBaseModifier
            from quantum_texture_attacker import QuantumTextureAttacker
            from quantum_edge_disruptor import QuantumEdgeDisruptor
            from quantum_color_distorter import QuantumColorDistorter
            from quantum_focal_attacker import QuantumFocalAttacker
            
            print("  Initializing Base Modifier...")
            self.base_modifier = QuantumBaseModifier(
                n_qubits=4, 
                n_layers=6, 
                epsilon=self.epsilon,
                device=self.device,
                seed=1001
            )
            
            print("  Initializing Texture Attacker...")
            self.texture_attacker = QuantumTextureAttacker(
                n_qubits=4, 
                n_layers=5, 
                epsilon=self.epsilon,
                device=self.device,
                seed=2002
            )
            
            print("  Initializing Edge Disruptor...")
            self.edge_disruptor = QuantumEdgeDisruptor(
                n_qubits=3, 
                n_layers=4, 
                epsilon=self.epsilon,
                device=self.device,
                seed=3003
            )
            
            print("  Initializing Color Distorter...")
            self.color_distorter = QuantumColorDistorter(
                n_qubits=3, 
                n_layers=4, 
                epsilon=self.epsilon,
                device=self.device,
                seed=4004
            )
            
            print("  Initializing Focal Attacker...")
            self.focal_attacker = QuantumFocalAttacker(
                target_model=self.target_model,
                n_qubits=3, 
                n_layers=6, 
                n_focal_regions=6,
                epsilon=self.epsilon,
                device=self.device,
                seed=5005
            )
            
            self.ensemble_members = [
                self.base_modifier,
                self.texture_attacker,
                self.edge_disruptor,
                self.color_distorter,
                self.focal_attacker
            ]
            
            for member in self.ensemble_members:
                member.to(self.device)
            
            print("  Setting unique cache IDs for ensemble members...")
            for i, member in enumerate(self.ensemble_members):
                if hasattr(member, 'q_circuit'):
                    member.q_circuit.member_id = (i + 1) * 1000
                    print(f"    Member {i+1}: Cache ID {member.q_circuit.member_id} assigned")
                    
                if hasattr(member, 'clustering_cache'):
                    member._cache_salt = (i + 1) * 123
                    
        except Exception as e:
            print(f"Error initializing ensemble members: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize ensemble members")

    def initialize_kalman_filter(self):
        self.n_members = 5
        self.n_classes = 10
        
        initial_variance = 1.0/12.0
        stability_factor = 0.95
        
        self.member_filters = []
        for i in range(self.n_members):
            F = np.eye(self.n_classes) * stability_factor
            x = np.zeros(self.n_classes)
            P = np.eye(self.n_classes) * 3.0 * initial_variance
            process_noise_scale = initial_variance * (1.0 - stability_factor**2)
            Q = np.eye(self.n_classes) * process_noise_scale
            R = np.eye(self.n_classes) * 0.1
            
            self.member_filters.append({
                'x': x,
                'P': P,
                'F': F,
                'H': np.eye(self.n_classes),
                'R': R,
                'Q': Q,
                'z': np.zeros(self.n_classes),
                'history': [],
                'last_valid_state': x.copy(),
                'update_count': 0
            })
            
        global_F = np.eye(self.n_members) * stability_factor
        global_x = np.ones(self.n_members) / self.n_members
        global_P = np.eye(self.n_members) * initial_variance
        global_Q = np.eye(self.n_members) * (initial_variance * 0.5 * (1.0 - stability_factor**2))
        global_R = np.eye(self.n_members) * 0.07
        
        self.global_filter = {
            'x': global_x,
            'P': global_P,
            'F': global_F,
            'H': np.eye(self.n_members),
            'R': global_R,
            'Q': global_Q,
            'z': np.zeros(self.n_members),
            'gain_history': [],
            'P_history': [],
            'last_valid_state': global_x.copy(),
            'last_valid_P': global_P.copy()
        }
        
        self.global_filter['P_history'].append(self.global_filter['P'].copy())
        
        self.coupling_factor = 0.15
        
        self.measurement_history = []
        
        self.kalman_update_count = 0
        
        self.filter_config = {
            'min_eigenvalue_threshold': 1e-5,
            'clamp_factor': 0.1,
            'adaptive_q_factors': {
                'base': 0.5,
                'min': 0.05,
                'max': 0.6
            },
            'adaptive_r_factors': {
                'base': 2.0,
                'min': 0.05,
                'max': 0.3
            },
            'coupling_limits': {
                'min': 0.2,
                'max': 0.6
            },
            'recovery': {
                'enabled': True,
                'max_tries': 3
            }
        }

    def setup_optimizers(self):
        param_groups = []
        for i, member in enumerate(self.ensemble_members):
            param_groups.append({
                'params': member.parameters(),
                'lr': 0.001,
                'name': f'member_{i}'
            })
        
        param_groups.append({
            'params': [self.ensemble_weights],
            'lr': 0.005,
            'name': 'ensemble_weights'
        })
        
        self.optimizer = optim.Adam(param_groups)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )

    def _process_ensemble_member_train(self, member_index, member, images, target_class=None):
        cache_key = self._generate_cache_key(member_index, images, target_class)
        if cache_key is not None and hasattr(self, 'member_cache') and member_index < len(self.member_cache):
            if cache_key in self.member_cache[member_index]:
                self.debug_stats['cache_hits'] += 1
                cached_result = self.member_cache[member_index][cache_key]
                perturbation = cached_result[0].to(self.device).detach().requires_grad_(True)
                adv_images = cached_result[1].to(self.device)
                actual_perturbation = adv_images - images
                perturbation = actual_perturbation.detach().requires_grad_(True)

                return perturbation, adv_images
        
        if hasattr(self, 'debug_stats'):
            self.debug_stats['cache_misses'] += 1
        
        try:
            if member_index == 4:
                perturbation, adv_images, _ = member(images, target_class)
            else:
                try:
                    perturbation, adv_images = member(images, target_class)
                except (TypeError, ValueError):
                    perturbation, adv_images = member(images)
            
            if not perturbation.requires_grad:
                print(f"Warning: Member {member_index} output has no grad_fn - adding requires_grad=True")
                perturbation = perturbation.detach().requires_grad_(True)
            
            if hasattr(self, 'cache_enabled') and self.cache_enabled and cache_key is not None and hasattr(self, 'member_cache'):
                cache_size = getattr(self, 'cache_size', 50)
                if len(self.member_cache[member_index]) >= cache_size:
                    oldest_key = next(iter(self.member_cache[member_index]))
                    del self.member_cache[member_index][oldest_key]
                self.member_cache[member_index][cache_key] = (perturbation.detach().cpu(), adv_images.detach().cpu())
            actual_perturbation = adv_images - images 
            perturbation = actual_perturbation.detach().requires_grad_(True) 
            
            return perturbation, adv_images
                
        except Exception as e:
            if hasattr(self, 'debug_stats'):
                self.debug_stats['member_failures'][member_index] += 1
            print(f"Error in ensemble member {member_index}: {str(e)}")
            
            torch.manual_seed(member_index * 100 + 42)
            pattern_type = member_index % 5
            fallback_pert = torch.zeros_like(images, requires_grad=True)
            
            with torch.enable_grad():
                if pattern_type == 0:
                    for c in range(images.shape[1]):
                        pattern = torch.sin(torch.linspace(0, 6*np.pi, images.shape[3])).unsqueeze(0).unsqueeze(0).repeat(images.shape[0], 1, images.shape[2], 1) * 0.02
                        fallback_pert.data[:, c] = pattern.to(self.device)
                elif pattern_type == 1:
                    for c in range(images.shape[1]):
                        pattern = torch.sin(torch.linspace(0, 6*np.pi, images.shape[2])).unsqueeze(0).unsqueeze(-1).repeat(images.shape[0], 1, 1, images.shape[3]) * 0.02
                        fallback_pert.data[:, c] = pattern.to(self.device)
                elif pattern_type == 2:
                    x = torch.linspace(0, 4*np.pi, images.shape[3])
                    y = torch.linspace(0, 4*np.pi, images.shape[2])
                    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                    pattern = torch.sin(grid_x) * torch.sin(grid_y) * 0.02
                    fallback_pert.data = pattern.to(self.device).repeat(images.shape[0], images.shape[1], 1, 1)
                elif pattern_type == 3:
                    x = torch.linspace(-1, 1, images.shape[3])
                    y = torch.linspace(-1, 1, images.shape[2])
                    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
                    dist = torch.sqrt(grid_x**2 + grid_y**2)
                    pattern = torch.sin(dist * 8 * np.pi) * 0.02
                    fallback_pert.data = pattern.to(self.device).repeat(images.shape[0], images.shape[1], 1, 1)
                else:
                    fallback_pert.data = torch.randn_like(images, device=self.device) * 0.02
            
            if target_class is not None:
                channel_boost = target_class % 3
                fallback_pert.data[:, channel_boost] = fallback_pert.data[:, channel_boost] * 1.5
            
            assert fallback_pert.requires_grad, "Fallback perturbation must have requires_grad=True"
            fallback_adv = torch.clamp(images + fallback_pert, 0, 1)
            return fallback_pert, fallback_adv
                
        except Exception as e:
            if hasattr(self, 'debug_stats'):
                self.debug_stats['member_failures'][member_index] += 1
            print(f"Error in ensemble member {member_index}: {str(e)}")
            
            torch.manual_seed(member_index * 100 + 42)
            pattern_type = member_index % 5
            fallback_pert = torch.zeros_like(images, requires_grad=True)
            
            with torch.enable_grad():
                if pattern_type == 0:
                    for c in range(images.shape[1]):
                        pattern = torch.sin(torch.linspace(0, 6*np.pi, images.shape[3])).unsqueeze(0).unsqueeze(0).repeat(images.shape[0], 1, images.shape[2], 1) * 0.02
                        fallback_pert[:, c] = pattern.to(self.device)
                elif pattern_type == 1:
                    for c in range(images.shape[1]):
                        pattern = torch.sin(torch.linspace(0, 6*np.pi, images.shape[2])).unsqueeze(0).unsqueeze(-1).repeat(images.shape[0], 1, 1, images.shape[3]) * 0.02
                        fallback_pert[:, c] = pattern.to(self.device)
                elif pattern_type == 2:
                    x = torch.linspace(0, 4*np.pi, images.shape[3])
                    y = torch.linspace(0, 4*np.pi, images.shape[2])
                    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                    pattern = torch.sin(grid_x) * torch.sin(grid_y) * 0.02
                    fallback_pert = pattern.to(self.device).repeat(images.shape[0], images.shape[1], 1, 1)
                elif pattern_type == 3:
                    x = torch.linspace(-1, 1, images.shape[3])
                    y = torch.linspace(-1, 1, images.shape[2])
                    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
                    dist = torch.sqrt(grid_x**2 + grid_y**2)
                    pattern = torch.sin(dist * 8 * np.pi) * 0.02
                    fallback_pert = pattern.to(self.device).repeat(images.shape[0], images.shape[1], 1, 1)
                else:
                    fallback_pert = torch.randn_like(images, device=self.device) * 0.02
            
            if target_class is not None:
                channel_boost = target_class % 3
                fallback_pert[:, channel_boost] = fallback_pert[:, channel_boost] * 1.5
            
            fallback_adv = torch.clamp(images + fallback_pert, 0, 1)
            return fallback_pert, fallback_adv



    @torch.no_grad()
    def _process_ensemble_member_eval(self, member_index, member, images, target_class=None):
        cache_key = self._generate_cache_key(member_index, images, target_class)
        if cache_key is not None and cache_key in self.member_cache[member_index]:
            self.debug_stats['cache_hits'] += 1
            result = self.member_cache[member_index][cache_key]
            return result[0].to(self.device), result[1].to(self.device)
        
        if hasattr(self, 'debug_stats'):
            self.debug_stats['cache_misses'] += 1
        
        try:
            if member_index == 4:
                perturbation, adv_images, _ = member(images, target_class)
            else:
                try:
                    perturbation, adv_images = member(images, target_class)
                except (TypeError, ValueError):
                    perturbation, adv_images = member(images)
                
            if hasattr(self, 'cache_enabled') and self.cache_enabled and cache_key is not None and hasattr(self, 'member_cache'):
                cache_size = getattr(self, 'cache_size', 50)
                if len(self.member_cache[member_index]) >= cache_size:
                    oldest_key = next(iter(self.member_cache[member_index]))
                    del self.member_cache[member_index][oldest_key]
                
                self.member_cache[member_index][cache_key] = (perturbation.cpu(), adv_images.cpu())
                
            return perturbation, adv_images
            
        except Exception as e:
            self.debug_stats['member_failures'][member_index] += 1
            print(f"Error in ensemble member {member_index}: {str(e)}")
            
            torch.manual_seed(member_index * 100 + 42)
            
            if target_class is None:
                fallback_pert = torch.zeros_like(images) + 0.01
            else:
                fallback_pert = torch.zeros_like(images)
                channel_boost = target_class % 3
                fallback_pert[:, channel_boost] = 0.015
                fallback_pert[:, (channel_boost+1)%3] = 0.008
                fallback_pert[:, (channel_boost+2)%3] = 0.008
            
            fallback_adv = torch.clamp(images + fallback_pert, 0, 1)
            
            return fallback_pert, fallback_adv


    
    def forward_train(self, images, target_class=None):
        batch_size = images.shape[0]
        
        member_outputs = []
        perturbations = []
        
        pert1, adv1 = self._process_ensemble_member_train(0, self.base_modifier, images, target_class)
        member_outputs.append(adv1)
        perturbations.append(pert1)
        
        pert2, adv2 = self._process_ensemble_member_train(1, self.texture_attacker, images, target_class)
        member_outputs.append(adv2)
        perturbations.append(pert2)
        
        pert3, adv3 = self._process_ensemble_member_train(2, self.edge_disruptor, images, target_class)
        member_outputs.append(adv3)
        perturbations.append(pert3)
        
        pert4, adv4 = self._process_ensemble_member_train(3, self.color_distorter, images, target_class)
        member_outputs.append(adv4)
        perturbations.append(pert4)
        
        pert5, adv5 = self._process_ensemble_member_train(4, self.focal_attacker, images, target_class)
        member_outputs.append(adv5)
        perturbations.append(pert5)
        
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        combined_perturbation = torch.zeros_like(perturbations[0])
        adversarial_images = torch.zeros_like(images)
        for i, adv_img in enumerate(member_outputs):
                adversarial_images = adversarial_images + (weights[i] * adv_img)            
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        real_perturbation = adversarial_images - images  
        return adversarial_images, real_perturbation, member_outputs, weights
    
    @torch.no_grad()
    def forward_eval(self, images, target_class=None):
        batch_size = images.shape[0]
        
        member_outputs = []
        perturbations = []
        
        pert1, adv1 = self._process_ensemble_member_eval(0, self.base_modifier, images, target_class)
        member_outputs.append(adv1)
        perturbations.append(pert1)
        
        pert2, adv2 = self._process_ensemble_member_eval(1, self.texture_attacker, images, target_class)
        member_outputs.append(adv2)
        perturbations.append(pert2)
        
        pert3, adv3 = self._process_ensemble_member_eval(2, self.edge_disruptor, images, target_class)
        member_outputs.append(adv3)
        perturbations.append(pert3)
        
        pert4, adv4 = self._process_ensemble_member_eval(3, self.color_distorter, images, target_class)
        member_outputs.append(adv4)
        perturbations.append(pert4)
        
        pert5, adv5 = self._process_ensemble_member_eval(4, self.focal_attacker, images, target_class)
        member_outputs.append(adv5)
        perturbations.append(pert5)
        
        try:
            _, _, gradcam_maps = self.focal_attacker(images[:1], target_class)
            self.last_gradcam_maps = gradcam_maps
        except Exception:
            self.last_gradcam_maps = None
        
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        adversarial_images = torch.zeros_like(images)
        for i, adv_img in enumerate(member_outputs):
            adversarial_images = adversarial_images + (weights[i] * adv_img)
        
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        real_perturbation = adversarial_images - images
        return adversarial_images, real_perturbation, member_outputs, weights



    def forward(self, images, target_class=None):
        if self.training:
            adv_images, perturbation, member_outputs, weights = self.forward_train(images, target_class)
            with torch.no_grad():
                actual_perturbation = adv_images - images
                if torch.norm(perturbation - actual_perturbation) > 1e-5:
                    perturbation = actual_perturbation.detach().requires_grad_(True)
                    
            return adv_images, perturbation, member_outputs, weights
        else:
            adv_images, perturbation, member_outputs, weights = self.forward_eval(images, target_class)
            
            actual_perturbation = adv_images - images
            if torch.norm(perturbation - actual_perturbation) > 1e-5:
                perturbation = actual_perturbation.detach()
                
            return adv_images, perturbation, member_outputs, weights        



    def update_kalman_filter(self, original_images, adversarial_outputs, member_outputs, true_labels, target_class=None):
        try:
            batch_size = original_images.shape[0]
            
            member_measurements = []
            
            with torch.no_grad():
                original_outputs = self.target_model(original_images)
                original_preds = torch.argmax(original_outputs, dim=1)
            
            class_counts = np.zeros(self.n_classes)
            for label in true_labels:
                class_counts[label.item()] += 1
            
            class_instances = {i: (class_counts[i] > 0) for i in range(self.n_classes)}
            classes_in_batch = sum(class_instances.values())
            
            for i, member_adv_images in enumerate(member_outputs):
                with torch.no_grad():
                    member_preds = self.target_model(member_adv_images)
                    member_pred_labels = torch.argmax(member_preds, dim=1)
                    
                success_counts = np.zeros(self.n_classes)
                sample_counts = np.zeros(self.n_classes)
                
                for b in range(batch_size):
                    true_label = true_labels[b].item()
                    
                    if target_class is None:
                        success = float(original_preds[b].item() != member_pred_labels[b].item())
                    else:
                        success = float(member_pred_labels[b].item() == target_class)
                    
                    success_counts[true_label] += success
                    sample_counts[true_label] += 1
                
                class_performance = np.zeros(self.n_classes)
                for c in range(self.n_classes):
                    if sample_counts[c] > 0:
                        class_performance[c] = success_counts[c] / sample_counts[c]
                
                member_measurements.append(class_performance)
            
            member_states = []
            
            for i, member_filter in enumerate(self.member_filters):
                member_filter['z'] = member_measurements[i]
                member_filter['update_count'] += 1
                
                x_pred = np.dot(member_filter['F'], member_filter['x'])
                P_pred = np.dot(np.dot(member_filter['F'], member_filter['P']), 
                                member_filter['F'].T) + member_filter['Q']
                
                P_pred = (P_pred + P_pred.T) / 2
                
                eigenvals, eigenvecs = np.linalg.eigh(P_pred)
                
                max_eigenval = np.max(eigenvals)
                min_threshold = max(
                    self.filter_config['min_eigenvalue_threshold'],
                    max_eigenval * self.filter_config['clamp_factor']
                )
                
                clamped_eigenvals = np.maximum(eigenvals, min_threshold)
                P_pred = np.dot(eigenvecs, np.dot(np.diag(clamped_eigenvals), eigenvecs.T))
                
                predicted_measurement = np.dot(member_filter['H'], x_pred)
                innovation = member_filter['z'] - predicted_measurement
                
                S = np.dot(np.dot(member_filter['H'], P_pred), 
                            member_filter['H'].T) + member_filter['R']
                
                S = (S + S.T) / 2
                eigenvals, eigenvecs = np.linalg.eigh(S)
                max_eigenval = np.max(eigenvals)
                min_threshold = max(
                    self.filter_config['min_eigenvalue_threshold'],
                    max_eigenval * self.filter_config['clamp_factor']
                )
                clamped_eigenvals = np.maximum(eigenvals, min_threshold)
                S = np.dot(eigenvecs, np.dot(np.diag(clamped_eigenvals), eigenvecs.T))
                
                U, s, Vh = np.linalg.svd(S, full_matrices=False)
                s_inv = 1.0 / s
                S_inv = np.dot(Vh.T, np.dot(np.diag(s_inv), U.T))
                K = np.dot(P_pred, np.dot(member_filter['H'].T, S_inv))
                
                member_filter['x'] = x_pred + np.dot(K, innovation)
                
                I = np.eye(self.n_classes)
                KH = np.dot(K, member_filter['H'])
                I_KH = I - KH
                member_filter['P'] = np.dot(I_KH, np.dot(P_pred, I_KH.T)) + np.dot(K, np.dot(member_filter['R'], K.T))
                
                member_filter['last_valid_state'] = member_filter['x'].copy()
                
                member_filter['history'].append(member_filter['x'].copy())
                member_states.append(member_filter['x'])
            
            global_measurement = np.zeros(self.n_members)
            
            class_weights = class_counts / np.sum(class_counts)
            
            for i in range(self.n_members):
                weighted_performance = 0
                weight_sum = 0
                
                for c in range(self.n_classes):
                    if class_counts[c] > 0:
                        weighted_performance += member_states[i][c] * class_weights[c]
                        weight_sum += class_weights[c]
                
                if weight_sum > 0:
                    global_measurement[i] = weighted_performance / weight_sum
                else:
                    global_measurement[i] = 0
            
            self.measurement_history.append(global_measurement.copy())
            
            self.global_filter['z'] = global_measurement
                    
            performance_variance = np.var(global_measurement)
            
            q_factor = self.filter_config['adaptive_q_factors']
            adaptive_q_scale = np.clip(
                performance_variance * q_factor['base'],
                q_factor['min'],
                q_factor['max']
            )
            
            self.global_filter['Q'] = np.eye(self.n_members) * adaptive_q_scale
            
            if self.kalman_update_count > 0 and hasattr(self, 'prev_measurement'):
                measurement_change = np.mean(np.abs(global_measurement - self.prev_measurement))
                
                r_factor = self.filter_config['adaptive_r_factors']
                adaptive_r_scale = np.clip(
                    measurement_change * r_factor['base'],
                    r_factor['min'],
                    r_factor['max']
                )
                
                self.global_filter['R'] = np.eye(self.n_members) * adaptive_r_scale
                
                if hasattr(self, 'metrics'):
                    self.metrics['adaptive_r_scale'] = self.metrics.get('adaptive_r_scale', []) + [adaptive_r_scale]
            
            self.prev_measurement = global_measurement.copy()
            
            x_pred = np.dot(self.global_filter['F'], self.global_filter['x'])
            P_pred = np.dot(np.dot(self.global_filter['F'], self.global_filter['P']), 
                            self.global_filter['F'].T) + self.global_filter['Q']
            
            P_pred = (P_pred + P_pred.T) / 2
            eigenvals, eigenvecs = np.linalg.eigh(P_pred)
            max_eigenval = np.max(eigenvals)
            min_threshold = max(
                self.filter_config['min_eigenvalue_threshold'],
                max_eigenval * self.filter_config['clamp_factor']
            )
            clamped_eigenvals = np.maximum(eigenvals, min_threshold)
            P_pred = np.dot(eigenvecs, np.dot(np.diag(clamped_eigenvals), eigenvecs.T))
            
            innovation = self.global_filter['z'] - np.dot(self.global_filter['H'], x_pred)
            
            S = np.dot(np.dot(self.global_filter['H'], P_pred), 
                    self.global_filter['H'].T) + self.global_filter['R']
            
            S = (S + S.T) / 2
            eigenvals, eigenvecs = np.linalg.eigh(S)
            max_eigenval = np.max(eigenvals)
            min_threshold = max(
                self.filter_config['min_eigenvalue_threshold'],
                max_eigenval * self.filter_config['clamp_factor']
            )
            clamped_eigenvals = np.maximum(eigenvals, min_threshold)
            S = np.dot(eigenvecs, np.dot(np.diag(clamped_eigenvals), eigenvecs.T))
            
            U, s, Vh = np.linalg.svd(S, full_matrices=False)
            s_inv = 1.0 / s
            S_inv = np.dot(Vh.T, np.dot(np.diag(s_inv), U.T))
            K = np.dot(P_pred, np.dot(self.global_filter['H'].T, S_inv))
            
            self.global_filter['x'] = x_pred + np.dot(K, innovation)
            
            I = np.eye(self.n_members)
            KH = np.dot(K, self.global_filter['H'])
            I_KH = I - KH
            self.global_filter['P'] = np.dot(I_KH, np.dot(P_pred, I_KH.T)) + \
                                    np.dot(K, np.dot(self.global_filter['R'], K.T))
            
            self.global_filter['last_valid_state'] = self.global_filter['x'].copy()
            self.global_filter['last_valid_P'] = self.global_filter['P'].copy()
            
            gain_magnitude = np.linalg.norm(K)
            self.global_filter['gain_history'].append(gain_magnitude)
            
            performance_range = np.max(global_measurement) - np.min(global_measurement)
            
            coupling_limits = self.filter_config['coupling_limits']
            adaptive_coupling = np.clip(
                self.coupling_factor + 0.25 * performance_range,
                coupling_limits['min'],
                coupling_limits['max']
            )
            
            for i, member_filter in enumerate(self.member_filters):
                influence = self.global_filter['x'][i]
                
                member_filter['x'] = (1 - adaptive_coupling) * member_filter['x'] + \
                                    adaptive_coupling * influence
            
            x_centered = self.global_filter['x'] - np.mean(self.global_filter['x'])
            exp_weights = np.exp(x_centered)
            updated_weights = exp_weights / np.sum(exp_weights)
            
            updated_weights = np.maximum(updated_weights, 0.01)
            updated_weights = updated_weights / np.sum(updated_weights)
            
            weights_tensor = torch.tensor(updated_weights, dtype=torch.float32, device=self.device)
            
            self.kalman_update_count += 1
            
            attack_type = "targeted" if target_class is not None else "non-targeted"
            
            if hasattr(self, 'metrics'):
                self.metrics['ensemble_weights'].append(updated_weights.tolist())
                self.metrics['kalman_gain'].append(gain_magnitude)
                self.metrics['adaptive_q_scale'] = self.metrics.get('adaptive_q_scale', []) + [adaptive_q_scale]
                self.metrics['adaptive_coupling'] = self.metrics.get('adaptive_coupling', []) + [adaptive_coupling]
                
                if 'attack_type' not in self.metrics:
                    self.metrics['attack_type'] = []
                self.metrics['attack_type'].append(attack_type)
                
                for i in range(self.n_members):
                    self.metrics['member_performance'][i].append(global_measurement[i])
            
            if 'P_history' in self.global_filter:
                self.global_filter['P_history'].append(self.global_filter['P'].copy())

            return weights_tensor
                
        except Exception as e:
            self.debug_stats['kalman_filter_errors'] += 1
            print(f"Error in Kalman filter update: {e}")
            
            recovery_enabled = self.filter_config.get('recovery', {}).get('enabled', True)
            
            if recovery_enabled and 'last_valid_state' in self.global_filter:
                print("Attempting recovery using last valid state...")
                try:
                    last_valid_x = self.global_filter['last_valid_state']
                    x_centered = last_valid_x - np.mean(last_valid_x)
                    exp_weights = np.exp(x_centered)
                    recovered_weights = exp_weights / np.sum(exp_weights)
                    
                    recovered_weights = np.maximum(recovered_weights, 0.01)
                    recovered_weights = recovered_weights / np.sum(recovered_weights)
                    
                    return torch.tensor(recovered_weights, dtype=torch.float32, device=self.device)
                except Exception as recovery_error:
                    print(f"Recovery failed: {recovery_error}")
                    
            print("Using uniform fallback weights")
            return torch.ones(self.n_members, device=self.device) / self.n_members

    @torch.no_grad()
    def compute_attack_success(self, original_outputs, adversarial_outputs, true_labels, target_class=None):
        batch_size = original_outputs.shape[0]
        
        orig_preds = torch.argmax(original_outputs, dim=1)
        adv_preds = torch.argmax(adversarial_outputs, dim=1)

        if target_class is None:
            success = (orig_preds != adv_preds).float()
        else:
            target_labels = torch.full_like(orig_preds, target_class)
            success = (adv_preds == target_labels).float()
        
        success_rate = success.sum() / batch_size
        return success_rate.item() * 100.0

    def train_step(self, images, labels, target_class=None):
        if target_class is not None and isinstance(target_class, str):
            try:
                target_class = int(target_class)
            except ValueError:
                print(f"Error: target_class must be a number, got '{target_class}'. Switching to non-targeted attack.")
                target_class = None

        batch_size = images.shape[0]
        step_start_time = time.time()

        with torch.no_grad():
            original_outputs = self.target_model(images)
            original_preds = torch.argmax(original_outputs, dim=1)

        if not self.training:
            self.train()
        
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No parameters require gradients!")

        adversarial_images, perturbation, member_outputs, weights = self.forward_train(images, target_class)
        with torch.no_grad():
            actual_perturbation = adversarial_images - images
            perturbation_diff = torch.norm((perturbation - actual_perturbation).view(-1), p=2).item()
            if perturbation_diff > 1e-5:
                print(f"Warning: Fixing perturbation in train_step. Difference: {perturbation_diff:.8f}")
                perturbation = actual_perturbation.detach().requires_grad_(True)

        if not perturbation.requires_grad:
            print("Warning in train_step: perturbation has no grad_fn")
            self.debug_stats['gradient_errors'] += 1

        with torch.no_grad():
            adversarial_outputs = self.target_model(adversarial_images)
            adversarial_preds = torch.argmax(adversarial_outputs, dim=1)         

        batch_attack_success = self.compute_attack_success(original_outputs, adversarial_outputs, labels, target_class)
        
        label_change_count = (original_preds != adversarial_preds).sum().item()
        label_change_rate = (label_change_count / batch_size) * 100
        
        class_changes = {}
        for i in range(batch_size):
            orig_label = original_preds[i].item()
            adv_label = adversarial_preds[i].item()
            true_label = labels[i].item()
            
            if orig_label not in class_changes:
                class_changes[orig_label] = {'total': 0, 'changed': 0}
            
            class_changes[orig_label]['total'] += 1
            if orig_label != adv_label:
                class_changes[orig_label]['changed'] += 1
        
        perturbation_norm = torch.norm(perturbation, p=2, dim=(1, 2, 3)).mean()

        member_perturbations = []
        for i, member_adv in enumerate(member_outputs):
            member_pert = member_adv - images
            member_perturbations.append(member_pert)

        diversity_loss = 0.0
        diversity_weight = 2.0
        pair_count = 0
        
        for i in range(len(member_perturbations)):
            for j in range(i+1, len(member_perturbations)):
                for b in range(batch_size):
                    pert_i = member_perturbations[i][b].flatten()
                    pert_j = member_perturbations[j][b].flatten()
                    
                    similarity = F.cosine_similarity(pert_i.unsqueeze(0), pert_j.unsqueeze(0), dim=1)
                    
                    diversity_loss += similarity
                    pair_count += 1
        
        if pair_count > 0:
            diversity_loss = diversity_loss / pair_count

        if perturbation.requires_grad:
            if target_class is None:
                attack_loss = -F.cross_entropy(adversarial_outputs, labels)
            else:
                if isinstance(target_class, str):
                    try:
                        target_class_int = int(target_class)
                    except ValueError:
                        print(f"Error: Invalid target_class '{target_class}'. Switching to non-targeted attack.")
                        attack_loss = -F.cross_entropy(adversarial_outputs, labels)
                    else:
                        target_labels = torch.full_like(labels, target_class_int)
                        attack_loss = F.cross_entropy(adversarial_outputs, target_labels)
                else:
                    target_labels = torch.full_like(labels, target_class)
                    attack_loss = F.cross_entropy(adversarial_outputs, target_labels)

            l2_loss = perturbation_norm
            variation_loss = -torch.mean(torch.var(adversarial_images, dim=[2, 3]))
            final_div_loss = -diversity_loss  

            alpha_attack = 1.0
            beta_l2 = 100.0
            gamma_var = 10.0
            delta_div = diversity_weight

            loss = (alpha_attack * attack_loss
                    + beta_l2     * l2_loss
                    + gamma_var   * variation_loss
                    + delta_div   * final_div_loss
            )
        else:
            loss = torch.mean(self.ensemble_weights**2)
            print("Warning: Using fallback loss calculation")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            new_weights = self.update_kalman_filter(images, adversarial_outputs, member_outputs, labels)
            self.ensemble_weights.copy_(new_weights)

        self.metrics['attack_success_rate'].append(batch_attack_success)
        self.metrics['perturbation_norm'].append(perturbation_norm.item())
        self.metrics['loss_history'].append(loss.item())
        self.metrics['train_times'].append(time.time() - step_start_time)

        if 'label_change_rate' not in self.metrics:
            self.metrics['label_change_rate'] = []
        self.metrics['label_change_rate'].append(label_change_rate)
        
        if 'per_class_success' not in self.metrics:
            self.metrics['per_class_success'] = {}
        for cls, stats in class_changes.items():
            if cls not in self.metrics['per_class_success']:
                self.metrics['per_class_success'][cls] = []
            
            success_rate = (stats['changed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            self.metrics['per_class_success'][cls].append(success_rate)

        if 'diversity_loss' not in self.metrics:
            self.metrics['diversity_loss'] = []
        self.metrics['diversity_loss'].append(diversity_loss.item())

        del original_outputs
        if not member_outputs[0].requires_grad:
            del member_outputs

        return loss, batch_attack_success, perturbation_norm.item()

    def train_model(self, train_loader, val_loader, epochs=50, save_every=5, patience=15, 
                    vis_every=5, target_class=None):
        print(f"Starting training for {epochs} epochs (early stopping patience: {patience})...")
        if target_class is not None and isinstance(target_class, str):
            try:
                target_class = int(target_class)
            except ValueError:
                print(f"[!] Warning: target_class '{target_class}' is not a number. Switching to non-targeted.")
                target_class = None

        best_success_rate = 0.0
        best_epoch = 0
        patience_counter = 0
        
        no_improvement_count = 0
        self.manage_cache(enabled=True)
    
        for member in self.ensemble_members:
            if hasattr(member, 'q_circuit'):
                member.q_circuit.cache.clear()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            if epoch > 0 and epoch % 5 == 0:
                print(f"Clearing caches at epoch {epoch}...")
                for member in self.ensemble_members:
                    if hasattr(member, 'q_circuit'):
                        member.q_circuit.cache.clear()
                    if hasattr(member, 'clustering_cache'):
                        member.clustering_cache = {}

            self.train()
            train_loss = 0.0
            train_success_rate = 0.0
            train_perturbation_norm = 0.0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch_idx, (images, labels) in enumerate(train_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                try:
                    images_copy = images.clone().detach()
                    
                    loss, success_rate, perturbation_norm = self.train_step(
                                        images_copy, 
                                        labels, 
                                        target_class=target_class  
                                    )                    
                    
                    train_loss += loss.item()
                    train_success_rate += success_rate
                    train_perturbation_norm += perturbation_norm
                    
                    del images_copy
                    
                except Exception as e:
                    print(f"Error in training step: {e}")
                    print(traceback.format_exc())
                    continue
                
                train_bar.set_postfix({
                    'loss': f"{train_loss/(batch_idx+1):.4f}",
                    'success': f"{train_success_rate/(batch_idx+1):.2f}%",
                    'pert_norm': f"{train_perturbation_norm/(batch_idx+1):.4f}"
                })
                
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            train_loss /= len(train_loader)
            train_success_rate /= len(train_loader)
            train_perturbation_norm /= len(train_loader)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.eval()
            val_loss = 0.0
            val_success_rate = 0.0
            val_perturbation_norm = 0.0
            
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_bar):
                    try:
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        original_outputs = self.target_model(images)
                        
                        adversarial_images, perturbation, _, _ = self.forward_eval(images)
                        
                        adversarial_outputs = self.target_model(adversarial_images)
                        
                        success_rate = self.compute_attack_success(original_outputs, adversarial_outputs, labels, target_class)
                        perturbation_norm = torch.norm(perturbation, p=2, dim=(1, 2, 3)).mean().item()
                        
                        l2_loss = perturbation_norm
                        variation_loss = -torch.mean(torch.var(adversarial_images, dim=[2, 3]))
                        loss = 100.0 * l2_loss + 10.0 * variation_loss
                        
                        val_loss += loss.item()
                        val_success_rate += success_rate
                        val_perturbation_norm += perturbation_norm
                        
                    except Exception as e:
                        print(f"Error in validation: {e}")
                        continue
                    
                    val_bar.set_postfix({
                        'loss': f"{val_loss/(batch_idx+1):.4f}",
                        'success': f"{val_success_rate/(batch_idx+1):.2f}%",
                        'pert_norm': f"{val_perturbation_norm/(batch_idx+1):.4f}"
                    })
                    
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if len(val_loader) > 0:
                val_loss /= len(val_loader)
                val_success_rate /= len(val_loader)
                val_perturbation_norm /= len(val_loader)
            
                self.scheduler.step(val_success_rate)
            
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
            print(f"Train - Loss: {train_loss:.4f}, Success: {train_success_rate:.2f}%, Perturbation: {train_perturbation_norm:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Success: {val_success_rate:.2f}%, Perturbation: {val_perturbation_norm:.4f}")
            
            print(f"Ensemble Weights: {self.ensemble_weights.detach().cpu().numpy()}")
            
            if 'adaptive_q_scale' in self.metrics and self.metrics['adaptive_q_scale']:
                latest_q = self.metrics['adaptive_q_scale'][-1]
                print(f"Adaptive Q scale: {latest_q:.4f}")
                
            if 'adaptive_r_scale' in self.metrics and self.metrics['adaptive_r_scale']:
                latest_r = self.metrics['adaptive_r_scale'][-1]
                print(f"Adaptive R scale: {latest_r:.4f}")
                
            if 'learning_rate' in self.metrics and self.metrics['learning_rate']:
                latest_lr = self.metrics['learning_rate'][-1]
                print(f"Kalman learning rate: {latest_lr:.4f}")
                
            if 'adaptive_coupling' in self.metrics and self.metrics['adaptive_coupling']:
                latest_coupling = self.metrics['adaptive_coupling'][-1]
                print(f"Adaptive coupling: {latest_coupling:.4f}")
            
            print(f"Gradient Errors: {self.debug_stats['gradient_errors']}")
            
            if val_success_rate > best_success_rate:
                best_success_rate = val_success_rate
                best_epoch = epoch + 1
                self.save_model('best_model.pth')
                print(f"New best model saved with success rate: {best_success_rate:.2f}%")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} epochs. Best success rate: {best_success_rate:.2f}% (epoch {best_epoch})")
            
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs with no improvement")
                break
            
            if (epoch + 1) % save_every == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
                self.save_visualizations(epoch + 1)
                self.save_metrics()

            if (epoch + 1) % vis_every == 0:
                self.visualize_kalman_filter(f'epoch_{epoch+1}')            
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        self.save_model('final_model.pth')
        self.save_visualizations('final')
        self.save_metrics()
        self.visualize_kalman_filter('final')
        
        print("\n" + "="*50)
        print(f"Training completed - Best success rate: {best_success_rate:.2f}% at epoch {best_epoch}")
        print(f"Final ensemble weights: {self.ensemble_weights.detach().cpu().numpy()}")
        print("="*50)
        
        return self.metrics

    def save_model(self, filename):
        save_path = os.path.join(self.results_dir, filename)
        
        try:
            state_dict = {
                'ensemble_weights': self.ensemble_weights.detach().cpu(),
                'base_modifier': self.base_modifier.state_dict(),
                'texture_attacker': self.texture_attacker.state_dict(),
                'edge_disruptor': self.edge_disruptor.state_dict(),
                'color_distorter': self.color_distorter.state_dict(),
                'focal_attacker': self.focal_attacker.state_dict(),
                'global_filter': self.global_filter,
                'member_filters': self.member_filters,
                'metrics': self.metrics,
                'debug_stats': self.debug_stats,
                'epoch': len(self.metrics['loss_history']),
                'timestamp': datetime.now().isoformat()
            }
            
            temp_path = save_path + ".tmp"
            torch.save(state_dict, temp_path)
            
            if os.path.exists(save_path):
                os.replace(temp_path, save_path)
            else:
                os.rename(temp_path, save_path)
                
            print(f"Model saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filename):
        load_path = os.path.join(self.results_dir, filename)
        alt_path = os.path.join('attacker_models', 'best_model.pth')
        if os.path.exists(alt_path):
            print(f"Trying alternate path: {alt_path}")
            load_path = alt_path 
        else:
            print(f"Alternate model file {alt_path} also not found!")
            return False        

        
        try:
            state_dict = torch.load(load_path, map_location=self.device)
            
            self.ensemble_weights.data = state_dict['ensemble_weights'].to(self.device)
            
            self.base_modifier.load_state_dict(state_dict['base_modifier'])
            self.texture_attacker.load_state_dict(state_dict['texture_attacker'])
            self.edge_disruptor.load_state_dict(state_dict['edge_disruptor'])
            self.color_distorter.load_state_dict(state_dict['color_distorter'])
            self.focal_attacker.load_state_dict(state_dict['focal_attacker'])
            
            self.global_filter = state_dict['global_filter']
            self.member_filters = state_dict['member_filters']
            
            self.metrics = state_dict['metrics']
            
            if 'debug_stats' in state_dict:
                self.debug_stats = state_dict['debug_stats']
                
            print(f"Model loaded from {load_path} (epoch {state_dict['epoch']})")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_metrics(self):
        metrics_path = os.path.join(self.results_dir, 'training_metrics.json')
        
        try:
            clean_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    clean_metrics[key] = [arr.tolist() for arr in value]
                elif isinstance(value, np.ndarray):
                    clean_metrics[key] = value.tolist()
                else:
                    clean_metrics[key] = value
            
            clean_metrics['debug_stats'] = self.debug_stats
            
            clean_metrics['timestamp'] = datetime.now().isoformat()
            
            temp_path = metrics_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(clean_metrics, f, indent=2)
                
            if os.path.exists(metrics_path):
                os.replace(temp_path, metrics_path)
            else:
                os.rename(temp_path, metrics_path)
                
            print(f"Metrics saved to {metrics_path}")
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def save_visualizations(self, epoch_or_name):
        vis_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['attack_success_rate'])
            plt.title('Attack Success Rate')
            plt.xlabel('Batch')
            plt.ylabel('Success Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, f'attack_success_rate_{epoch_or_name}.png'))
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['perturbation_norm'])
            plt.title('Perturbation L2 Norm')
            plt.xlabel('Batch')
            plt.ylabel('L2 Norm')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, f'perturbation_norm_{epoch_or_name}.png'))
            plt.close()
            
            if self.metrics['ensemble_weights']:
                weights_array = np.array(self.metrics['ensemble_weights'])
                plt.figure(figsize=(10, 6))
                for i in range(self.n_members):
                    plt.plot(weights_array[:, i], label=f'Member {i+1}')
                plt.title('Ensemble Weights Evolution')
                plt.xlabel('Update')
                plt.ylabel('Weight')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(vis_dir, f'ensemble_weights_{epoch_or_name}.png'))
                plt.close()
            
            member_perf = self.metrics['member_performance']
            if all(member_perf) and all(perf for perf in member_perf):
                plt.figure(figsize=(10, 6))
                for i in range(self.n_members):
                    plt.plot(member_perf[i], label=f'Member {i+1}')
                plt.title('Individual Member Performance')
                plt.xlabel('Update')
                plt.ylabel('Performance')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(vis_dir, f'member_performance_{epoch_or_name}.png'))
                plt.close()
            
            if self.metrics['kalman_gain']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['kalman_gain'])
                plt.title('Kalman Gain Magnitude')
                plt.xlabel('Update')
                plt.ylabel('Gain Magnitude')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(vis_dir, f'kalman_gain_{epoch_or_name}.png'))
                plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['loss_history'])
            plt.title('Loss History')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, f'loss_history_{epoch_or_name}.png'))
            plt.close()
            
            if self.metrics['train_times']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics['train_times'])
                plt.title('Training Step Times')
                plt.xlabel('Batch')
                plt.ylabel('Time (seconds)')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(vis_dir, f'train_times_{epoch_or_name}.png'))
                plt.close()
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in visualizations: {e}")
    



    @torch.no_grad()
    def visualize_ensemble_attack(self, image, label, filename=None, target_class=None):
        try:
            img_batch = image.unsqueeze(0).to(self.device)
            label_batch = torch.tensor([label], device=self.device)
            
            self.eval()
            
            original_output = self.target_model(img_batch)
            original_pred = torch.argmax(original_output, dim=1)[0].item()
            
            adv_images, combined_pert, member_outputs, weights = self.forward_eval(img_batch, target_class)
            
            ensemble_output = self.target_model(adv_images)
            ensemble_pred = torch.argmax(ensemble_output, dim=1)[0].item()
            
            member_preds = []
            for member_output in member_outputs:
                member_pred = self.target_model(member_output)
                member_preds.append(torch.argmax(member_pred, dim=1)[0].item())
            
            img_np = image.detach().cpu().permute(1, 2, 0).numpy()
            adv_np = adv_images[0].detach().cpu().permute(1, 2, 0).numpy()
            combined_pert_np = combined_pert[0].detach().cpu().permute(1, 2, 0).numpy()
            
            pert_amplified = np.clip(combined_pert_np * 10, -1, 1)
            
            member_perts = []
            member_advs = []
            for i, member_output in enumerate(member_outputs):
                member_pert = member_output[0] - img_batch[0]
                member_perts.append(member_pert.detach().cpu().permute(1, 2, 0).numpy())
                member_advs.append(member_output[0].detach().cpu().permute(1, 2, 0).numpy())
            
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            
            fig = plt.figure(figsize=(20, 12))
            grid_spec = fig.add_gridspec(3, 6)
            
            ax_orig = fig.add_subplot(grid_spec[0, 0])
            ax_orig.imshow(np.clip(img_np, 0, 1))
            ax_orig.set_title(f"Original\nPred: {class_names[original_pred]}\nTrue: {class_names[label]}")
            ax_orig.axis("off")
            
            ax_pert = fig.add_subplot(grid_spec[0, 1])
            ax_pert.imshow(np.clip(pert_amplified + 0.5, 0, 1))
            ax_pert.set_title(f"Combined Perturbation\n(Amplified 10x)")
            ax_pert.axis("off")
            
            ax_adv = fig.add_subplot(grid_spec[0, 2])
            ax_adv.imshow(np.clip(adv_np, 0, 1))
            
            if target_class is None:
                success_text = "SUCCESS" if ensemble_pred != original_pred else "FAILED"
            else:
                success_text = "SUCCESS" if ensemble_pred == target_class else "FAILED"
                
            ax_adv.set_title(f"Ensemble Attack\nPred: {class_names[ensemble_pred]}\n{success_text}")
            ax_adv.axis("off")
            
            ax_weights = fig.add_subplot(grid_spec[0, 3:])
            bars = ax_weights.bar(
                ['Base', 'Texture', 'Edge', 'Color', 'Focal'],
                weights.detach().cpu().numpy(),
                color=['royalblue', 'forestgreen', 'firebrick', 'darkorange', 'darkviolet']
            )
            ax_weights.set_title("Ensemble Member Weights")
            ax_weights.set_ylim(0, 1)
            for bar in bars:
                height = bar.get_height()
                ax_weights.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center'
                )
            
            member_names = ['Base Modifier', 'Texture Attacker', 'Edge Disruptor', 
                        'Color Distorter', 'Focal Attacker']
            member_colors = ['royalblue', 'forestgreen', 'firebrick', 'darkorange', 'darkviolet']
            
            for i in range(5):
                ax_m_pert = fig.add_subplot(grid_spec[1, i])
                member_pert_amplified = np.clip(member_perts[i] * 10, -1, 1)
                ax_m_pert.imshow(np.clip(member_pert_amplified + 0.5, 0, 1))
                ax_m_pert.set_title(f"{member_names[i]}\nPerturbation (10x)")
                ax_m_pert.axis("off")
                
                ax_m_adv = fig.add_subplot(grid_spec[2, i])
                ax_m_adv.imshow(np.clip(member_advs[i], 0, 1))
                





                if target_class is None:



                    member_success = "SUCCESS" if member_preds[i] != original_pred else "FAILED"
                else:
                    member_success = "SUCCESS" if member_preds[i] == target_class else "FAILED"
                    
                ax_m_adv.set_title(f"Pred: {class_names[member_preds[i]]}\n{member_success}")
                ax_m_adv.axis("off")
                
                for ax in [ax_m_pert, ax_m_adv]:
                    ax.spines['top'].set_color(member_colors[i])
                    ax.spines['bottom'].set_color(member_colors[i])
                    ax.spines['left'].set_color(member_colors[i])
                    ax.spines['right'].set_color(member_colors[i])
                    ax.spines['top'].set_linewidth(5)
                    ax.spines['bottom'].set_linewidth(5)
                    ax.spines['left'].set_linewidth(5)
                    ax.spines['right'].set_linewidth(5)
            
            if hasattr(self, 'last_gradcam_maps') and self.last_gradcam_maps is not None:
                gradcam = self.last_gradcam_maps[0].detach().cpu().numpy()
                
                ax_gradcam = fig.add_subplot(grid_spec[1, 5])
                ax_gradcam.imshow(np.clip(img_np, 0, 1))
                ax_gradcam.imshow(gradcam, cmap='jet', alpha=0.5)
                ax_gradcam.set_title("Grad-CAM Heatmap")
                ax_gradcam.axis("off")
                
                ax_gradcam.spines['top'].set_color(member_colors[4])
                ax_gradcam.spines['bottom'].set_color(member_colors[4])
                ax_gradcam.spines['left'].set_color(member_colors[4])
                ax_gradcam.spines['right'].set_color(member_colors[4])
                ax_gradcam.spines['top'].set_linewidth(5)
                ax_gradcam.spines['bottom'].set_linewidth(5)
                ax_gradcam.spines['left'].set_linewidth(5)
                ax_gradcam.spines['right'].set_linewidth(5)
            
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)
                
            del img_batch, label_batch, original_output, adv_images
            del combined_pert, member_outputs, ensemble_output, member_perts, member_advs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in ensemble visualization: {e}")
            if filename:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Visualization error: {str(e)}", 
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(filename)
                plt.close()

    
    @torch.no_grad()
    def visualize_kalman_filter(self, filename_prefix=None):
        if filename_prefix:
            vis_dir = os.path.join(self.results_dir, 'kalman_visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
        plt.figure(figsize=(12, 8))
        
        if 'ensemble_weights' in self.metrics and len(self.metrics['ensemble_weights']) > 0:
            weights_array = np.array(self.metrics['ensemble_weights'])
            for i in range(self.n_members):
                plt.plot(weights_array[:, i], label=f'Member {i+1}')
                
            plt.title('Ensemble Weights Evolution', fontsize=14)
            plt.xlabel('Updates', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            if filename_prefix:
                plt.savefig(os.path.join(vis_dir, f'{filename_prefix}_weights.png'), dpi=300)
            else:
                plt.show()
        else:
            print("No ensemble weights history available.")
        
        plt.close()
        
        plt.figure(figsize=(12, 6))
        if 'kalman_gain' in self.metrics and len(self.metrics['kalman_gain']) > 0:
            gain_array = np.array(self.metrics['kalman_gain'])
            plt.plot(gain_array, 'r-', linewidth=2)
            
            plt.title('Kalman Gain Magnitude', fontsize=14)
            plt.xlabel('Updates', fontsize=12)
            plt.ylabel('Gain Magnitude', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if filename_prefix:
                plt.savefig(os.path.join(vis_dir, f'{filename_prefix}_gain.png'), dpi=300)
            else:
                plt.show()
        else:
            print("No Kalman gain history available.")
        
        plt.close()
        
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        if 'adaptive_q_scale' in self.metrics and len(self.metrics['adaptive_q_scale']) > 0:
            plt.plot(self.metrics['adaptive_q_scale'], 'b-', linewidth=2)
            plt.title('Adaptive Q Scale (Process Noise)', fontsize=12)
            plt.xlabel('Updates', fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        if 'adaptive_r_scale' in self.metrics and len(self.metrics['adaptive_r_scale']) > 0:
            plt.plot(self.metrics['adaptive_r_scale'], 'g-', linewidth=2)
            plt.title('Adaptive R Scale (Measurement Noise)', fontsize=12)
            plt.xlabel('Updates', fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        if 'learning_rate' in self.metrics and len(self.metrics['learning_rate']) > 0:
            plt.plot(self.metrics['learning_rate'], 'c-', linewidth=2)
            plt.title('Kalman Learning Rate', fontsize=12)
            plt.xlabel('Updates', fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        if 'adaptive_coupling' in self.metrics and len(self.metrics['adaptive_coupling']) > 0:
            plt.plot(self.metrics['adaptive_coupling'], 'm-', linewidth=2)
            plt.title('Adaptive Coupling Factor', fontsize=12)
            plt.xlabel('Updates', fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename_prefix:
            plt.savefig(os.path.join(vis_dir, f'{filename_prefix}_adaptive_params.png'), dpi=300)
        else:
            plt.show()
        
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        if 'member_performance' in self.metrics and all(self.metrics['member_performance']):
            for i in range(self.n_members):
                if self.metrics['member_performance'][i]:
                    plt.plot(self.metrics['member_performance'][i], label=f'Member {i+1}')
            
            plt.title('Individual Member Performance', fontsize=14)
            plt.xlabel('Updates', fontsize=12)
            plt.ylabel('Performance Score', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            if filename_prefix:
                plt.savefig(os.path.join(vis_dir, f'{filename_prefix}_member_performance.png'), dpi=300)
            else:
                plt.show()
        else:
            print("No member performance history available.")
        
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        updates = range(len(self.metrics.get('ensemble_weights', [])))
        if updates:
            current_weights = self.ensemble_weights.detach().cpu().numpy()
            
            state_estimates = np.array(self.metrics.get('ensemble_weights', []))
            if 'measurements' not in self.metrics and hasattr(self, 'global_filter'):
                latest_measurement = self.global_filter['z']
                plt.axhline(y=latest_measurement[0], color='r', linestyle='--', 
                        label=f'Latest Measurement (Member 1)')
            
            if len(state_estimates) > 0:
                plt.plot(updates, state_estimates[:, 0], 'b-', linewidth=2, 
                    label='State Estimate (Member 1)')
                
                plt.title('Kalman Filter State vs Measurements', fontsize=14)
                plt.xlabel('Updates', fontsize=12)
                plt.ylabel('Value', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)
                
                if filename_prefix:
                    plt.savefig(os.path.join(vis_dir, f'{filename_prefix}_state_vs_measurement.png'), dpi=300)
                else:
                    plt.show()
        else:
            print("No updates recorded yet.")
        
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        if hasattr(self, 'global_filter') and self.global_filter.get('P_history'):
            p_det = [np.linalg.det(p) for p in self.global_filter['P_history']]
            plt.semilogy(p_det, 'k-', linewidth=2)
            plt.title('Kalman Filter Uncertainty (det(P))', fontsize=14)
            plt.xlabel('Updates', fontsize=12)
            plt.ylabel('Determinant of P (log scale)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if filename_prefix:
                plt.savefig(os.path.join(vis_dir, f'{filename_prefix}_uncertainty.png'), dpi=300)
            else:
                plt.show()
        else:
            if hasattr(self, 'global_filter'):
                det_p = np.linalg.det(self.global_filter['P'])
                print(f"Current Kalman filter uncertainty (det(P)): {det_p:.6e}")
        
        plt.close()

    @torch.no_grad()
    def evaluate_on_dataset(self, data_loader, num_samples=100, target_class=None):
        self.eval()
        
        results = {
            'success_rate': 0.0,
            'perturbation_norm': 0.0,
            'class_success_rates': {i: {'count': 0, 'success': 0} for i in range(10)},
            'prediction_changes': 0,
            'member_success_rates': [0.0] * self.n_members,
            'member_samples': 0,
            'errors': 0
        }
        
        samples_dir = os.path.join(self.results_dir, 'evaluation_samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        processed_samples = 0
        



        if target_class is not None and isinstance(target_class, str):
            try:
                target_class = int(target_class)
                print(f"Evaluating with targeted attack to class {target_class}")
            except ValueError:
                print(f"Error: target_class '{target_class}' is not a valid number. Using non-targeted attack.")
                target_class = None
        
        if target_class is not None:
            print(f"Using targeted attack with target class: {target_class}")
        else:
            print("Using non-targeted attack (any class change)")
        
        try:
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
                    try:
                        images, labels = images.to(self.device), labels.to(self.device)
                        batch_size = images.shape[0]
                        
                        original_outputs = self.target_model(images)
                        orig_preds = torch.argmax(original_outputs, dim=1)
                        
                        adv_images, perturbation, member_outputs, _ = self.forward_eval(images, target_class)
                        
                        adversarial_outputs = self.target_model(adv_images)
                        adv_preds = torch.argmax(adversarial_outputs, dim=1)
                        
                        member_success_counts = [0] * self.n_members
                        for i, member_output in enumerate(member_outputs):
                            member_pred = self.target_model(member_output)
                            member_preds = torch.argmax(member_pred, dim=1)
                            
                            if target_class is None:
                                member_success = (member_preds != orig_preds).sum().item()
                            else:
                                member_success = (member_preds == target_class).sum().item()
                                
                            member_success_counts[i] += member_success
                        
                        if target_class is None:
                            success_count = (adv_preds != orig_preds).sum().item()
                        else:
                            success_count = (adv_preds == target_class).sum().item()

                        perturbation_norm = torch.norm(perturbation, p=2, dim=(1, 2, 3)).mean().item()
                        
                        results['prediction_changes'] += success_count
                        results['perturbation_norm'] += perturbation_norm * batch_size
                        
                        for i in range(batch_size):
                            label = labels[i].item()
                            results['class_success_rates'][label]['count'] += 1
                            
                            if target_class is None:
                                if adv_preds[i] != orig_preds[i]:
                                    results['class_success_rates'][label]['success'] += 1
                            else:
                                if adv_preds[i] == target_class:
                                    results['class_success_rates'][label]['success'] += 1
                        
                        for i in range(self.n_members):
                            results['member_success_rates'][i] += member_success_counts[i]
                        results['member_samples'] += batch_size
                        
                        if batch_idx < 5:
                            for i in range(min(5, batch_size)):
                                img_idx = batch_idx * batch_size + i
                                if img_idx < num_samples:
                                    vis_path = os.path.join(samples_dir, f'sample_{img_idx}_label_{labels[i].item()}.png')
                                    self.visualize_ensemble_attack(images[i], labels[i].item(), vis_path, target_class)

                        
                        processed_samples += batch_size
                        
                        del images, labels, original_outputs, adv_images, perturbation
                        del member_outputs, adversarial_outputs, orig_preds, adv_preds
                        
                        if processed_samples >= num_samples:
                            break
                            
                    except Exception as e:
                        print(f"Error in evaluation batch {batch_idx}: {e}")
                        results['errors'] += 1
                        continue
                    
                    if batch_idx % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
            if processed_samples > 0:
                results['success_rate'] = results['prediction_changes'] / processed_samples * 100.0
                results['perturbation_norm'] /= processed_samples
            
                for class_idx, stats in results['class_success_rates'].items():
                    if stats['count'] > 0:
                        stats['success_rate'] = stats['success'] / stats['count'] * 100.0
                    else:
                        stats['success_rate'] = 0.0
            
                for i in range(self.n_members):
                    if processed_samples > 0:
                        results['member_success_rates'][i] = results['member_success_rates'][i] / processed_samples * 100.0
            
            attack_type = "targeted to class " + str(target_class) if target_class is not None else "non-targeted"
            print(f"\nEvaluation results on {processed_samples} samples ({attack_type}):")
            print(f"Overall success rate: {results['success_rate']:.2f}%")
            print(f"Perturbation L2 norm: {results['perturbation_norm']:.4f}")
            print("\nClass-specific success rates:")
            for class_idx, stats in results['class_success_rates'].items():
                if stats['count'] > 0:
                    print(f"  Class {class_idx}: {stats['success_rate']:.2f}% ({stats['success']}/{stats['count']})")
            
            print("\nIndividual member success rates:")
            for i in range(self.n_members):
                print(f"  Member {i+1}: {results['member_success_rates'][i]:.2f}%")
            
            results['attack_type'] = 'targeted' if target_class is not None else 'non-targeted'
            if target_class is not None:
                results['target_class'] = target_class
            
            results_path = os.path.join(self.results_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                clean_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        clean_results[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in value.items()}
                    elif isinstance(value, np.ndarray):
                        clean_results[key] = value.tolist()
                    else:
                        clean_results[key] = value
                
                json.dump(clean_results, f, indent=2)
                
            print(f"Evaluation results saved to {results_path}")
            
            return results
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            traceback.print_exc()
            return results

def create_quantum_ensemble(target_model, device="cuda" if torch.cuda.is_available() else "cpu"):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    if device == "cuda":
        torch.cuda.empty_cache()
    
    ensemble = QuantumEnsembleManager(
        target_model=target_model,
        device=device,
        epsilon=0.1
    )
    
    print(f"Quantum ensemble initialized on device: {device}")
    print(f"Number of ensemble members: 5")
    print(f"Target model: {type(target_model).__name__}")
    
    return ensemble

