import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from collections import Counter

class DatasetAnalyzer:   
    def __init__(self, dataset_loader, train_loader, val_loader, test_loader, data_res):
        self.dataset_loader = dataset_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_res = data_res
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2023, 0.1994, 0.2016])
        try:
            self.class_names = self.dataset_loader.get_class_names()
        except:
            self.class_names = [str(i) for i in range(10)]  

    def analyze_all(self, figsize_multiplier=1.0):
        print("\n======= CIFAR-10 Dataset Analysis =======")
        
        plt.rcParams['figure.figsize'] = [
            plt.rcParams['figure.figsize'][0] * figsize_multiplier,
            plt.rcParams['figure.figsize'][1] * figsize_multiplier]
        self.plot_class_distribution()
        self.analyze_tensor_shapes()
        self.analyze_pixel_values()
        self.show_images_by_class()
        self.plot_dataset_sizes()

    def plot_class_distribution(self):
        print("\nClass Distribution:")
        
        try:
            train_dist = self.dataset_loader.get_class_distribution(self.train_loader)
            val_dist = self.dataset_loader.get_class_distribution(self.val_loader)
            test_dist = self.dataset_loader.get_class_distribution(self.test_loader)
            class_names = list(train_dist.keys())
            train_counts = [train_dist[name] for name in class_names]
            val_counts = [val_dist[name] for name in class_names]
            test_counts = [test_dist[name] for name in class_names]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(class_names))
            width = 0.25
            ax.bar(x - width, train_counts, width, label='Train')
            ax.bar(x, val_counts, width, label='Validation')
            ax.bar(x + width, test_counts, width, label='Test')
            ax.set_title('Class Distribution across Train/Val/Test Sets')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Number of Samples')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_res, 'Class_Distribution_across1.png'))            
            plt.show()
            plt.figure(figsize=(10, 6))
            plt.pie(train_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
            plt.title('Training Set Class Distribution')
            plt.axis('equal')
            plt.savefig(os.path.join(self.data_res, 'Training_Set_Class_Distribution.png'))            
            plt.show()

        except Exception as e:
            print(f"Could not analyze class distribution: {e}")
            print("Trying alternative approach...")
            self._analyze_class_distribution_manually()
    
    def _analyze_class_distribution_manually(self):
        train_labels = []
        val_labels = []
        test_labels = []
        
        for images, labels in self.train_loader:
            train_labels.extend(labels.numpy())
            if len(train_labels) > 1000:  
                break
                
        for images, labels in self.val_loader:
            val_labels.extend(labels.numpy())
            if len(val_labels) > 1000:
                break
                
        for images, labels in self.test_loader:
            test_labels.extend(labels.numpy())
            if len(test_labels) > 1000:
                break
        
        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)
        test_counts = Counter(test_labels)
        
        class_indices = sorted(list(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_indices))
        width = 0.25
        
        train_values = [train_counts.get(idx, 0) for idx in class_indices]
        val_values = [val_counts.get(idx, 0) for idx in class_indices]
        test_values = [test_counts.get(idx, 0) for idx in class_indices]
        
        ax.bar(x - width, train_values, width, label='Train')
        ax.bar(x, val_values, width, label='Validation')
        ax.bar(x + width, test_values, width, label='Test')
        
        ax.set_title('Class Distribution across Train/Val/Test Sets (Sample)')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Number of Samples')
        ax.set_xticks(x)
        ax.set_xticklabels([self.class_names[i] if i < len(self.class_names) else f"Class {i}" for i in class_indices])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_res, 'Class_Distribution.png'))            
        plt.show()
    
    def analyze_tensor_shapes(self):
        print("\nTensor Shape Analysis:")
        
        try:
            images, labels = next(iter(self.train_loader))
            
            print(f"Batch shape: {images.shape}")
            print(f"Single image shape: {images[0].shape}")
            print(f"Labels shape: {labels.shape}")
            
            img = images[0]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
            
            for i, (ax, name) in enumerate(zip(axes, channel_names)):
                channel_data = img[i].numpy()
                im = ax.imshow(channel_data, cmap='viridis')
                ax.set_title(f"{name}\nShape: {channel_data.shape}")
                fig.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_res, 'Tensor_Shape_Analysis.png'))            
            plt.show()
            
            mean_per_channel = images.mean(dim=[0, 2, 3])
            std_per_channel = images.std(dim=[0, 2, 3])
            
            print("\nChannel Statistics (Current Batch):")
            for i, (mean, std) in enumerate(zip(mean_per_channel, std_per_channel)):
                print(f"Channel {i}: Mean = {mean:.4f}, Std = {std:.4f}")
        
        except Exception as e:
            print(f"Could not analyze tensor shapes: {e}")
    
    def analyze_pixel_values(self):
        print("\nPixel Value Distribution:")
        
        try:
            images, _ = next(iter(self.train_loader))
            
            if images.min() < 0 or images.max() > 1.5:
                unnormalized = images * torch.tensor(self.std)[None, :, None, None] + torch.tensor(self.mean)[None, :, None, None]
            else:
                unnormalized = images
            
            r_pixels = unnormalized[:, 0, :, :].flatten().numpy()
            g_pixels = unnormalized[:, 1, :, :].flatten().numpy()
            b_pixels = unnormalized[:, 2, :, :].flatten().numpy()
            
            plt.figure(figsize=(12, 6))
            
            plt.hist(r_pixels, bins=50, alpha=0.5, color='red', label='Red Channel')
            plt.hist(g_pixels, bins=50, alpha=0.5, color='green', label='Green Channel')
            plt.hist(b_pixels, bins=50, alpha=0.5, color='blue', label='Blue Channel')
            
            plt.title('Distribution of Pixel Values by Channel')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.data_res, 'Distribution_of_Pixel_Values_by_Channel.png'))            
            plt.show()
        
        except Exception as e:
            print(f"Could not analyze pixel values: {e}")
    
    def show_images_by_class(self):
        print("\nSample Images by Class:")
        
        try:
            class_samples = {i: [] for i in range(len(self.class_names))}
            
            batch_iter = iter(self.train_loader)
            
            max_attempts = 50
            attempts = 0
            
            while min(len(samples) for samples in class_samples.values()) < 5 and attempts < max_attempts:
                try:
                    images, labels = next(batch_iter)
                    attempts += 1
                    
                    for i, label in enumerate(labels):
                        label_idx = label.item()
                        if len(class_samples[label_idx]) < 5:
                            img = images[i].permute(1, 2, 0).numpy()
                            if img.min() < 0 or img.max() > 1.5:
                                img = self.std * img + self.mean
                            img = np.clip(img, 0, 1)
                            
                            class_samples[label_idx].append(img)
                
                except StopIteration:
                    break
            
            if all(len(samples) > 0 for samples in class_samples.values()):
                fig, axes = plt.subplots(len(class_samples), min(5, max(len(samples) for samples in class_samples.values())), 
                                        figsize=(15, 2*len(class_samples)))
                
                for i, class_idx in enumerate(sorted(class_samples.keys())):
                    samples = class_samples[class_idx]
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
                    
                    for j, img in enumerate(samples[:5]):
                        if len(class_samples) == 1:
                            ax = axes[j]
                        elif len(samples) == 1:
                            ax = axes[i]
                        else:
                            ax = axes[i, j]
                            
                        ax.imshow(img)
                        if j == 0:
                            ax.set_ylabel(class_name, fontsize=12)
                        ax.axis('off')
                
                plt.tight_layout()
                plt.suptitle('Sample Images by Class', fontsize=16, y=1.0)
                plt.savefig(os.path.join(self.data_res, 'Sample_Images_by_Class.png'))
                plt.show()
            else:
                print("Could not collect enough samples for each class")
        
        except Exception as e:
            print(f"Could not show images by class: {e}")
    
    def plot_dataset_sizes(self):
        print("\nDataset Size Comparison:")
        
        try:
            batch_size = next(iter(self.train_loader))[0].shape[0]
            
            train_size = len(self.train_loader) * batch_size
            val_size = len(self.val_loader) * batch_size
            test_size = len(self.test_loader) * batch_size
            total_size = train_size + val_size + test_size
            
            sizes = [train_size, val_size, test_size]
            labels = ['Training', 'Validation', 'Testing']
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            plt.figure(figsize=(10, 6))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Dataset Split Distribution')
            plt.axis('equal')
            plt.savefig(os.path.join(self.data_res, 'Dataset_Split_Distribution.png'))
            plt.show()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(labels, sizes, color=colors)
            ax.set_ylabel('Number of Samples')
            ax.set_title('Dataset Size Comparison')
            
            for i, v in enumerate(sizes):
                ax.text(i, v + 0.02 * total_size, f"{v} ({v/total_size:.1%})", 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.show()
        
        except Exception as e:
            print(f"Could not plot dataset sizes: {e}")


def analyze_cifar10_dataset(dataset_loader, train_loader, val_loader, test_loader, data_res, figsize_multiplier=1.0):
    analyzer = DatasetAnalyzer(dataset_loader, train_loader, val_loader, test_loader, data_res)
    analyzer.analyze_all(figsize_multiplier)
    return analyzer
