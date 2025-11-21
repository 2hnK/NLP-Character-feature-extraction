import torch
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict

class PKSampler(Sampler):
    """
    Sampler that ensures each batch contains P classes and K samples per class.
    Total batch size = P * K.
    """
    def __init__(self, data_source, p, k, shuffle=True):
        """
        Args:
            data_source (Dataset): Dataset to sample from. Must implement get_labels().
            p (int): Number of classes per batch.
            k (int): Number of samples per class.
            shuffle (bool): Whether to shuffle the data.
        """
        self.data_source = data_source
        self.p = p
        self.k = k
        self.shuffle = shuffle
        self.batch_size = p * k
        
        if not hasattr(data_source, 'get_labels'):
            raise ValueError("DataSource must implement get_labels() method")
            
        self.labels = data_source.get_labels()
        self.label_to_indices = defaultdict(list)
        
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        self.classes = list(self.label_to_indices.keys())
        
        # Calculate length
        # We want to use as much data as possible, but we are constrained by P*K structure.
        # A simple approximation is total_samples // batch_size
        self.num_samples = len(self.labels)
        self.num_batches = self.num_samples // self.batch_size
        
    def __iter__(self):
        # 1. Select P classes for each batch
        # If we have many classes, we want to iterate through them.
        # If we have few classes (e.g. 8), and P=8, we use all classes every time.
        
        # Strategy:
        # Create a list of classes to sample from.
        # Ideally we want to exhaust all samples.
        
        # For simplicity and robustness in Triplet Mining:
        # We randomly select P classes, then randomly select K samples from each.
        # We do this num_batches times.
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Select P classes
            if len(self.classes) < self.p:
                 # If we have fewer classes than P, allow replacement or reduce P (but P is fixed)
                 # Here we sample with replacement if needed
                 selected_classes = np.random.choice(self.classes, self.p, replace=True)
            else:
                selected_classes = np.random.choice(self.classes, self.p, replace=False)
            
            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                
                # Select K samples from this class
                if len(indices) < self.k:
                    # If not enough samples, sample with replacement
                    selected_indices = np.random.choice(indices, self.k, replace=True)
                else:
                    selected_indices = np.random.choice(indices, self.k, replace=False)
                    
                batch_indices.extend(selected_indices)
            
            if self.shuffle:
                np.random.shuffle(batch_indices)
                
            yield from batch_indices

    def __len__(self):
        return self.num_batches * self.batch_size
