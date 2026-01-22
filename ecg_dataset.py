"""
ECG Dataset Loader for MIT-BIH Arrhythmia Database
Loads data from PhysioNet MIT-BIH database stored locally
"""

import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

try:
    import wfdb
except ImportError:
    wfdb = None


class MITBIHDataset(Dataset):
    """
    Simple MIT-BIH Dataset Loader - loads CSV data as-is
    
    Parameters:
    -----------
    root_dir : str
        Path to the directory (not used if CSV files are in current directory)
    train : bool
        If True, load training data; if False, load test data
    window_size : int
        Size of the ECG window (not used, just for compatibility)
    transform : callable, optional
        Optional transform to be applied on a sample
    oversample_minority : bool
        Not used, just for compatibility
    """
    
    def __init__(self, root_dir, train=True, window_size=128, transform=None, oversample_minority=False):
        self.root_dir = root_dir
        self.train = train
        self.window_size = window_size
        self.transform = transform
        
        # Class names according to AAMI EC57 standard
        self.class_names = [
            'Normal (N)',
            'Supraventricular (S)', 
            'Ventricular (V)',
            'Fusion (F)',
            'Unknown (Q)'
        ]
        
        # Load the data
        self.data, self.labels = self._load_data()
    
    def _load_data(self):
        """
        Load data from PhysioNet MIT-BIH database
        """
        if wfdb is None:
            raise ImportError(
                "wfdb library is required to load PhysioNet data.\n"
                "Install it with: pip install wfdb"
            )
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"PhysioNet directory not found: {self.root_dir}\n"
                f"Please download the MIT-BIH database from:\n"
                f"https://physionet.org/content/mitdb/1.0.0/\n"
                f"Or use: wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/"
            )
        
        # AAMI EC57 classification mapping
        aami_classes = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
            'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular
            'V': 2, 'E': 2,                           # Ventricular
            'F': 3,                                   # Fusion
            '/': 4, 'f': 4, 'Q': 4,                  # Unknown
        }
        
        # Train/test split (standard in literature)
        if self.train:
            records = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                      122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
        else:
            records = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                      210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
        
        all_data = []
        all_labels = []
        
        print(f"Loading data from {self.root_dir}...")
        for record in records:
            try:
                record_path = os.path.join(self.root_dir, str(record))
                
                # Read signal and annotation
                signals, fields = wfdb.rdsamp(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                
                # Use MLII lead (channel 0)
                ecg_signal = signals[:, 0]
                
                # Extract windows around each beat
                for position, symbol in zip(annotation.sample, annotation.symbol):
                    if symbol not in aami_classes:
                        continue
                    
                    # Extract window centered on beat
                    start = position - self.window_size // 2
                    end = start + self.window_size
                    
                    if start < 0 or end > len(ecg_signal):
                        continue
                    
                    window = ecg_signal[start:end]
                    
                    if len(window) == self.window_size:
                        all_data.append(window)
                        all_labels.append(aami_classes[symbol])
                
                print(f"  Processed record {record}: {len(all_data)} total samples")
                
            except Exception as e:
                print(f"  Warning: Could not load record {record}: {e}")
                continue
        
        if len(all_data) == 0:
            raise ValueError(
                f"No data was loaded from {self.root_dir}.\n"
                f"Make sure the directory contains valid MIT-BIH .dat and .atr files."
            )
        
        data = np.array(all_data)
        labels = np.array(all_labels)
        
        print(f"\nTotal samples loaded: {len(data)}")
        print(f"Data shape: {data.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing MITBIHDataset...")
    print("=" * 60)
    
    try:
        dataset = MITBIHDataset(
            root_dir='.',
            train=True
        )
        
        print(f"\nDataset successfully loaded!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Data shape: {dataset.data.shape}")
        print(f"Labels shape: {dataset.labels.shape}")
        print(f"\nClass distribution:")
        for i, name in enumerate(dataset.class_names):
            count = np.sum(dataset.labels == i)
            percentage = 100 * count / len(dataset.labels)
            print(f"  {name}: {count} ({percentage:.2f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
