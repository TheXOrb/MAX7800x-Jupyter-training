"""
ECG Dataset Loader for MIT-BIH Arrhythmia Database + PTB-XL
Loads data from PhysioNet MIT-BIH and PTB-XL databases stored locally
"""

import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from scipy.signal import resample

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


class PTBXLDataset(Dataset):
    """
    PTB-XL Dataset Loader - loads from PhysioNet PTB-XL database
    Automatically resamples from 100 Hz to 360 Hz for consistency with MIT-BIH
    
    Parameters:
    -----------
    root_dir : str
        Path to the ptb-xl records directory (e.g., './physionet.org/files/ptb-xl/1.0.1/records100')
    train : bool
        If True, use training split; if False, use test split
    window_size : int
        Size of the ECG window (default 128 samples at 360 Hz)
    transform : callable, optional
        Optional transform to be applied on a sample
    oversample_minority : bool
        Not used, for compatibility
    target_sr : int
        Target sample rate to resample to (default 360 Hz to match MIT-BIH)
    """
    
    def __init__(self, root_dir, train=True, window_size=128, transform=None, 
                 oversample_minority=False, target_sr=360):
        self.root_dir = root_dir
        self.train = train
        self.window_size = window_size
        self.transform = transform
        self.target_sr = target_sr
        self.source_sr = 100  # PTB-XL native sample rate
        
        # Class names - using simplified mapping to AAMI classes
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
        Load data from PTB-XL database and resample to 360 Hz
        """
        if wfdb is None:
            raise ImportError(
                "wfdb library is required to load PhysioNet data.\n"
                "Install it with: pip install wfdb"
            )
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"PTB-XL directory not found: {self.root_dir}\n"
                f"Please download the PTB-XL database from:\n"
                f"https://physionet.org/content/ptb-xl/1.0.1/"
            )
        
        # Load the metadata CSV
        metadata_path = os.path.join(self.root_dir, '..', 'ptbxl_database.csv')
        if not os.path.exists(metadata_path):
            # Alternative path
            metadata_path = os.path.join(self.root_dir.replace('records100', ''), 'ptbxl_database.csv')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"PTB-XL metadata file not found: {metadata_path}\n"
                f"Expected ptbxl_database.csv in PTB-XL root directory"
            )
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Split into train/test using 10-fold split
        if self.train:
            df_split = df[df['strat_fold'] != 10]  # Folds 1-9 for training
        else:
            df_split = df[df['strat_fold'] == 10]  # Fold 10 for testing
        
        all_data = []
        all_labels = []
        error_count = 0
        success_count = 0
        
        print(f"Loading {'training' if self.train else 'test'} data from {self.root_dir}...")
        print(f"Found {len(df_split)} records in metadata")
        
        # Simplified label mapping to AAMI classes
        # Maps SCP-ECG codes to AAMI EC57 standard classes
        label_map = {
            # Normal and structural abnormalities -> Normal (N)
            'NORM': 0,  # Normal
            'MI': 0,    # Myocardial infarction
            'STTC': 0,  # ST/T changes
            'CD': 0,    # Conduction disturbance
            'HYP': 0,   # Hypertrophy
            'LVOLT': 0, # Low voltage
            'SR': 0,    # Sinus rhythm
            'CLBBB': 0, # Complete left bundle branch block
            'CRBBB': 0, # Complete right bundle branch block
            'ILBBB': 0, # Incomplete left bundle branch block
            'IRBBB': 0, # Incomplete right bundle branch block
            
            # Supraventricular arrhythmias (S)
            'PAC': 1,   # Premature atrial contraction
            'AFIB': 1,  # Atrial fibrillation
            'AFLUT': 1, # Atrial flutter
            'SVPB': 1,  # Supraventricular premature beat
            'SINUS': 1, # Sinus arrhythmia
            'JUNCTIONAL': 1, # Junctional arrhythmia
            
            # Ventricular arrhythmias (V)
            'PVC': 2,   # Premature ventricular contraction
            'VT': 2,    # Ventricular tachycardia
            'VF': 2,    # Ventricular fibrillation
            'VENT': 2,  # Ventricular beat
            
            # Fusion beats (F)
            'FUSION': 3, # Fusion beat
            
            # Unknown/Other (Q)
            'UNKNOWN': 4,
        }
        
        def parse_scp_codes(scp_codes_str):
            """
            Parse SCP codes dictionary string and return the code with highest likelihood
            Example: "{'NORM': 100.0, 'LVOLT': 0.0}" -> 'NORM'
            """
            try:
                if pd.isna(scp_codes_str):
                    return 'NORM'
                
                # Convert string representation to dictionary
                import ast
                scp_dict = ast.literal_eval(str(scp_codes_str))
                
                if not scp_dict:
                    return 'NORM'
                
                # Find the code with highest likelihood
                max_code = max(scp_dict.items(), key=lambda x: x[1])[0]
                return max_code
                
            except Exception as e:
                print(f"Warning: Could not parse scp_codes '{scp_codes_str}': {e}")
                return 'NORM'
        
        # Print first few filenames for debugging
        if len(df_split) > 0:
            first_filename = df_split.iloc[0]['filename_lr']
            print(f"First filename from CSV: {first_filename}")
            if first_filename.startswith('records100/'):
                stripped = first_filename.replace('records100/', '', 1)
                print(f"After stripping prefix: {stripped}")
                test_path = os.path.join(self.root_dir, stripped)
                print(f"Full path will be: {test_path}")
                print(f"Path exists: {os.path.exists(test_path + '.dat')}")
        
        for idx, row in df_split.iterrows():
            try:
                filename = row['filename_lr']
                
                # Strip 'records100/' prefix if present since root_dir already points to records100
                if filename.startswith('records100/'):
                    filename = filename.replace('records100/', '', 1)
                
                record_path = os.path.join(self.root_dir, filename)
                
                # Read signal (100 Hz version)
                signals, fields = wfdb.rdsamp(record_path)
                
                # Use first lead (Lead I) if available
                ecg_signal = signals[:, 0].astype(np.float32)
                
                # Get diagnostic codes from scp_codes column
                scp_codes_str = row['scp_codes']
                diag_code = parse_scp_codes(scp_codes_str)
                
                # Map to AAMI class
                if diag_code in label_map:
                    aami_label = label_map[diag_code]
                else:
                    # Default to Normal if unknown code
                    aami_label = 0
                
                # Resample from 100 Hz to 360 Hz
                num_samples_resampled = int(len(ecg_signal) * self.target_sr / self.source_sr)
                ecg_resampled = resample(ecg_signal, num_samples_resampled)
                
                # Extract multiple windows from the signal
                # Use striding to get multiple samples per record
                stride = self.window_size // 2  # 50% overlap
                
                for start in range(0, len(ecg_resampled) - self.window_size, stride):
                    end = start + self.window_size
                    window = ecg_resampled[start:end]
                    
                    if len(window) == self.window_size:
                        all_data.append(window)
                        all_labels.append(aami_label)
                
                success_count += 1
                
                # Reduce print frequency to avoid IOPub data rate limit
                if len(df_split) > 1 and idx % max(1, len(df_split) // 5) == 0:
                    print(f"  Processed {idx}/{len(df_split)} records: {len(all_data)} total samples")
                    
            except Exception as e:
                error_count += 1
                # Only print first 5 errors to avoid IOPub limit
                if error_count <= 5:
                    print(f"  Warning: Could not load record {row['filename_lr']}: {e}")
                continue
        
        print(f"\nLoad summary: {success_count} successful, {error_count} failed")
        
        if len(all_data) == 0:
            raise ValueError(
                f"No data was loaded from {self.root_dir}.\n"
                f"Make sure the directory contains valid PTB-XL records."
            )
        
        data = np.array(all_data, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)
        
        print(f"\nPTB-XL samples loaded: {len(data)}")
        print(f"Data shape: {data.shape}")
        print(f"Resampled from {self.source_sr} Hz to {self.target_sr} Hz")
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


class CombinedECGDataset(Dataset):
    """
    Combines multiple ECG datasets (MIT-BIH, PTB-XL, etc.) into a single dataset
    
    Parameters:
    -----------
    datasets : list of Dataset objects
        List of datasets to combine (all should have compatible data/labels)
    """
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.class_names = datasets[0].class_names if datasets else []
        
        # Combine data and labels from all datasets
        all_data = []
        all_labels = []
        
        for dataset in datasets:
            all_data.append(dataset.data)
            all_labels.append(dataset.labels)
        
        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        print(f"\n{'='*60}")
        print(f"Combined Dataset Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data):,}")
        print(f"Data shape: {self.data.shape}")
        print(f"\nCombined class distribution:")
        for i, name in enumerate(self.class_names):
            count = np.sum(self.labels == i)
            percentage = 100 * count / len(self.labels)
            print(f"  {name}: {count:6,} ({percentage:5.2f}%)")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        return sample, label


if __name__ == "__main__":
    # Test the dataset loaders
    print("Testing ECG Dataset Loaders...")
    print("=" * 60)
    
    # Test MIT-BIH
    try:
        print("\n1. Testing MITBIHDataset...")
        mitbih_dataset = MITBIHDataset(
            root_dir='./physionet.org/files/mitdb/1.0.0',
            train=True
        )
        
        print(f"✓ MIT-BIH Dataset successfully loaded!")
        print(f"  Number of samples: {len(mitbih_dataset):,}")
        print(f"  Data shape: {mitbih_dataset.data.shape}")
        
    except Exception as e:
        print(f"✗ MIT-BIH Error: {e}")
        mitbih_dataset = None
    
    # Test PTB-XL
    try:
        print("\n2. Testing PTBXLDataset...")
        ptbxl_dataset = PTBXLDataset(
            root_dir='./physionet.org/files/ptb-xl/1.0.1/records100',
            train=True,
            target_sr=360
        )
        
        print(f"✓ PTB-XL Dataset successfully loaded!")
        print(f"  Number of samples: {len(ptbxl_dataset):,}")
        print(f"  Data shape: {ptbxl_dataset.data.shape}")
        
    except Exception as e:
        print(f"✗ PTB-XL Error: {e}")
        ptbxl_dataset = None
    
    # Test combined dataset
    if mitbih_dataset is not None and ptbxl_dataset is not None:
        try:
            print("\n3. Testing CombinedECGDataset...")
            combined_dataset = CombinedECGDataset([mitbih_dataset, ptbxl_dataset])
            print(f"✓ Combined Dataset successfully created!")
            
        except Exception as e:
            print(f"✗ Combined Dataset Error: {e}")
