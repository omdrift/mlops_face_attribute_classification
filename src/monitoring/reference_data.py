"""
Reference data management for drift detection
"""
import os
import pickle
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


class ReferenceDataManager:
    """
    Manages reference data for drift detection
    """
    
    def __init__(self, storage_path: str = 'data/reference'):
        """
        Initialize reference data manager
        
        Args:
            storage_path: Directory to store reference data
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_reference_data(
        self,
        data: pd.DataFrame,
        name: str = 'default',
        metadata: Optional[dict] = None
    ) -> str:
        """
        Save reference dataset
        
        Args:
            data: Reference DataFrame
            name: Reference dataset name
            metadata: Optional metadata dictionary
        
        Returns:
            Path to saved reference data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reference_{name}_{timestamp}.pkl"
        filepath = os.path.join(self.storage_path, filename)
        
        # Prepare data bundle
        bundle = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns)
        }
        
        # Save to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
        
        print(f"✓ Reference data saved to {filepath}")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        
        return filepath
    
    def load_reference_data(
        self,
        name: str = 'default',
        latest: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load reference dataset
        
        Args:
            name: Reference dataset name
            latest: Load the latest version
        
        Returns:
            Reference DataFrame or None
        """
        # Find matching files
        import glob
        pattern = os.path.join(self.storage_path, f"reference_{name}_*.pkl")
        files = sorted(glob.glob(pattern), reverse=True)
        
        if not files:
            print(f"⚠ No reference data found for '{name}'")
            return None
        
        filepath = files[0] if latest else files[-1]
        
        # Load data
        with open(filepath, 'rb') as f:
            bundle = pickle.load(f)
        
        data = bundle['data']
        metadata = bundle.get('metadata', {})
        
        print(f"✓ Loaded reference data from {filepath}")
        print(f"  Timestamp: {bundle['timestamp']}")
        print(f"  Shape: {bundle['shape']}")
        
        return data
    
    def list_references(self) -> list:
        """
        List all available reference datasets
        
        Returns:
            List of reference dataset info
        """
        import glob
        pattern = os.path.join(self.storage_path, "reference_*.pkl")
        files = glob.glob(pattern)
        
        references = []
        for filepath in sorted(files, reverse=True):
            with open(filepath, 'rb') as f:
                bundle = pickle.load(f)
            
            info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'timestamp': bundle.get('timestamp'),
                'shape': bundle.get('shape'),
                'columns': bundle.get('columns'),
            }
            references.append(info)
        
        return references
    
    def create_reference_from_training_data(
        self,
        training_data_path: str = 'data/processed/train_data_s1.pt',
        name: str = 'training',
        sample_size: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Create reference data from training dataset
        
        Args:
            training_data_path: Path to training data
            name: Name for reference dataset
            sample_size: Number of samples to use (None = all)
        
        Returns:
            Reference DataFrame
        """
        import torch
        
        if not os.path.exists(training_data_path):
            print(f"⚠ Training data not found at {training_data_path}")
            return None
        
        # Load training data
        data = torch.load(training_data_path)
        X = data['X'].numpy()
        y = data['y'].numpy()
        
        # Create DataFrame
        df = pd.DataFrame(
            y,
            columns=['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
        )
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Save as reference
        metadata = {
            'source': training_data_path,
            'original_size': len(y),
            'sampled': sample_size is not None,
        }
        
        self.save_reference_data(df, name=name, metadata=metadata)
        
        return df
    
    def update_reference_with_production_data(
        self,
        production_data: pd.DataFrame,
        name: str = 'production',
        window_size: int = 10000
    ) -> pd.DataFrame:
        """
        Create or update reference with recent production data
        
        Args:
            production_data: Recent production data
            name: Reference name
            window_size: Size of sliding window
        
        Returns:
            Updated reference DataFrame
        """
        # Load existing reference if available
        existing = self.load_reference_data(name)
        
        if existing is not None:
            # Concatenate and keep recent window
            combined = pd.concat([existing, production_data], ignore_index=True)
            reference = combined.tail(window_size)
        else:
            # Use production data as reference
            reference = production_data.tail(window_size)
        
        # Save updated reference
        metadata = {
            'updated_from_production': True,
            'window_size': window_size,
            'production_samples_added': len(production_data),
        }
        
        self.save_reference_data(reference, name=name, metadata=metadata)
        
        return reference


def load_reference_from_training(
    path: str = 'data/processed/train_data_s1.pt'
) -> Optional[pd.DataFrame]:
    """
    Quick helper to load reference data from training
    
    Args:
        path: Path to training data
    
    Returns:
        Reference DataFrame
    """
    import torch
    
    if not os.path.exists(path):
        print(f"⚠ Training data not found at {path}")
        return None
    
    data = torch.load(path)
    y = data['y'].numpy()
    
    df = pd.DataFrame(
        y,
        columns=['beard', 'mustache', 'glasses', 'hair_color', 'hair_length']
    )
    
    print(f"✓ Loaded reference data from training: {len(df)} samples")
    return df


if __name__ == '__main__':
    # Example usage
    print("Reference Data Manager - Example Usage")
    
    # Initialize manager
    manager = ReferenceDataManager()
    
    # Create reference from training data
    ref_data = manager.create_reference_from_training_data()
    
    if ref_data is not None:
        # List available references
        print("\nAvailable references:")
        for ref_info in manager.list_references():
            print(f"  {ref_info['filename']}")
            print(f"    Timestamp: {ref_info['timestamp']}")
            print(f"    Shape: {ref_info['shape']}")
        
        # Load reference
        loaded = manager.load_reference_data('training')
        print(f"\nLoaded reference shape: {loaded.shape}")
        print(f"Columns: {list(loaded.columns)}")
