"""
Tests for model architecture
"""
import pytest
import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.architecture import ResidualBlock, CustomMultiHeadCNN


class TestResidualBlock:
    """Test cases for ResidualBlock"""
    
    def test_residual_block_forward_same_channels(self, sample_batch):
        """Test forward pass with same input/output channels"""
        block = ResidualBlock(in_channels=3, out_channels=3, stride=1)
        output = block(sample_batch)
        
        # Output shape should match input shape
        assert output.shape == sample_batch.shape
        
    def test_residual_block_forward_different_channels(self):
        """Test forward pass with different input/output channels"""
        x = torch.randn(2, 64, 32, 32)
        block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        output = block(x)
        
        # Output should have different channels and spatial dimensions
        assert output.shape == (2, 128, 16, 16)
        
    def test_residual_block_shortcut(self):
        """Test that shortcut connection is created when needed"""
        block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        
        # Shortcut should not be empty when channels or stride changes
        assert len(block.shortcut) > 0


class TestCustomMultiHeadCNN:
    """Test cases for CustomMultiHeadCNN"""
    
    def test_model_initialization(self):
        """Test that model can be initialized"""
        model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=0.3)
        assert model is not None
        
    def test_model_forward(self, sample_batch):
        """Test forward pass produces correct output structure"""
        model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=0.3)
        outputs = model(sample_batch)
        
        # Check that all expected keys are present
        expected_keys = ["beard", "mustache", "glasses", "hair_color", "hair_length"]
        assert set(outputs.keys()) == set(expected_keys)
        
    def test_model_output_shapes(self, sample_batch):
        """Test that output shapes are correct"""
        batch_size = sample_batch.shape[0]
        model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=0.3)
        outputs = model(sample_batch)
        
        # Binary classification outputs
        assert outputs["beard"].shape == (batch_size,)
        assert outputs["mustache"].shape == (batch_size,)
        assert outputs["glasses"].shape == (batch_size,)
        
        # Multi-class outputs
        assert outputs["hair_color"].shape == (batch_size, 5)
        assert outputs["hair_length"].shape == (batch_size, 3)
        
    def test_model_with_different_classes(self):
        """Test model with different number of classes"""
        model = CustomMultiHeadCNN(n_color=7, n_length=4, dropout=0.5)
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)
        
        assert outputs["hair_color"].shape == (2, 7)
        assert outputs["hair_length"].shape == (2, 4)
        
    def test_model_eval_mode(self, sample_batch):
        """Test model in evaluation mode"""
        model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=0.3)
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_batch)
            
        # Should still produce valid outputs
        assert outputs["beard"].shape[0] == sample_batch.shape[0]
        
    def test_model_parameter_count(self):
        """Test that model has trainable parameters"""
        model = CustomMultiHeadCNN(n_color=5, n_length=3, dropout=0.3)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
