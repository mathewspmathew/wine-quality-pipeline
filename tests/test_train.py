"""
Unit Tests for Model Training
"""
import pytest
import os
from src.train import train_model

class TestModelTraining:
    """Tests for train_model function"""
    
    def test_train_model_runs(self):
        """Test that training completes without errors"""
        accuracy = train_model()
        assert accuracy is not None
    
    def test_model_accuracy_threshold(self):
        """Test model meets minimum accuracy"""
        accuracy = train_model()
        assert accuracy >= 0.85, f"Accuracy {accuracy} below 85% threshold"
    
    def test_model_file_created(self):
        """Test that model file is saved"""
        train_model()
        assert os.path.exists('models/wine_model.pkl')
    
    def test_model_file_not_empty(self):
        """Test model file has content"""
        train_model()
        file_size = os.path.getsize('models/wine_model.pkl')
        assert file_size > 0, "Model file is empty"