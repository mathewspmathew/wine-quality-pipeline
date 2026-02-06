"""
Unit Tests for Prediction Module
"""
import pytest
from src.predict import predict, load_model
from src.train import train_model
import os

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure model exists before tests"""
    if not os.path.exists('models/wine_model.pkl'):
        train_model()

class TestPredictFunction:
    """Tests for predict function"""
    
    def test_predict_returns_string(self):
        """Test prediction returns a string"""
        features = [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.28, 0.45, 6.5, 1.05, 3.33, 820.0]
        result = predict(features)
        assert isinstance(result, str)
    
    def test_predict_valid_class(self):
        """Test prediction returns valid class name"""
        features = [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.28, 0.45, 6.5, 1.05, 3.33, 820.0]
        result = predict(features)
        valid_classes = ['class_0', 'class_1', 'class_2']
        assert result in valid_classes
    
    def test_predict_class_0(self):
        """Test prediction for class 0 wine"""
        # Sample features for class 0
        features = [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]
        result = predict(features)
        assert result == 'class_0'
    
    def test_predict_wrong_feature_count(self):
        """Test prediction fails with wrong features"""
        with pytest.raises(Exception):
            predict([1.0, 2.0, 3.0])  # Only 3 features instead of 13
    
    def test_predict_consistency(self):
        """Test same input gives same output"""
        features = [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.28, 0.45, 6.5, 1.05, 3.33, 820.0]
        result1 = predict(features)
        result2 = predict(features)
        assert result1 == result2

class TestLoadModel:
    """Tests for load_model function"""
    
    def test_load_model_succeeds(self):
        """Test model loads successfully"""
        model = load_model()
        assert model is not None
    
    def test_load_model_has_predict(self):
        """Test loaded model has predict method"""
        model = load_model()
        assert hasattr(model, 'predict')