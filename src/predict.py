"""
Wine Quality Prediction
"""
import pickle
import numpy as np

def load_model():
    """Load trained model"""
    with open('models/wine_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(features):
    """
    Predict wine class
    
    Args:
        features: List of 13 features
        
    Returns:
        Predicted class name
    """
    model = load_model()
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    
    class_names = ['class_0', 'class_1', 'class_2']
    return class_names[prediction[0]]

if __name__ == "__main__":
    # Example prediction
    sample = [13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 0.58, 0.28, 0.45, 6.5, 1.05, 3.33, 820.0]
    result = predict(sample)
    print(f"Prediction: {result}")