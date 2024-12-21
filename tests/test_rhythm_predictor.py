import pytest
import numpy as np
from src.rhythm_predictor import RhythmPredictor
from src.feature_extractor import FeatureExtractor

def test_feature_extractor():
    extractor = FeatureExtractor()
    # Create synthetic audio data (2 seconds of 440Hz sine wave)
    duration = 2
    t = np.linspace(0, duration, int(22050 * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    features = extractor.process_audio(audio)
    
    # Test feature properties
    assert len(features) == 5
    assert isinstance(features, np.ndarray)
    assert features.dtype in (np.float32, np.float64)

def test_rhythm_predictor_init():
    predictor = RhythmPredictor()
    assert predictor.sr == 22050
    assert predictor.hop_length == 512
    assert predictor.buffer is not None

def test_model_training():
    predictor = RhythmPredictor()
    
    # Create synthetic training data (5 different frequencies)
    duration = 2
    t = np.linspace(0, duration, int(22050 * duration))
    audio_data = []
    
    for freq in [440, 880, 220, 660, 330]:
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        features = predictor.feature_extractor.process_audio(audio)
        audio_data.append(features)
    
    # Convert to numpy array
    audio_data = np.array(audio_data)
    labels = np.array([120, 130, 140, 150, 160])
    
    # Train model with feature vectors directly
    score = predictor.model.fit(audio_data, labels).score(audio_data, labels)
    assert isinstance(score, float)
    assert 0 <= score <= 1