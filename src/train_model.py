from src.rhythm_predictor import RhythmPredictor
import librosa
import numpy as np
import joblib
import os

def train_rhythm_model(audio_path):
    # Initialize the predictor
    predictor = RhythmPredictor()
    
    # Load audio file
    print(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(audio_path)
    
    # Process audio in chunks to gather training data
    chunk_size = 22050  # 1 second of audio at 22050Hz
    features_list = []
    tempo_labels = []
    
    print("\nProcessing training data...")
    
    # Use librosa to get the ground truth tempo for training
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:  # Only process full chunks
            # Extract features
            features = predictor.feature_extractor.process_audio(chunk)
            features_list.append(features)
            tempo_labels.append(tempo)
    
    # Convert to numpy arrays
    X_train = np.array(features_list)
    y_train = np.array(tempo_labels)
    
    # Train the model
    print("Training model...")
    predictor.model.fit(X_train, y_train)
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/trained_rhythm_model.joblib'
    joblib.dump(predictor.model, model_path)
    print(f"Model saved to {model_path}")
    
    return predictor

if __name__ == "__main__":
    audio_file = "data/audio_samples/sample.wav"
    trained_predictor = train_rhythm_model(audio_file)