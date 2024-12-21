from src.rhythm_predictor import RhythmPredictor
import librosa
import numpy as np
import joblib

def test_with_audio_file(audio_path):
    # Initialize the predictor
    predictor = RhythmPredictor()
    
    # Load the trained model
    try:
        predictor.model = joblib.load('models/trained_rhythm_model.joblib')
    except FileNotFoundError:
        print("Error: No trained model found. Please run train_model.py first.")
        return
    
    # Load audio file
    print(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(audio_path)
    
    # Process audio in chunks to simulate real-time
    chunk_size = 22050  # 1 second of audio at 22050Hz
    
    print("\nAnalyzing audio...")
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:  # Only process full chunks
            # Extract features
            features = predictor.feature_extractor.process_audio(chunk)
            
            # Make prediction
            tempo_prediction = predictor.model.predict([features])
            
            # Print results for this chunk
            print(f"Chunk {i//chunk_size + 1} - Predicted tempo: {tempo_prediction[0]:.2f} BPM")

if __name__ == "__main__":
    audio_file = "data/audio_samples/sample.wav"
    test_with_audio_file(audio_file)