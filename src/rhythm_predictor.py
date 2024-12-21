import numpy as np
import queue
import threading
import time
from .feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestRegressor
import sounddevice as sd

class RhythmPredictor:
    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
        self.buffer = queue.Queue()
        self.feature_extractor = FeatureExtractor(sr=sr, hop_length=hop_length)
        self.model = RandomForestRegressor(n_estimators=100)
        
    def train_model(self, inputs, labels):
        """
        Train the rhythm prediction model.
        
        Args:
            inputs: List of either file paths or audio arrays
            labels (list): Corresponding rhythm/tempo labels
        
        Returns:
            float: Model accuracy score
        """
        features_list = []
        
        for input_data in inputs:
            if isinstance(input_data, (list, np.ndarray)):
                # If input is already audio data
                features = self.feature_extractor.process_audio(input_data)
            else:
                # If input is a file path
                features = self.feature_extractor.process_file(input_data)
            features_list.append(features)
            
        X = np.array(features_list)
        y = np.array(labels)
        
        return self.model.fit(X, y).score(X, y)
    
    def real_time_predict(self, duration=10, buffer_size=2048):
        """
        Perform real-time prediction on audio input.
        
        Args:
            duration (int): Duration to run prediction in seconds
            buffer_size (int): Audio buffer size
        """
        def audio_callback(indata, frames, time, status):
            self.buffer.put(indata.copy())
        
        stream = sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=buffer_size,
            callback=audio_callback
        )
        
        with stream:
            print("Starting real-time prediction...")
            start_time = time.time()
            
            while time.time() - start_time < duration:
                if not self.buffer.empty():
                    audio_chunk = self.buffer.get()
                    features = self.feature_extractor.process_audio(audio_chunk.flatten())
                    prediction = self.model.predict([features])
                    print(f"Predicted rhythm pattern: {prediction[0]}")
                    
                time.sleep(0.1)