import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sr=22050, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
    
    def process_file(self, file_path):
        """Process an audio file and extract features."""
        audio, _ = librosa.load(file_path, sr=self.sr)
        return self.process_audio(audio)
    
    def process_audio(self, audio):
        """Extract features from audio data."""
        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Tempo and beat frames
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        
        # Convert all features to single float values
        features = np.array([
            float(np.mean(onset_env)),
            float(tempo),
            float(np.mean(spec_cent)),
            float(np.mean(spec_rolloff)),
            float(np.mean(rms))
        ])
        
        return features