# Audio Analysis and Rhythm Prediction

# Project Structure

audio_analysis/

├── src/
│   ├── __init__.py
│   ├── rhythm_predictor.py
│   ├── feature_extractor.py
│   └── utils.py
├── data/
│   └── audio_samples/
│       └── sample.wav
├── tests/
│   ├── __init__.py
│   └── test_rhythm_predictor.py
├── requirements.txt
├── README.md
├── setup.py
└── report.md

## Overview
This project implements a real-time audio analysis system capable of predicting rhythm patterns and tempo from audio input. It was developed as part of an internship project for Sutra Sphere LLC.

## Features
- Real-time audio processing
- Rhythm and tempo prediction
- Feature extraction from audio signals
- Machine learning model for pattern recognition

## Installation
1. Clone the repository:
```bash
git clone https://github.com/harshvr1735/audio_analysis.git
cd audio_analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.rhythm_predictor import RhythmPredictor

# Initialize predictor
predictor = RhythmPredictor()

# Train model
predictor.train_model(audio_files, labels)

# Start real-time prediction
predictor.real_time_predict(duration=30)
```

## Testing
Run tests using pytest:
```bash
pytest tests/
```
