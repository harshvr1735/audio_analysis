from setuptools import setup, find_packages

setup(
    name="audio_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'librosa>=0.10.1',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.2.0',
        'sounddevice>=0.4.6',
    ],
    author="Harshvardhan Rathore",
    author_email="harshvar@uci.edu",
    description="Real-time audio analysis and rhythm prediction system",
    keywords="audio, machine learning, rhythm prediction",
    python_requires=">=3.8",
)