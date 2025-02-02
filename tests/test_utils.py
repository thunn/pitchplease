import numpy as np
import pytest
import torch

from pitchplease.utils import Audio


def test_audio_torch_tensor():
    """Test Audio class with random torch tensor input"""
    sr = 16000
    # Create random torch tensor (1D)
    audio_data = torch.rand(16000)  # 1 second of audio at 16kHz
    audio = Audio(audio_data, sr=sr)

    assert audio.waveform.dim() == 2
    waveform_channels = audio.waveform.shape[0]
    assert waveform_channels == 1


def test_audio_torch_tensor_1_channel():
    """Test Audio class with 1 channel torch tensor input"""
    sr = 16000
    audio_data = torch.rand(16000)
    audio = Audio(audio_data, sr=sr)

    assert audio.waveform.dim() == 2
    waveform_channels = audio.waveform.shape[0]
    assert waveform_channels == 1


def test_audio_torch_tensor_2_channels():
    """Test Audio class with 2 channel torch tensor input"""
    sr = 16000
    audio_data = torch.rand(2, 16000)
    audio = Audio(audio_data, sr=sr)

    assert audio.waveform.dim() == 2
    waveform_channels = audio.waveform.shape[0]
    assert waveform_channels == 1  # mono is set to True as default


def test_auidio_torch_tensor_2_channels_mono():
    """Test Audio class with 2 channel torch tensor input (mono)"""
    sr = 16000
    audio_data = torch.rand(2, 16000)
    audio = Audio(audio_data, sr=sr, mono=True)

    assert audio.waveform.dim() == 2
    waveform_channels = audio.waveform.shape[0]
    assert waveform_channels == 1


def test_audio_torch_tensor_3_channels():
    """Test Audio class with 3 channel torch tensor input"""
    sr = 16000
    audio_data = torch.rand(3, 16000)
    with pytest.raises(
        ValueError, match=r"Tensor must have shape \[channels, samples\] - more than 2 channels is not supported"
    ):
        Audio(audio_data, sr=sr)


def test_audio_numpy():
    """Test Audio class with random numpy array input"""
    sr = 16000
    # Create random numpy array
    audio_data = np.random.rand(16000)
    audio = Audio(audio_data, sr=sr)

    assert audio.waveform.dim() == 2
    waveform_channels = audio.waveform.shape[0]
    assert waveform_channels == 1


def test_audio_missing_sample_rate():
    """Test Audio class without providing sample rate"""
    audio_data = torch.rand(16000)

    with pytest.raises(ValueError, match="sample_rate must be provided when input is a tensor or numpy array"):
        Audio(audio_data)


def test_audio_resample():
    """Test Audio class with new target sample rate"""
    original_sr = 44100
    target_sr = 16000
    # Create 1 second of audio at 44.1kHz
    audio_data = torch.rand(44100)

    audio = Audio(audio_data, sr=original_sr, target_sr=target_sr)

    # Expected length after resampling (1 second * new_rate)
    expected_length = int(len(audio_data) * target_sr / original_sr)
    assert len(audio) == expected_length
