from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio


@dataclass
class Audio:
    """Audio class that can handle torch tensor waveforms, sample rates, or audio file paths."""

    waveform: torch.Tensor  # [channels, samples] where channels is 1 or 2
    sr: int

    def __init__(
        self,
        source: Union[str, Path, torch.Tensor, np.ndarray],
        sr: Optional[int] = None,
        target_sr: Optional[int] = None,
        mono: bool = True,
    ):
        if isinstance(source, (str, Path)):
            # Load audio from file path
            waveform, sr = torchaudio.load(str(source))
        elif isinstance(source, (torch.Tensor, np.ndarray)):
            # Handle direct tensor input
            if sr is None:
                raise ValueError("sample_rate must be provided when input is a tensor or numpy array")

            waveform = source
            if isinstance(source, np.ndarray):
                waveform = torch.from_numpy(source)

            # check tensor shape is [channels, samples]
            if waveform.dim() > 2:
                raise ValueError("Tensor must have shape [channels, samples] or [samples]")
            elif waveform.dim() == 2 and waveform.shape[0] > 2:
                raise ValueError("Tensor must have shape [channels, samples] - more than 2 channels is not supported")
            elif waveform.dim() == 1:
                # expand [samples] to [channels, samples]
                waveform = waveform.unsqueeze(0)

            # check if mono
            if mono:
                waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise TypeError(f"Unsupported source type: {type(source)} ")

        # resample if target sample rate is provided
        if target_sr is not None and target_sr != sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
            sr = target_sr

        self.waveform = waveform
        self.sr = sr

    @property
    def duration(self) -> float:
        """Returns duration of audio in seconds."""
        return self.waveform.shape[-1] / self.sr

    def to_mono(self) -> "Audio":
        """Convert audio to mono by averaging channels."""
        if self.num_channels > 1:
            mono_waveform = self.waveform.mean(dim=0, keepdim=True)
            return Audio(mono_waveform, self.sample_rate)
        return self

    def resample(self, new_sr: int) -> "Audio":
        """Resample audio to new sample rate."""
        if new_sr == self.sr:
            return self

        resampler = torchaudio.transforms.Resample(self.sr, new_sr)
        resampled_waveform = resampler(self.waveform)
        return Audio(resampled_waveform, new_sr)

    def __len__(self) -> int:
        """Returns number of samples in audio."""
        return self.waveform.shape[-1]


class PitchPredictor(ABC):
    """Abstract base class for pitch prediction models."""

    @abstractmethod
    def predict(self, audio: Audio) -> torch.Tensor:
        """
        Predict pitch contour from audio.

        Args:
            audio (Audio): Input audio to predict pitch from

        Returns:
            torch.Tensor: Predicted pitch contour in Hz. Shape: [num_frames]
        """
        pass
