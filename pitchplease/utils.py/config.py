from dataclasses import dataclass


class MelConfig(dataclass):
    fmin: int
    fmax: int
    window_length: int


class InputConfig(dataclass):
    sample_rate: int
    mel_config: MelConfig


class ModelConfig(dataclass):
    model: str
    input_config: InputConfig
