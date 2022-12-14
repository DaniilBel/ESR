import numpy as np
import librosa


def noise(data):
    noise_amp = 0.015 * np.random.uniform() * np.amax(data)  # 0.035
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.1):  # 0.8
    return librosa.effects.time_stretch(data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)  # 1000
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.1):  # 0.7
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)
