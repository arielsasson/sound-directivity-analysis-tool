"""
Audio processing module for directivity analysis.

This module contains functions for reading audio files, applying filters,
calculating SPL values, and processing calibration data.
"""

import numpy as np
import soundfile as sf
import math
from scipy.signal import butter, sosfilt
from typing import Optional, Tuple

from config import (
    CALIBRATION_FREQUENCY_LOWCUT,
    CALIBRATION_FREQUENCY_HIGHCUT,
    CALIBRATION_SPL_LEVEL
)


def read_wav_normalized(path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Read a WAV file and return the signal and sample rate.
    
    Args:
        path: Path to the WAV file
        
    Returns:
        Tuple of (signal, sample_rate) or (None, None) if error
    """
    try:
        signal, fs = sf.read(path)
        return signal, fs
    except FileNotFoundError:
        print(f"[Archivo no encontrado] {path}")
        return None, None
    except Exception as e:
        print(f"[Error leyendo archivo] {path} -> {e}")
        return None, None


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 6):
    """
    Create a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        fs: Sample rate
        order: Filter order
        
    Returns:
        Second-order sections filter coefficients
    """
    return butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')


def filter_and_calculate_rms(signal: np.ndarray, fs: int, lowcut: float, highcut: float) -> float:
    """
    Apply bandpass filter and calculate RMS in dB.
    
    Args:
        signal: Input audio signal
        fs: Sample rate
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        
    Returns:
        RMS value in dB
    """
    sos = butter_bandpass(lowcut, highcut, fs)
    filtered_signal = sosfilt(sos, signal)
    rms = np.sqrt(np.mean(filtered_signal**2))
    return 20 * np.log10(rms)


def calculate_calibration_offset(
    calibration_wav_path: str, 
    lowcut: float, 
    highcut: float, 
    real_spl_level: float = CALIBRATION_SPL_LEVEL
) -> Optional[float]:
    """
    Calculate calibration offset from calibration file.
    
    Args:
        calibration_wav_path: Path to calibration WAV file
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        real_spl_level: Real SPL level in dB
        
    Returns:
        Calibration offset in dB or None if error
    """
    signal, fs = read_wav_normalized(calibration_wav_path)
    if not fs:
        return None
    
    dbfs = filter_and_calculate_rms(signal, fs, lowcut, highcut)
    offset = real_spl_level - dbfs
    return offset


def calculate_recording_spl(
    recording_wav_path: str, 
    offset_db: float, 
    lowcut: float, 
    highcut: float
) -> Optional[float]:
    """
    Calculate SPL for a recording using calibration offset.
    
    Args:
        recording_wav_path: Path to recording WAV file
        offset_db: Calibration offset in dB
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        
    Returns:
        SPL value in dB or None if error
    """
    signal, fs = read_wav_normalized(recording_wav_path)
    if not fs:
        return None
    
    # Ensure highcut doesn't exceed Nyquist frequency
    max_highcut = min(highcut, math.floor(fs/2) - 1)
    dbfs = filter_and_calculate_rms(signal, fs, lowcut, max_highcut)
    return dbfs + offset_db


def get_third_octave_edges(center_freq: float) -> Tuple[float, float]:
    """
    Calculate third-octave band edges for a given center frequency.
    
    Args:
        center_freq: Center frequency in Hz
        
    Returns:
        Tuple of (low_edge, high_edge) frequencies
    """
    low_edge = center_freq / (2 ** (1/6))
    high_edge = center_freq * (2 ** (1/6))
    return low_edge, high_edge


def get_octave_edges(center_freq: float) -> Tuple[float, float]:
    """
    Calculate octave band edges for a given center frequency.
    
    Args:
        center_freq: Center frequency in Hz
        
    Returns:
        Tuple of (low_edge, high_edge) frequencies
    """
    low_edge = center_freq / (2 ** (1/2))
    high_edge = center_freq * (2 ** (1/2))
    return low_edge, high_edge
