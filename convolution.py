#!/usr/bin/env python3
"""Convolution Reverb implementation using FFT-based convolution.

Convolves the input signal with a recorded or synthetic impulse response (IR)
using scipy.signal.fftconvolve for efficient frequency-domain processing.

Reference: Smith, J.O. "Mathematics of the Discrete Fourier Transform (DFT)"
           (online book, CCRMA Stanford)
"""

import argparse

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Convolution reverb constants
# ---------------------------------------------------------------------------
REFERENCE_SAMPLE_RATE = 44100

# IR length mapping: ir_length 0.0 → MIN_IR_SECONDS, 1.0 → MAX_IR_SECONDS
MIN_IR_SECONDS = 0.1
MAX_IR_SECONDS = 5.0

# Default user-facing parameters (all 0-1)
DEFAULT_IR_LENGTH = 0.5
DEFAULT_DECAY = 0.5
DEFAULT_WET = 0.33
DEFAULT_DRY = 0.67
DEFAULT_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ImpulseResponse:
    """Generates a synthetic IR or loads one from a WAV file."""

    def __init__(
        self,
        sample_rate: int,
        ir_length: float = DEFAULT_IR_LENGTH,
        decay: float = DEFAULT_DECAY,
        ir_file: str | None = None,
    ):
        if ir_file is not None:
            self.ir_left, self.ir_right = self._load_ir(ir_file)
        else:
            self.ir_left, self.ir_right = self._generate_ir(
                sample_rate, ir_length, decay
            )

    def _generate_ir(
        self, sample_rate: int, ir_length: float, decay: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic IR using exponentially decaying white noise."""
        actual_seconds = MIN_IR_SECONDS + ir_length * (MAX_IR_SECONDS - MIN_IR_SECONDS)
        length = int(actual_seconds * sample_rate)

        rng = np.random.default_rng(42)
        noise_l = rng.normal(0.0, 1.0, length)
        noise_r = rng.normal(0.0, 1.0, length)

        # Exponential decay envelope
        decay_rate = 1.0 + decay * 20.0
        t = np.linspace(0.0, 1.0, length, endpoint=False)
        envelope = np.exp(-decay_rate * t)

        ir_left = noise_l * envelope
        ir_right = noise_r * envelope

        # Normalize to unit peak
        peak = max(np.max(np.abs(ir_left)), np.max(np.abs(ir_right)))
        if peak > 0:
            ir_left /= peak
            ir_right /= peak

        return ir_left, ir_right

    @staticmethod
    def _load_ir(path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load an impulse response from a WAV file."""
        _sr, data = wavfile.read(path)

        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float64) / 2147483648.0
        elif data.dtype in (np.float32, np.float64):
            data = data.astype(np.float64)
        else:
            raise ValueError(f"Unsupported IR WAV format: {data.dtype}")

        if data.ndim == 1:
            return data, data.copy()
        return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Convolution reverb processor
# ---------------------------------------------------------------------------
class ConvolutionReverb:
    """FFT-based convolution reverb processor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    ir_length : float
        Synthetic IR length (0-1, mapped to 0.1-5.0 s).
    decay : float
        Exponential decay rate for the synthetic IR (0-1).
    wet : float
        Wet (reverb) signal level.
    dry : float
        Dry (original) signal level.
    width : float
        Stereo width of the reverb (0-1).
    ir_file : str, optional
        Path to a WAV file to use as the impulse response.
    """

    def __init__(
        self,
        sample_rate: int = REFERENCE_SAMPLE_RATE,
        ir_length: float = DEFAULT_IR_LENGTH,
        decay: float = DEFAULT_DECAY,
        wet: float = DEFAULT_WET,
        dry: float = DEFAULT_DRY,
        width: float = DEFAULT_WIDTH,
        ir_file: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.ir_length = ir_length
        self.decay = decay
        self.wet = wet
        self.dry = dry
        self.width = width

        self.ir = ImpulseResponse(sample_rate, ir_length, decay, ir_file)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply convolution reverb to *audio*.

        Parameters
        ----------
        audio : np.ndarray
            Mono ``(N,)`` or stereo ``(N, 2)`` float array in [-1, 1].

        Returns
        -------
        np.ndarray
            Stereo ``(N, 2)`` float64 result.
        """
        if audio.ndim == 1:
            left_in = right_in = audio.astype(np.float64)
        else:
            left_in = audio[:, 0].astype(np.float64)
            right_in = audio[:, 1].astype(np.float64)

        num_samples = len(left_in)

        # FFT-based convolution with the impulse response
        conv_l = fftconvolve(left_in, self.ir.ir_left, mode="full")[:num_samples]
        conv_r = fftconvolve(right_in, self.ir.ir_right, mode="full")[:num_samples]

        # Stereo width and wet/dry mix
        wet1 = self.wet * (self.width / 2.0 + 0.5)
        wet2 = self.wet * ((1.0 - self.width) / 2.0)

        result_l = conv_l * wet1 + conv_r * wet2 + left_in * self.dry
        result_r = conv_r * wet1 + conv_l * wet2 + right_in * self.dry

        return np.column_stack([result_l, result_r])


# ---------------------------------------------------------------------------
# WAV file helper
# ---------------------------------------------------------------------------
def apply_reverb(
    input_path: str,
    output_path: str,
    ir_length: float = DEFAULT_IR_LENGTH,
    decay: float = DEFAULT_DECAY,
    wet: float = DEFAULT_WET,
    dry: float = DEFAULT_DRY,
    width: float = DEFAULT_WIDTH,
    ir_file: str | None = None,
) -> tuple:
    """Read a WAV file, apply convolution reverb, and write the result.

    Returns ``(sample_rate, output_array)``.
    """
    sample_rate, data = wavfile.read(input_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
        out_dtype, out_scale = np.int16, 32767.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
        out_dtype, out_scale = np.int32, 2147483647.0
    elif data.dtype in (np.float32, np.float64):
        audio = data.astype(np.float64)
        out_dtype, out_scale = data.dtype, None
    else:
        raise ValueError(f"Unsupported WAV sample format: {data.dtype}")

    reverb = ConvolutionReverb(sample_rate, ir_length, decay, wet, dry, width, ir_file)
    result = reverb.process(audio)

    result = np.clip(result, -1.0, 1.0)
    if out_scale is not None:
        result = (result * out_scale).astype(out_dtype)
    else:
        result = result.astype(out_dtype)

    wavfile.write(output_path, sample_rate, result)
    return sample_rate, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apply convolution reverb to a WAV file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--ir-length", type=float, default=DEFAULT_IR_LENGTH)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--wet", type=float, default=DEFAULT_WET)
    parser.add_argument("--dry", type=float, default=DEFAULT_DRY)
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--ir-file",
        type=str,
        default=None,
        help="Path to a WAV impulse response file (overrides synthetic IR)",
    )
    args = parser.parse_args()

    apply_reverb(
        args.input,
        args.output,
        args.ir_length,
        args.decay,
        args.wet,
        args.dry,
        args.width,
        args.ir_file,
    )
    print(f"Convolution reverb applied: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
