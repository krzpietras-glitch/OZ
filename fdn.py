#!/usr/bin/env python3
"""Feedback Delay Network (FDN) reverb implementation using only numpy and scipy.

An FDN uses multiple delay lines whose outputs are mixed through a unitary
matrix (Hadamard) and fed back into each other, producing dense, smooth
reverberation tails.  Input allpass diffusers increase early echo density.

Reference: Jot, J.-M. & Chaigne, A., "Digital Delay Networks for Designing
Artificial Reverberators" (1991).
"""

import argparse

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# FDN constants (reference delay lengths at 44100 Hz — all primes)
# ---------------------------------------------------------------------------
DELAY_LENGTHS = [1087, 1187, 1291, 1399, 1493, 1601, 1699, 1801]
NUM_DELAYS = len(DELAY_LENGTHS)
DIFFUSER_DELAYS = [142, 107, 379, 277]
STEREO_SPREAD = 23
REFERENCE_SAMPLE_RATE = 44100

# Default user-facing parameters
DEFAULT_ROOM_SIZE = 0.5
DEFAULT_DAMPING = 0.5
DEFAULT_DENSITY = 0.5
DEFAULT_DECAY = 0.5
DEFAULT_WET = 0.33
DEFAULT_DRY = 0.67
DEFAULT_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Filter building blocks
# ---------------------------------------------------------------------------
class DelayLine:
    """Simple fixed-length delay line."""

    def __init__(self, delay_length: int):
        self.buffer = np.zeros(max(delay_length, 1))
        self.buf_len = max(delay_length, 1)
        self.index = 0

    def process(self, input_sample: float) -> float:
        output = self.buffer[self.index]
        self.buffer[self.index] = input_sample
        self.index = (self.index + 1) % self.buf_len
        return output


class DampingFilter:
    """One-pole lowpass filter for damping in the feedback path."""

    def __init__(self, damping: float):
        self.damp1 = damping
        self.damp2 = 1.0 - damping
        self.state = 0.0

    def process(self, input_sample: float) -> float:
        self.state = input_sample * self.damp2 + self.state * self.damp1
        return self.state


class InputDiffuser:
    """Schroeder allpass filter used for input diffusion."""

    def __init__(self, delay_length: int, feedback: float):
        self.buffer = np.zeros(max(delay_length, 1))
        self.buf_len = max(delay_length, 1)
        self.index = 0
        self.feedback = feedback

    def process(self, input_sample: float) -> float:
        buffered = self.buffer[self.index]
        output = buffered - input_sample * self.feedback
        self.buffer[self.index] = input_sample + buffered * self.feedback
        self.index = (self.index + 1) % self.buf_len
        return output


# ---------------------------------------------------------------------------
# Hadamard matrix construction
# ---------------------------------------------------------------------------
def _hadamard_matrix(n: int) -> np.ndarray:
    """Build a normalised *n* x *n* Hadamard matrix (*n* must be a power of 2)."""
    if n == 1:
        return np.array([[1.0]])
    half = _hadamard_matrix(n // 2)
    top = np.hstack([half, half])
    bottom = np.hstack([half, -half])
    return np.vstack([top, bottom]) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# FDN Reverb processor
# ---------------------------------------------------------------------------
class FDNReverb:
    """Feedback Delay Network reverb processor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    room_size : float
        Scales delay line lengths (0-1).  Higher = larger virtual room.
    damping : float
        Lowpass damping in feedback paths (0-1).  Higher = darker tail.
    density : float
        Input diffusion amount (0-1).  Higher = denser early reflections.
    decay : float
        Feedback gain applied after Hadamard mixing (0-1).  Higher = longer tail.
    wet : float
        Wet (reverb) signal level.
    dry : float
        Dry (original) signal level.
    width : float
        Stereo width of the reverb (0-1).
    """

    def __init__(
        self,
        sample_rate: int = REFERENCE_SAMPLE_RATE,
        room_size: float = DEFAULT_ROOM_SIZE,
        damping: float = DEFAULT_DAMPING,
        density: float = DEFAULT_DENSITY,
        decay: float = DEFAULT_DECAY,
        wet: float = DEFAULT_WET,
        dry: float = DEFAULT_DRY,
        width: float = DEFAULT_WIDTH,
    ):
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.damping = damping
        self.density = density
        self.decay = decay
        self.wet = wet
        self.dry = dry
        self.width = width

        scale = sample_rate / REFERENCE_SAMPLE_RATE
        room_scale = 0.5 + room_size  # range 0.5x – 1.5x

        self.delays = [
            DelayLine(max(int(d * scale * room_scale), 1))
            for d in DELAY_LENGTHS
        ]
        self.dampers = [DampingFilter(damping) for _ in range(NUM_DELAYS)]

        # Input diffusers — density controls allpass feedback
        self.diffusers = [
            InputDiffuser(int(d * scale), density * 0.75)
            for d in DIFFUSER_DELAYS
        ]

        # Unitary mixing matrix
        self.mix_matrix = _hadamard_matrix(NUM_DELAYS)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply FDN reverb to *audio*.

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
        out_l = np.zeros(num_samples)
        out_r = np.zeros(num_samples)

        delays = self.delays
        dampers = self.dampers
        diffusers = self.diffusers
        mix_matrix = self.mix_matrix
        decay = self.decay
        n = NUM_DELAYS

        feedback = np.zeros(n)

        for i in range(num_samples):
            inp = (left_in[i] + right_in[i]) * 0.5

            # Input diffusion
            diffused = inp
            for diff in diffusers:
                diffused = diff.process(diffused)

            # Read from delay lines (write input + feedback)
            delay_outs = np.zeros(n)
            for j in range(n):
                delay_outs[j] = delays[j].process(diffused + feedback[j])

            # Damping
            for j in range(n):
                delay_outs[j] = dampers[j].process(delay_outs[j])

            # Hadamard mixing with decay
            feedback = mix_matrix @ delay_outs * decay

            # Stereo tap: even delay lines -> left, odd -> right
            out_l[i] = np.sum(delay_outs[0::2])
            out_r[i] = np.sum(delay_outs[1::2])

        # Normalise tapped output (N/2 lines per channel)
        out_l /= (n / 2)
        out_r /= (n / 2)

        # Stereo width and wet/dry mix
        wet1 = self.wet * (self.width / 2.0 + 0.5)
        wet2 = self.wet * ((1.0 - self.width) / 2.0)

        result_l = out_l * wet1 + out_r * wet2 + left_in * self.dry
        result_r = out_r * wet1 + out_l * wet2 + right_in * self.dry

        return np.column_stack([result_l, result_r])


# ---------------------------------------------------------------------------
# WAV file helper
# ---------------------------------------------------------------------------
def apply_reverb(
    input_path: str,
    output_path: str,
    room_size: float = DEFAULT_ROOM_SIZE,
    damping: float = DEFAULT_DAMPING,
    density: float = DEFAULT_DENSITY,
    decay: float = DEFAULT_DECAY,
    wet: float = DEFAULT_WET,
    dry: float = DEFAULT_DRY,
    width: float = DEFAULT_WIDTH,
) -> tuple:
    """Read a WAV file, apply FDN reverb, and write the result.

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

    reverb = FDNReverb(sample_rate, room_size, damping, density, decay, wet, dry, width)
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
        description="Apply FDN reverb to a WAV file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--room-size", type=float, default=DEFAULT_ROOM_SIZE)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--wet", type=float, default=DEFAULT_WET)
    parser.add_argument("--dry", type=float, default=DEFAULT_DRY)
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    apply_reverb(
        args.input,
        args.output,
        args.room_size,
        args.damping,
        args.density,
        args.decay,
        args.wet,
        args.dry,
        args.width,
    )
    print(f"Reverb applied: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
