#!/usr/bin/env python3
"""Gerzon Reverb — Feedback Delay Network with Hadamard mixing.

Michael Gerzon's approach uses multiple delay lines mixed through a unitary
(Hadamard) matrix with slowly modulated delay lengths to prevent metallic
artifacts.  This creates dense, natural-sounding reverberation.

Reference: Michael Gerzon, "Unitary (Energy-Preserving) Multichannel Networks
with Feedback" (1976).
"""

import argparse
import math

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Gerzon FDN constants (reference delay lengths at 44100 Hz)
# ---------------------------------------------------------------------------
DELAY_LENGTHS = [1087, 1283, 1447, 1601]
NUM_DELAYS = len(DELAY_LENGTHS)
REFERENCE_SAMPLE_RATE = 44100

# 4x4 Hadamard matrix normalised to be unitary (H H^T = I)
HADAMARD_4 = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float64,
) * 0.5

# Default user-facing parameters
DEFAULT_ROOM_SIZE = 0.5
DEFAULT_DAMPING = 0.5
DEFAULT_MODULATION_DEPTH = 0.3
DEFAULT_MODULATION_RATE = 0.5
DEFAULT_WET = 0.33
DEFAULT_DRY = 0.67
DEFAULT_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Filter building blocks
# ---------------------------------------------------------------------------
class DampingFilter:
    """One-pole lowpass filter for damping in the feedback path."""

    def __init__(self, damping: float):
        self.damp1 = damping
        self.damp2 = 1.0 - damping
        self.state = 0.0

    def process(self, sample: float) -> float:
        self.state = sample * self.damp2 + self.state * self.damp1
        return self.state


class ModulatedDelayLine:
    """Delay line with LFO-modulated read position.

    Because the FDN architecture requires reading all delay outputs before
    computing the feedback signals, this class exposes separate ``read`` and
    ``write`` methods rather than a single ``process`` call.
    """

    def __init__(self, delay_length: int, max_mod_samples: int):
        self.buf_len = delay_length + max_mod_samples + 2
        self.buffer = np.zeros(self.buf_len)
        self.write_index = 0
        self.base_delay = delay_length
        self.max_mod = max_mod_samples

    def read(self, modulation: float = 0.0) -> float:
        """Read from the delay line with a modulated offset.

        Parameters
        ----------
        modulation : float
            Value in ``[-1, 1]`` controlling the read-position offset.
            Scaled internally by *max_mod_samples*.
        """
        mod_offset = modulation * self.max_mod
        total_delay = self.base_delay + mod_offset
        read_pos = (self.write_index - int(round(total_delay))) % self.buf_len
        return self.buffer[read_pos]

    def write(self, sample: float) -> None:
        """Write *sample* at the current position and advance the pointer."""
        self.buffer[self.write_index] = sample
        self.write_index = (self.write_index + 1) % self.buf_len


# ---------------------------------------------------------------------------
# Gerzon Reverb processor
# ---------------------------------------------------------------------------
class GerzonReverb:
    """Feedback Delay Network reverb with Hadamard mixing and LFO modulation.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    room_size : float
        Feedback gain for delay lines (0-1).  Higher = longer decay.
    damping : float
        Lowpass damping in feedback path (0-1).  Higher = darker tail.
    modulation_depth : float
        Depth of delay-length modulation (0-1).
    modulation_rate : float
        Rate of LFO modulation (0-1), scaled to 0-2 Hz.
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
        modulation_depth: float = DEFAULT_MODULATION_DEPTH,
        modulation_rate: float = DEFAULT_MODULATION_RATE,
        wet: float = DEFAULT_WET,
        dry: float = DEFAULT_DRY,
        width: float = DEFAULT_WIDTH,
    ):
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.damping = damping
        self.modulation_depth = modulation_depth
        self.modulation_rate = modulation_rate
        self.wet = wet
        self.dry = dry
        self.width = width

        scale = sample_rate / REFERENCE_SAMPLE_RATE

        # Maximum modulation excursion in samples (0 when depth is 0)
        max_mod_samples = int(16 * scale * modulation_depth)

        self.delays = [
            ModulatedDelayLine(int(d * scale), max_mod_samples)
            for d in DELAY_LENGTHS
        ]
        self.dampers = [DampingFilter(damping) for _ in range(NUM_DELAYS)]

        # LFO phases staggered across delay lines
        self.lfo_phases = [i * math.pi * 0.5 for i in range(NUM_DELAYS)]
        # LFO frequency in radians/sample (rate 0-1 → 0-2 Hz)
        self.lfo_increment = 2.0 * math.pi * (modulation_rate * 2.0) / sample_rate

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply reverb to *audio*.

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
        feedback = self.room_size
        lfo_phases = list(self.lfo_phases)
        lfo_inc = self.lfo_increment

        h = HADAMARD_4

        for i in range(num_samples):
            inp = (left_in[i] + right_in[i]) * 0.5

            # --- Read from delay lines with LFO modulation ---
            delay_outs = [0.0] * NUM_DELAYS
            for k in range(NUM_DELAYS):
                mod = math.sin(lfo_phases[k])
                delay_outs[k] = delays[k].read(mod)
                lfo_phases[k] += lfo_inc

            # --- Mix through Hadamard matrix ---
            mixed = [0.0] * NUM_DELAYS
            for k in range(NUM_DELAYS):
                s = 0.0
                for j in range(NUM_DELAYS):
                    s += h[k, j] * delay_outs[j]
                mixed[k] = s

            # --- Apply damping and write back with feedback ---
            for k in range(NUM_DELAYS):
                damped = dampers[k].process(mixed[k])
                delays[k].write(inp + damped * feedback)

            # --- Tap stereo outputs: L from 0,1  R from 2,3 ---
            out_l[i] = delay_outs[0] + delay_outs[1]
            out_r[i] = delay_outs[2] + delay_outs[3]

        self.lfo_phases = lfo_phases

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
    modulation_depth: float = DEFAULT_MODULATION_DEPTH,
    modulation_rate: float = DEFAULT_MODULATION_RATE,
    wet: float = DEFAULT_WET,
    dry: float = DEFAULT_DRY,
    width: float = DEFAULT_WIDTH,
) -> tuple:
    """Read a WAV file, apply Gerzon reverb, and write the result.

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

    reverb = GerzonReverb(
        sample_rate, room_size, damping, modulation_depth, modulation_rate,
        wet, dry, width,
    )
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
        description="Apply Gerzon FDN reverb to a WAV file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--room-size", type=float, default=DEFAULT_ROOM_SIZE)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--modulation-depth", type=float, default=DEFAULT_MODULATION_DEPTH)
    parser.add_argument("--modulation-rate", type=float, default=DEFAULT_MODULATION_RATE)
    parser.add_argument("--wet", type=float, default=DEFAULT_WET)
    parser.add_argument("--dry", type=float, default=DEFAULT_DRY)
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    apply_reverb(
        args.input,
        args.output,
        args.room_size,
        args.damping,
        args.modulation_depth,
        args.modulation_rate,
        args.wet,
        args.dry,
        args.width,
    )
    print(f"Reverb applied: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
