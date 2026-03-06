#!/usr/bin/env python3
"""Schroeder reverb implementation using only numpy and scipy.

Manfred Schroeder's classic reverb (1962) uses 4 parallel feedback comb
filters summed together and fed into 2 series allpass filters.  Stereo
spread is achieved by offsetting the right-channel delay lines.

Reference:
    M. R. Schroeder, "Natural Sounding Artificial Reverberation,"
    Journal of the Audio Engineering Society, vol. 10, no. 3, 1962.

Reference delay lengths are for 44100 Hz and are scaled for other rates.
"""

import argparse

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Schroeder constants (classic values at 44100 Hz)
# ---------------------------------------------------------------------------
COMB_DELAYS = [1687, 1601, 1493, 1423]
ALLPASS_DELAYS = [241, 83]
STEREO_SPREAD = 23
ALLPASS_FEEDBACK = 0.7
REFERENCE_SAMPLE_RATE = 44100

# Default user-facing parameters
DEFAULT_ROOM_SIZE = 0.5
DEFAULT_DAMPING = 0.5
DEFAULT_WET = 0.33
DEFAULT_DRY = 0.67
DEFAULT_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Filter building blocks
# ---------------------------------------------------------------------------
class CombFilter:
    """Feedback comb filter with a one-pole lowpass in the feedback path."""

    def __init__(self, delay_length: int, feedback: float, damping: float):
        self.buffer = np.zeros(delay_length)
        self.buf_len = delay_length
        self.index = 0
        self.feedback = feedback
        self.damp1 = damping
        self.damp2 = 1.0 - damping
        self.filterstore = 0.0

    def process(self, input_sample: float) -> float:
        output = self.buffer[self.index]
        self.filterstore = output * self.damp2 + self.filterstore * self.damp1
        self.buffer[self.index] = input_sample + self.filterstore * self.feedback
        self.index = (self.index + 1) % self.buf_len
        return output


class AllpassFilter:
    """Schroeder allpass filter."""

    def __init__(self, delay_length: int, feedback: float = ALLPASS_FEEDBACK):
        self.buffer = np.zeros(delay_length)
        self.buf_len = delay_length
        self.index = 0
        self.feedback = feedback

    def process(self, input_sample: float) -> float:
        buffered = self.buffer[self.index]
        output = buffered - input_sample
        self.buffer[self.index] = input_sample + buffered * self.feedback
        self.index = (self.index + 1) % self.buf_len
        return output


# ---------------------------------------------------------------------------
# Schroeder reverb processor
# ---------------------------------------------------------------------------
class SchroederReverb:
    """Stereo Schroeder reverb processor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    room_size : float
        Feedback gain for comb filters (0-1). Higher = longer decay.
    damping : float
        Lowpass damping in comb feedback path (0-1). Higher = darker tail.
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
        wet: float = DEFAULT_WET,
        dry: float = DEFAULT_DRY,
        width: float = DEFAULT_WIDTH,
    ):
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.damping = damping
        self.wet = wet
        self.dry = dry
        self.width = width

        scale = sample_rate / REFERENCE_SAMPLE_RATE

        self.combs_l = [
            CombFilter(int(d * scale), room_size, damping) for d in COMB_DELAYS
        ]
        self.combs_r = [
            CombFilter(int(d * scale) + STEREO_SPREAD, room_size, damping)
            for d in COMB_DELAYS
        ]
        self.allpasses_l = [AllpassFilter(int(d * scale)) for d in ALLPASS_DELAYS]
        self.allpasses_r = [
            AllpassFilter(int(d * scale) + STEREO_SPREAD) for d in ALLPASS_DELAYS
        ]

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

        combs_l = self.combs_l
        combs_r = self.combs_r
        allpasses_l = self.allpasses_l
        allpasses_r = self.allpasses_r

        for i in range(num_samples):
            inp = (left_in[i] + right_in[i]) * 0.5

            # Parallel comb filters
            cl = 0.0
            cr = 0.0
            for c in combs_l:
                cl += c.process(inp)
            for c in combs_r:
                cr += c.process(inp)

            # Series allpass filters
            for ap in allpasses_l:
                cl = ap.process(cl)
            for ap in allpasses_r:
                cr = ap.process(cr)

            out_l[i] = cl
            out_r[i] = cr

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
    wet: float = DEFAULT_WET,
    dry: float = DEFAULT_DRY,
    width: float = DEFAULT_WIDTH,
) -> tuple:
    """Read a WAV file, apply Schroeder reverb, and write the result.

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

    reverb = SchroederReverb(sample_rate, room_size, damping, wet, dry, width)
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
        description="Apply Schroeder reverb to a WAV file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--room-size", type=float, default=DEFAULT_ROOM_SIZE)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--wet", type=float, default=DEFAULT_WET)
    parser.add_argument("--dry", type=float, default=DEFAULT_DRY)
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    apply_reverb(
        args.input,
        args.output,
        args.room_size,
        args.damping,
        args.wet,
        args.dry,
        args.width,
    )
    print(f"Reverb applied: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
