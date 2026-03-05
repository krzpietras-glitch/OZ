#!/usr/bin/env python3
"""Gardner reverb implementation using only numpy and scipy.

Gardner uses nested allpass structures where allpass filters contain other
allpass filters in their delay lines.  This creates efficient high-density
reverberation with a smooth, natural-sounding tail.

Reference: Bill Gardner, "The Virtual Acoustic Room" (1992).
"""

import argparse

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Gardner constants (reference values at 44100 Hz)
# ---------------------------------------------------------------------------
OUTER_ALLPASS_DELAYS = [1051, 337, 1627]
INNER_ALLPASS_DELAYS = [241, 163, 419]
STEREO_SPREAD = 23
ALLPASS_FEEDBACK = 0.5
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


class NestedAllpass:
    """Nested allpass filter with an inner allpass embedded in its delay line.

    The outer allpass reads from its delay buffer, applies a one-pole lowpass
    for damping, then routes the signal through an inner allpass before
    completing the standard allpass computation.
    """

    def __init__(
        self,
        outer_delay: int,
        inner_delay: int,
        feedback: float = ALLPASS_FEEDBACK,
        damping: float = 0.0,
    ):
        self.buffer = np.zeros(outer_delay)
        self.buf_len = outer_delay
        self.index = 0
        self.feedback = feedback
        self.damp1 = damping
        self.damp2 = 1.0 - damping
        self.filterstore = 0.0
        self.inner_ap = AllpassFilter(inner_delay, feedback=feedback)

    def process(self, input_sample: float) -> float:
        buffered = self.buffer[self.index]
        # One-pole lowpass damping in the read path
        self.filterstore = buffered * self.damp2 + self.filterstore * self.damp1
        # Route through inner allpass
        nested_out = self.inner_ap.process(self.filterstore)
        # Standard allpass computation using nested result
        output = nested_out - input_sample
        self.buffer[self.index] = input_sample + nested_out * self.feedback
        self.index = (self.index + 1) % self.buf_len
        return output


# ---------------------------------------------------------------------------
# Gardner reverb processor
# ---------------------------------------------------------------------------
class GardnerReverb:
    """Gardner nested-allpass reverb processor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    room_size : float
        Controls feedback gain in allpass filters (0-1). Higher = longer decay.
    damping : float
        Lowpass damping in allpass feedback path (0-1). Higher = darker tail.
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

        # Left channel: series of nested allpasses
        self.nested_l = [
            NestedAllpass(
                int(od * scale),
                int(id_ * scale),
                feedback=room_size,
                damping=damping,
            )
            for od, id_ in zip(OUTER_ALLPASS_DELAYS, INNER_ALLPASS_DELAYS)
        ]
        # Right channel: offset delays for stereo spread
        self.nested_r = [
            NestedAllpass(
                int(od * scale) + STEREO_SPREAD,
                int(id_ * scale) + STEREO_SPREAD,
                feedback=room_size,
                damping=damping,
            )
            for od, id_ in zip(OUTER_ALLPASS_DELAYS, INNER_ALLPASS_DELAYS)
        ]

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply Gardner reverb to *audio*.

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

        nested_l = self.nested_l
        nested_r = self.nested_r

        for i in range(num_samples):
            inp = (left_in[i] + right_in[i]) * 0.5

            # Series nested allpasses — left channel
            sl = inp
            for nap in nested_l:
                sl = nap.process(sl)

            # Series nested allpasses — right channel
            sr = inp
            for nap in nested_r:
                sr = nap.process(sr)

            out_l[i] = sl
            out_r[i] = sr

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
    """Read a WAV file, apply Gardner reverb, and write the result.

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

    reverb = GardnerReverb(sample_rate, room_size, damping, wet, dry, width)
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
        description="Apply Gardner reverb to a WAV file."
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
