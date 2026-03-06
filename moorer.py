#!/usr/bin/env python3
"""Moorer reverb implementation using only numpy and scipy.

Moorer's reverb improves on Schroeder's design by adding early reflections
via a tapped delay line and replacing plain comb filters with lowpass-feedback
comb filters that simulate frequency-dependent absorption.

Reference: James A. Moorer, "About This Reverberation Business" (1979).
Reference delay lengths are for 44100 Hz and are scaled for other rates.
"""

import argparse

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Moorer constants (reference values at 44100 Hz)
# ---------------------------------------------------------------------------
EARLY_TAP_DELAYS = [190, 949, 993, 1183, 1315, 2021]
EARLY_TAP_GAINS = [0.841, 0.504, 0.491, 0.379, 0.380, 0.346]
COMB_DELAYS = [2205, 2470, 2690, 2999, 3175, 3440]
ALLPASS_DELAYS = [347, 113]
STEREO_SPREAD = 23
ALLPASS_FEEDBACK = 0.7
REFERENCE_SAMPLE_RATE = 44100

# Default user-facing parameters
DEFAULT_ROOM_SIZE = 0.5
DEFAULT_DAMPING = 0.5
DEFAULT_EARLY_REFLECTIONS = 0.5
DEFAULT_WET = 0.33
DEFAULT_DRY = 0.67
DEFAULT_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Filter building blocks
# ---------------------------------------------------------------------------
class TappedDelayLine:
    """Tapped delay line for early reflections.

    Multiple taps at different delays with independent gains simulate
    the first reflections arriving from room surfaces.
    """

    def __init__(self, tap_delays: list[int], tap_gains: list[float], gain: float = 1.0):
        self.max_delay = max(tap_delays)
        self.buf_len = self.max_delay + 1
        self.buffer = np.zeros(self.buf_len)
        self.index = 0
        self.tap_delays = tap_delays
        self.tap_gains = [g * gain for g in tap_gains]

    def process(self, input_sample: float) -> float:
        self.buffer[self.index] = input_sample
        output = 0.0
        for delay, gain in zip(self.tap_delays, self.tap_gains):
            tap_index = (self.index - delay) % self.buf_len
            output += self.buffer[tap_index] * gain
        self.index = (self.index + 1) % self.buf_len
        return output


class LowpassCombFilter:
    """Feedback comb filter with a one-pole lowpass in the feedback path.

    Moorer's improvement over Schroeder's plain comb filter.  The lowpass
    filter in the feedback loop simulates frequency-dependent absorption,
    causing high frequencies to decay faster than low frequencies.
    """

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
# Moorer reverb processor
# ---------------------------------------------------------------------------
class MoorerReverb:
    """Stereo Moorer reverb processor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    room_size : float
        Feedback gain for comb filters (0-1). Higher = longer decay.
    damping : float
        Lowpass damping in comb feedback path (0-1). Higher = darker tail.
    early_reflections : float
        Level of early reflections from the tapped delay line (0-1).
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
        early_reflections: float = DEFAULT_EARLY_REFLECTIONS,
        wet: float = DEFAULT_WET,
        dry: float = DEFAULT_DRY,
        width: float = DEFAULT_WIDTH,
    ):
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.damping = damping
        self.early_reflections = early_reflections
        self.wet = wet
        self.dry = dry
        self.width = width

        scale = sample_rate / REFERENCE_SAMPLE_RATE

        # Early reflections — tapped delay lines (L/R with stereo spread)
        scaled_taps_l = [max(1, int(d * scale)) for d in EARLY_TAP_DELAYS]
        scaled_taps_r = [max(1, int(d * scale) + STEREO_SPREAD) for d in EARLY_TAP_DELAYS]
        self.early_l = TappedDelayLine(scaled_taps_l, EARLY_TAP_GAINS, early_reflections)
        self.early_r = TappedDelayLine(scaled_taps_r, EARLY_TAP_GAINS, early_reflections)

        # Late reverb — parallel lowpass comb filters
        self.combs_l = [
            LowpassCombFilter(int(d * scale), room_size, damping) for d in COMB_DELAYS
        ]
        self.combs_r = [
            LowpassCombFilter(int(d * scale) + STEREO_SPREAD, room_size, damping)
            for d in COMB_DELAYS
        ]

        # Series allpass filters
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

        early_l = self.early_l
        early_r = self.early_r
        combs_l = self.combs_l
        combs_r = self.combs_r
        allpasses_l = self.allpasses_l
        allpasses_r = self.allpasses_r

        for i in range(num_samples):
            inp = (left_in[i] + right_in[i]) * 0.5

            # Early reflections
            er_l = early_l.process(inp)
            er_r = early_r.process(inp)

            # Late reverb: parallel lowpass comb filters
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

            # Combine early reflections with late reverb
            out_l[i] = er_l + cl
            out_r[i] = er_r + cr

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
    early_reflections: float = DEFAULT_EARLY_REFLECTIONS,
    wet: float = DEFAULT_WET,
    dry: float = DEFAULT_DRY,
    width: float = DEFAULT_WIDTH,
) -> tuple:
    """Read a WAV file, apply Moorer reverb, and write the result.

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

    reverb = MoorerReverb(sample_rate, room_size, damping, early_reflections, wet, dry, width)
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
        description="Apply Moorer reverb to a WAV file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--room-size", type=float, default=DEFAULT_ROOM_SIZE)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--early-reflections", type=float, default=DEFAULT_EARLY_REFLECTIONS)
    parser.add_argument("--wet", type=float, default=DEFAULT_WET)
    parser.add_argument("--dry", type=float, default=DEFAULT_DRY)
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    apply_reverb(
        args.input,
        args.output,
        args.room_size,
        args.damping,
        args.early_reflections,
        args.wet,
        args.dry,
        args.width,
    )
    print(f"Reverb applied: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
