#!/usr/bin/env python3
"""Dattorro Plate Reverb implementation using only numpy and scipy.

Jon Dattorro, "Effect Design Part 1: Reverberator and Other Filters",
Journal of the Audio Engineering Society, 1997.

Topology: input → pre-delay → bandwidth filter → 4 input diffusers
(series allpass) → tank with two cross-coupled decay paths, each having
decay diffuser → delay → damping → decay diffuser → delay.  Stereo output
is derived from multiple tap points within the tank.

Reference delay lengths are for 44100 Hz and are scaled for other rates.
"""

import argparse

import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Dattorro constants (delay lengths at 44100 Hz, scaled from 29761 Hz)
# ---------------------------------------------------------------------------
REFERENCE_SAMPLE_RATE = 44100

# Input diffusion allpass delays
INPUT_DIFFUSER_DELAYS = [210, 159, 562, 411]

# Tank delays — left half (path A)
DECAY_DIFFUSER_1A_DELAY = 996
DELAY_LINE_1A_LENGTH = 6599
DECAY_DIFFUSER_2A_DELAY = 2668
DELAY_LINE_2A_LENGTH = 5514

# Tank delays — right half (path B)
DECAY_DIFFUSER_1B_DELAY = 1346
DELAY_LINE_1B_LENGTH = 6250
DECAY_DIFFUSER_2B_DELAY = 3937
DELAY_LINE_2B_LENGTH = 4688

# Pre-delay
MAX_PRE_DELAY_SAMPLES = 4410  # ~100 ms at 44100 Hz

# Output tap positions at 44100 Hz (scaled from 29761 Hz paper values)
# Left output taps
TAP_LEFT_DL1B_A = 394
TAP_LEFT_DL1B_B = 4408
TAP_LEFT_DD2B = 2835
TAP_LEFT_DL2B = 2958
TAP_LEFT_DL1A = 1580
TAP_LEFT_DD2A = 277

# Right output taps
TAP_RIGHT_DL1A_A = 523
TAP_RIGHT_DL1A_B = 5375
TAP_RIGHT_DD2A = 1820
TAP_RIGHT_DL2A = 3962
TAP_RIGHT_DL1B = 3129
TAP_RIGHT_DD2B = 496

# Internal diffusion coefficients
DECAY_DIFFUSION_1 = 0.7
INPUT_DIFFUSION_RATIO = 0.833  # ratio between diff_2 and diff_1

# Default user-facing parameters
DEFAULT_PRE_DELAY = 0.0
DEFAULT_BANDWIDTH = 0.7
DEFAULT_DAMPING = 0.5
DEFAULT_DECAY = 0.5
DEFAULT_INPUT_DIFFUSION = 0.75
DEFAULT_WET = 0.3
DEFAULT_DRY = 0.6
DEFAULT_STEREO_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Filter building blocks
# ---------------------------------------------------------------------------
class DelayLine:
    """Simple delay line with multi-tap support."""

    def __init__(self, length: int):
        self.buffer = np.zeros(max(length, 1))
        self.length = max(length, 1)
        self.index = 0

    def process(self, sample: float) -> float:
        """Write *sample*, return the oldest sample in the buffer."""
        out = self.buffer[self.index]
        self.buffer[self.index] = sample
        self.index = (self.index + 1) % self.length
        return out

    def tap(self, delay: int) -> float:
        """Read *delay* samples behind the write head (0 = most recent)."""
        delay = min(delay, self.length - 1)
        idx = (self.index - 1 - delay) % self.length
        return self.buffer[idx]


class OnePoleFilter:
    """One-pole lowpass: y[n] = (1-g)*x[n] + g*y[n-1].

    g = 0 → pass-through;  g → 1 → heavy lowpass.
    """

    def __init__(self, g: float):
        self.g = g
        self.state = 0.0

    def process(self, sample: float) -> float:
        self.state = (1.0 - self.g) * sample + self.g * self.state
        return self.state


class AllpassFilter:
    """Schroeder allpass with tappable internal delay buffer."""

    def __init__(self, delay_length: int, feedback: float):
        self.buffer = np.zeros(max(delay_length, 1))
        self.buf_len = max(delay_length, 1)
        self.index = 0
        self.feedback = feedback

    def process(self, input_sample: float) -> float:
        buffered = self.buffer[self.index]
        output = buffered - input_sample
        self.buffer[self.index] = input_sample + buffered * self.feedback
        self.index = (self.index + 1) % self.buf_len
        return output

    def tap(self, delay: int) -> float:
        """Read from internal buffer *delay* steps behind write head."""
        delay = min(delay, self.buf_len - 1)
        idx = (self.index - 1 - delay) % self.buf_len
        return self.buffer[idx]


# ---------------------------------------------------------------------------
# Dattorro Plate Reverb processor
# ---------------------------------------------------------------------------
class DattorroPlate:
    """Dattorro plate reverb processor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    pre_delay : float
        Pre-delay amount (0-1), mapped to 0–100 ms.
    bandwidth : float
        Input lowpass cutoff (0-1). Higher = brighter input.
    damping : float
        Tank damping (0-1). Higher = darker reverb tail.
    decay : float
        Tank decay factor (0-1). Higher = longer reverb.
    input_diffusion : float
        Diffusion amount for input allpass chain (0-1).
    wet : float
        Wet (reverb) signal level.
    dry : float
        Dry (original) signal level.
    stereo_width : float
        Stereo width of the reverb output (0-1).
    """

    def __init__(
        self,
        sample_rate: int = REFERENCE_SAMPLE_RATE,
        pre_delay: float = DEFAULT_PRE_DELAY,
        bandwidth: float = DEFAULT_BANDWIDTH,
        damping: float = DEFAULT_DAMPING,
        decay: float = DEFAULT_DECAY,
        input_diffusion: float = DEFAULT_INPUT_DIFFUSION,
        wet: float = DEFAULT_WET,
        dry: float = DEFAULT_DRY,
        stereo_width: float = DEFAULT_STEREO_WIDTH,
    ):
        self.sample_rate = sample_rate
        self.wet = wet
        self.dry = dry
        self.stereo_width = stereo_width
        self.decay = decay

        scale = sample_rate / REFERENCE_SAMPLE_RATE

        # Pre-delay
        predelay_len = max(int(pre_delay * MAX_PRE_DELAY_SAMPLES * scale), 1)
        self.predelay = DelayLine(predelay_len)

        # Bandwidth filter (one-pole lowpass, g = 1 - bandwidth)
        self.bandwidth_filter = OnePoleFilter(1.0 - bandwidth)

        # Input diffusers (4 series allpass filters)
        diff1 = input_diffusion
        diff2 = input_diffusion * INPUT_DIFFUSION_RATIO
        delays = INPUT_DIFFUSER_DELAYS
        self.input_diffusers = [
            AllpassFilter(int(delays[0] * scale), diff1),
            AllpassFilter(int(delays[1] * scale), diff1),
            AllpassFilter(int(delays[2] * scale), diff2),
            AllpassFilter(int(delays[3] * scale), diff2),
        ]

        # Decay diffusion coefficients
        decay_diff_2 = min(max(decay * 0.5 + 0.15, 0.25), 0.5)

        # Tank — left half (path A)
        self.dd_1a = AllpassFilter(
            int(DECAY_DIFFUSER_1A_DELAY * scale), -DECAY_DIFFUSION_1
        )
        self.dl_1a = DelayLine(int(DELAY_LINE_1A_LENGTH * scale))
        self.damp_a = OnePoleFilter(damping)
        self.dd_2a = AllpassFilter(
            int(DECAY_DIFFUSER_2A_DELAY * scale), decay_diff_2
        )
        self.dl_2a = DelayLine(int(DELAY_LINE_2A_LENGTH * scale))

        # Tank — right half (path B)
        self.dd_1b = AllpassFilter(
            int(DECAY_DIFFUSER_1B_DELAY * scale), -DECAY_DIFFUSION_1
        )
        self.dl_1b = DelayLine(int(DELAY_LINE_1B_LENGTH * scale))
        self.damp_b = OnePoleFilter(damping)
        self.dd_2b = AllpassFilter(
            int(DECAY_DIFFUSER_2B_DELAY * scale), decay_diff_2
        )
        self.dl_2b = DelayLine(int(DELAY_LINE_2B_LENGTH * scale))

        # Scale tap positions (clamped to buffer lengths in tap())
        self._tap_l_dl1b_a = int(TAP_LEFT_DL1B_A * scale)
        self._tap_l_dl1b_b = int(TAP_LEFT_DL1B_B * scale)
        self._tap_l_dd2b = int(TAP_LEFT_DD2B * scale)
        self._tap_l_dl2b = int(TAP_LEFT_DL2B * scale)
        self._tap_l_dl1a = int(TAP_LEFT_DL1A * scale)
        self._tap_l_dd2a = int(TAP_LEFT_DD2A * scale)

        self._tap_r_dl1a_a = int(TAP_RIGHT_DL1A_A * scale)
        self._tap_r_dl1a_b = int(TAP_RIGHT_DL1A_B * scale)
        self._tap_r_dd2a = int(TAP_RIGHT_DD2A * scale)
        self._tap_r_dl2a = int(TAP_RIGHT_DL2A * scale)
        self._tap_r_dl1b = int(TAP_RIGHT_DL1B * scale)
        self._tap_r_dd2b = int(TAP_RIGHT_DD2B * scale)

        # Tank state (cross-coupling feedback)
        self._left_tank_out = 0.0
        self._right_tank_out = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply Dattorro plate reverb to *audio*.

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

        # Local references for inner-loop speed
        predelay = self.predelay
        bw_filter = self.bandwidth_filter
        input_diffs = self.input_diffusers
        dd_1a, dl_1a, damp_a = self.dd_1a, self.dl_1a, self.damp_a
        dd_2a, dl_2a = self.dd_2a, self.dl_2a
        dd_1b, dl_1b, damp_b = self.dd_1b, self.dl_1b, self.damp_b
        dd_2b, dl_2b = self.dd_2b, self.dl_2b
        decay = self.decay

        for i in range(num_samples):
            inp = (left_in[i] + right_in[i]) * 0.5

            # Pre-delay
            inp = predelay.process(inp)

            # Bandwidth filter
            inp = bw_filter.process(inp)

            # Input diffusers (series allpass chain)
            for d in input_diffs:
                inp = d.process(inp)

            # Save previous tank outputs for cross-coupling
            prev_left = self._left_tank_out
            prev_right = self._right_tank_out

            # Tank — left half (path A)
            la = inp + decay * prev_right
            la = dd_1a.process(la)
            la = dl_1a.process(la)
            la = damp_a.process(la)
            la *= decay
            la = dd_2a.process(la)
            self._left_tank_out = dl_2a.process(la)

            # Tank — right half (path B)
            rb = inp + decay * prev_left
            rb = dd_1b.process(rb)
            rb = dl_1b.process(rb)
            rb = damp_b.process(rb)
            rb *= decay
            rb = dd_2b.process(rb)
            self._right_tank_out = dl_2b.process(rb)

            # Output taps — left (from right half + left half cross-taps)
            out_l[i] = (
                dl_1b.tap(self._tap_l_dl1b_a)
                + dl_1b.tap(self._tap_l_dl1b_b)
                - dd_2b.tap(self._tap_l_dd2b)
                + dl_2b.tap(self._tap_l_dl2b)
                - dl_1a.tap(self._tap_l_dl1a)
                - dd_2a.tap(self._tap_l_dd2a)
            )

            # Output taps — right (from left half + right half cross-taps)
            out_r[i] = (
                dl_1a.tap(self._tap_r_dl1a_a)
                + dl_1a.tap(self._tap_r_dl1a_b)
                - dd_2a.tap(self._tap_r_dd2a)
                + dl_2a.tap(self._tap_r_dl2a)
                - dl_1b.tap(self._tap_r_dl1b)
                - dd_2b.tap(self._tap_r_dd2b)
            )

        # Stereo width and wet/dry mix
        wet1 = self.wet * (self.stereo_width / 2.0 + 0.5)
        wet2 = self.wet * ((1.0 - self.stereo_width) / 2.0)

        result_l = out_l * wet1 + out_r * wet2 + left_in * self.dry
        result_r = out_r * wet1 + out_l * wet2 + right_in * self.dry

        return np.column_stack([result_l, result_r])


# ---------------------------------------------------------------------------
# WAV file helper
# ---------------------------------------------------------------------------
def apply_reverb(
    input_path: str,
    output_path: str,
    pre_delay: float = DEFAULT_PRE_DELAY,
    bandwidth: float = DEFAULT_BANDWIDTH,
    damping: float = DEFAULT_DAMPING,
    decay: float = DEFAULT_DECAY,
    input_diffusion: float = DEFAULT_INPUT_DIFFUSION,
    wet: float = DEFAULT_WET,
    dry: float = DEFAULT_DRY,
    stereo_width: float = DEFAULT_STEREO_WIDTH,
) -> tuple:
    """Read a WAV file, apply Dattorro plate reverb, and write the result.

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

    reverb = DattorroPlate(
        sample_rate, pre_delay, bandwidth, damping, decay,
        input_diffusion, wet, dry, stereo_width,
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
        description="Apply Dattorro plate reverb to a WAV file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--pre-delay", type=float, default=DEFAULT_PRE_DELAY)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--input-diffusion", type=float, default=DEFAULT_INPUT_DIFFUSION)
    parser.add_argument("--wet", type=float, default=DEFAULT_WET)
    parser.add_argument("--dry", type=float, default=DEFAULT_DRY)
    parser.add_argument("--stereo-width", type=float, default=DEFAULT_STEREO_WIDTH)
    args = parser.parse_args()

    apply_reverb(
        args.input,
        args.output,
        args.pre_delay,
        args.bandwidth,
        args.damping,
        args.decay,
        args.input_diffusion,
        args.wet,
        args.dry,
        args.stereo_width,
    )
    print(f"Reverb applied: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
