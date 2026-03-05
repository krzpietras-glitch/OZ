"""Tests for the Dattorro Plate Reverb implementation."""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from dattorro import AllpassFilter, DattorroPlate, DelayLine, OnePoleFilter, apply_reverb


# ---------------------------------------------------------------------------
# DelayLine tests
# ---------------------------------------------------------------------------
class TestDelayLine:
    def test_delays_signal(self):
        """Output should be delayed by the buffer length."""
        delay = 10
        dl = DelayLine(delay)
        impulse = np.zeros(25)
        impulse[0] = 1.0
        out = np.array([dl.process(s) for s in impulse])

        assert np.all(out[:delay] == 0.0)
        assert out[delay] == 1.0
        assert np.all(out[delay + 1 :] == 0.0)

    def test_tap_reads_correct_position(self):
        """tap(0) returns most recently written; tap(length-1) returns oldest."""
        dl = DelayLine(5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            dl.process(v)
        # buffer is now full: [1, 2, 3, 4, 5], index wrapped to 0
        assert dl.tap(0) == 5.0  # most recent
        assert dl.tap(4) == 1.0  # oldest

    def test_tap_clamped_to_length(self):
        """Tap delay exceeding length should be clamped safely."""
        dl = DelayLine(5)
        dl.process(1.0)
        # Should not raise even when delay > length
        val = dl.tap(100)
        assert isinstance(val, (float, np.floating))


# ---------------------------------------------------------------------------
# OnePoleFilter tests
# ---------------------------------------------------------------------------
class TestOnePoleFilter:
    def test_passthrough_when_g_zero(self):
        """g=0 should pass the input through unmodified."""
        filt = OnePoleFilter(0.0)
        sig = [1.0, 0.5, -0.3, 0.7]
        out = [filt.process(s) for s in sig]
        np.testing.assert_allclose(out, sig)

    def test_heavy_lowpass_attenuates(self):
        """High g should attenuate rapid changes (high frequencies)."""
        filt_lo = OnePoleFilter(0.0)
        filt_hi = OnePoleFilter(0.95)

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, 500)
        out_lo = np.array([filt_lo.process(s) for s in noise])
        out_hi = np.array([filt_hi.process(s) for s in noise])

        # Filtered signal should have lower variance than pass-through
        assert np.var(out_hi) < np.var(out_lo)

    def test_dc_convergence(self):
        """Constant input should converge to the DC value."""
        filt = OnePoleFilter(0.9)
        for _ in range(1000):
            out = filt.process(1.0)
        assert abs(out - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# AllpassFilter tests
# ---------------------------------------------------------------------------
class TestAllpassFilter:
    def test_bounded_energy(self):
        """Output energy should stay within a reasonable range (no blow-up)."""
        ap = AllpassFilter(100, feedback=0.5)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, 10000)
        out = np.array([ap.process(s) for s in noise])

        e_in = np.mean(noise[1000:] ** 2)
        e_out = np.mean(out[1000:] ** 2)
        assert e_out < e_in * 5.0
        assert e_out > e_in * 0.1

    def test_output_differs_from_input(self):
        """The allpass filter should alter the phase / waveform shape."""
        ap = AllpassFilter(50, feedback=0.5)
        rng = np.random.default_rng(7)
        sig = rng.normal(0, 0.3, 500)
        out = np.array([ap.process(s) for s in sig])
        assert not np.allclose(sig, out)

    def test_tap_reads_internal_buffer(self):
        """tap() should read values from the internal delay buffer."""
        ap = AllpassFilter(10, feedback=0.5)
        for v in range(20):
            ap.process(float(v))
        # tap(0) should return a non-trivial value from the internal buffer
        assert ap.tap(0) != 0.0


# ---------------------------------------------------------------------------
# DattorroPlate processor tests
# ---------------------------------------------------------------------------
class TestDattorroPlate:
    def test_mono_produces_stereo(self):
        mono = np.zeros(1000)
        mono[0] = 1.0
        result = DattorroPlate().process(mono)
        assert result.shape == (1000, 2)

    def test_stereo_produces_stereo(self):
        stereo = np.zeros((1000, 2))
        stereo[0, 0] = 1.0
        result = DattorroPlate().process(stereo)
        assert result.shape == (1000, 2)

    def test_dry_only_preserves_input(self):
        """wet=0 → output equals dry input on both channels."""
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.3, 500)
        result = DattorroPlate(wet=0.0, dry=1.0).process(mono)
        np.testing.assert_allclose(result[:, 0], mono, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], mono, atol=1e-12)

    def test_wet_adds_reverb_tail(self):
        """Reverb tail should contain non-trivial energy after the impulse."""
        impulse = np.zeros(10000)
        impulse[0] = 0.8
        result = DattorroPlate(decay=0.8, wet=0.5, dry=0.5).process(impulse)
        tail_energy = np.mean(result[4000:] ** 2)
        assert tail_energy > 1e-6

    def test_silence_in_silence_out(self):
        silence = np.zeros(500)
        result = DattorroPlate().process(silence)
        assert np.allclose(result, 0.0)

    def test_higher_decay_longer_tail(self):
        """Higher decay value should produce a longer reverb tail."""
        impulse = np.zeros(15000)
        impulse[0] = 0.8

        result_lo = DattorroPlate(decay=0.3, wet=1.0, dry=0.0).process(impulse)
        result_hi = DattorroPlate(decay=0.9, wet=1.0, dry=0.0).process(impulse)

        tail = slice(8000, None)
        energy_lo = np.mean(result_lo[tail] ** 2)
        energy_hi = np.mean(result_hi[tail] ** 2)
        assert energy_hi > energy_lo


# ---------------------------------------------------------------------------
# WAV round-trip tests
# ---------------------------------------------------------------------------
class TestWavRoundTrip:
    def _make_sine_wav(self, path, sr=44100, duration=0.25, channels=1):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        if channels == 2:
            sig = np.column_stack([sig, sig])
        wavfile.write(path, sr, sig)

    def test_mono_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "in.wav")
            out = os.path.join(tmp, "out.wav")
            self._make_sine_wav(inp)

            sr, data = apply_reverb(inp, out, decay=0.5, wet=0.3, dry=0.7)
            assert sr == 44100
            assert data.ndim == 2 and data.shape[1] == 2
            assert data.dtype == np.int16
            assert os.path.isfile(out)

    def test_stereo_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "in.wav")
            out = os.path.join(tmp, "out.wav")
            self._make_sine_wav(inp, channels=2)

            sr, data = apply_reverb(inp, out)
            assert sr == 44100
            assert data.shape[1] == 2
            assert data.dtype == np.int16

    def test_output_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "in.wav")
            out = os.path.join(tmp, "out.wav")
            self._make_sine_wav(inp)

            _, data = apply_reverb(inp, out, wet=0.5, dry=0.5)
            assert np.any(data != 0)
