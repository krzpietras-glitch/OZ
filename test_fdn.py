"""Tests for the FDN reverb implementation."""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from fdn import (
    DampingFilter,
    DelayLine,
    FDNReverb,
    InputDiffuser,
    apply_reverb,
)


# ---------------------------------------------------------------------------
# DelayLine tests
# ---------------------------------------------------------------------------
class TestDelayLine:
    def test_delays_signal(self):
        """An impulse should appear at the output after exactly delay_length samples."""
        delay = 10
        dl = DelayLine(delay)
        impulse = np.zeros(25)
        impulse[0] = 1.0
        out = np.array([dl.process(s) for s in impulse])

        assert np.all(out[:delay] == 0.0)
        assert out[delay] == 1.0
        assert np.all(out[delay + 1 :] == 0.0)

    def test_preserves_order(self):
        """Samples should come out in the same order they went in."""
        delay = 5
        dl = DelayLine(delay)
        sig = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        out = np.array([dl.process(s) for s in sig])
        assert out[delay] == 1.0
        assert out[delay + 1] == 2.0
        assert out[delay + 2] == 3.0


# ---------------------------------------------------------------------------
# DampingFilter tests
# ---------------------------------------------------------------------------
class TestDampingFilter:
    def test_zero_damping_passes_signal(self):
        """With damping=0 the filter should pass the input directly."""
        df = DampingFilter(damping=0.0)
        sig = [1.0, 0.0, 0.0, 0.0, 0.0]
        out = [df.process(s) for s in sig]
        assert out[0] == 1.0
        assert all(o == 0.0 for o in out[1:])

    def test_high_damping_smoothes(self):
        """Higher damping should spread the impulse energy over more samples."""
        df_lo = DampingFilter(damping=0.0)
        df_hi = DampingFilter(damping=0.9)
        impulse = np.zeros(50)
        impulse[0] = 1.0
        out_lo = np.array([df_lo.process(s) for s in impulse])
        out_hi = np.array([df_hi.process(s) for s in impulse])

        # High damping spreads the energy over more samples (smoothing)
        assert np.sum(out_hi[1:] ** 2) > np.sum(out_lo[1:] ** 2)


# ---------------------------------------------------------------------------
# InputDiffuser tests
# ---------------------------------------------------------------------------
class TestInputDiffuser:
    def test_bounded_energy(self):
        """Output energy should stay within a reasonable range."""
        diff = InputDiffuser(100, feedback=0.5)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, 10000)
        out = np.array([diff.process(s) for s in noise])

        e_in = np.mean(noise[1000:] ** 2)
        e_out = np.mean(out[1000:] ** 2)
        assert e_out < e_in * 5.0
        assert e_out > e_in * 0.1

    def test_output_differs_from_input(self):
        """The diffuser should alter the phase / waveform shape."""
        diff = InputDiffuser(50, feedback=0.5)
        rng = np.random.default_rng(7)
        sig = rng.normal(0, 0.3, 500)
        out = np.array([diff.process(s) for s in sig])
        assert not np.allclose(sig, out)


# ---------------------------------------------------------------------------
# FDNReverb processor tests
# ---------------------------------------------------------------------------
class TestFDNReverb:
    def test_mono_produces_stereo(self):
        mono = np.zeros(1000)
        mono[0] = 1.0
        result = FDNReverb().process(mono)
        assert result.shape == (1000, 2)

    def test_stereo_produces_stereo(self):
        stereo = np.zeros((1000, 2))
        stereo[0, 0] = 1.0
        result = FDNReverb().process(stereo)
        assert result.shape == (1000, 2)

    def test_dry_only_preserves_input(self):
        """wet=0 -> output equals dry input on both channels."""
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.3, 500)
        result = FDNReverb(wet=0.0, dry=1.0).process(mono)
        np.testing.assert_allclose(result[:, 0], mono, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], mono, atol=1e-12)

    def test_wet_adds_reverb_tail(self):
        """Reverb tail should contain non-trivial energy after the impulse."""
        impulse = np.zeros(5000)
        impulse[0] = 0.8
        result = FDNReverb(decay=0.8, wet=0.5, dry=0.5).process(impulse)
        tail_energy = np.mean(result[2000:] ** 2)
        assert tail_energy > 1e-6

    def test_silence_in_silence_out(self):
        silence = np.zeros(500)
        result = FDNReverb().process(silence)
        assert np.allclose(result, 0.0)

    def test_higher_decay_longer_tail(self):
        """A higher decay value should produce more energy in the late tail."""
        impulse = np.zeros(8000)
        impulse[0] = 0.8
        result_lo = FDNReverb(decay=0.3, wet=1.0, dry=0.0).process(impulse)
        result_hi = FDNReverb(decay=0.8, wet=1.0, dry=0.0).process(impulse)
        tail = slice(4000, None)
        assert np.mean(result_hi[tail] ** 2) > np.mean(result_lo[tail] ** 2)


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

            sr, data = apply_reverb(inp, out, room_size=0.5, wet=0.3, dry=0.7)
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
