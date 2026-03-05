"""Tests for the Gardner reverb implementation."""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from gardner import AllpassFilter, NestedAllpass, GardnerReverb, apply_reverb


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
        # Allow up to 5x energy (allpass is approximately unit-gain)
        assert e_out < e_in * 5.0
        assert e_out > e_in * 0.1

    def test_output_differs_from_input(self):
        """The allpass filter should alter the phase / waveform shape."""
        ap = AllpassFilter(50, feedback=0.5)
        rng = np.random.default_rng(7)
        sig = rng.normal(0, 0.3, 500)
        out = np.array([ap.process(s) for s in sig])
        assert not np.allclose(sig, out)


# ---------------------------------------------------------------------------
# NestedAllpass tests
# ---------------------------------------------------------------------------
class TestNestedAllpass:
    def test_delays_signal(self):
        """Nested allpass should produce delayed energy after the initial sample."""
        nap = NestedAllpass(100, 30, feedback=0.5, damping=0.0)
        impulse = np.zeros(300)
        impulse[0] = 1.0
        out = np.array([nap.process(s) for s in impulse])
        # Should have energy past the outer delay length
        assert np.sum(out[50:] ** 2) > 0

    def test_feedback_decay(self):
        """Non-zero feedback should produce decaying energy over time."""
        nap = NestedAllpass(50, 20, feedback=0.5, damping=0.0)
        impulse = np.zeros(500)
        impulse[0] = 1.0
        out = np.array([nap.process(s) for s in impulse])
        # Early energy should exceed late energy
        early_energy = np.sum(out[:200] ** 2)
        late_energy = np.sum(out[200:] ** 2)
        assert early_energy > late_energy

    def test_damping_reduces_tail_energy(self):
        """Higher damping should reduce energy in the reverb tail."""
        impulse = np.zeros(500)
        impulse[0] = 1.0

        nap_lo = NestedAllpass(80, 30, feedback=0.6, damping=0.0)
        nap_hi = NestedAllpass(80, 30, feedback=0.6, damping=0.9)

        out_lo = np.array([nap_lo.process(s) for s in impulse])
        out_hi = np.array([nap_hi.process(s) for s in impulse])

        tail = slice(200, None)
        assert np.sum(out_hi[tail] ** 2) < np.sum(out_lo[tail] ** 2)


# ---------------------------------------------------------------------------
# GardnerReverb processor tests
# ---------------------------------------------------------------------------
class TestGardnerReverb:
    def test_mono_produces_stereo(self):
        mono = np.zeros(1000)
        mono[0] = 1.0
        result = GardnerReverb().process(mono)
        assert result.shape == (1000, 2)

    def test_stereo_produces_stereo(self):
        stereo = np.zeros((1000, 2))
        stereo[0, 0] = 1.0
        result = GardnerReverb().process(stereo)
        assert result.shape == (1000, 2)

    def test_dry_only_preserves_input(self):
        """wet=0 → output equals dry input on both channels."""
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.3, 500)
        result = GardnerReverb(wet=0.0, dry=1.0).process(mono)
        np.testing.assert_allclose(result[:, 0], mono, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], mono, atol=1e-12)

    def test_wet_adds_reverb_tail(self):
        """Reverb tail should contain non-trivial energy after the impulse."""
        impulse = np.zeros(5000)
        impulse[0] = 0.8
        result = GardnerReverb(room_size=0.8, wet=0.5, dry=0.5).process(impulse)
        tail_energy = np.mean(result[2000:] ** 2)
        assert tail_energy > 1e-6

    def test_silence_in_silence_out(self):
        silence = np.zeros(500)
        result = GardnerReverb().process(silence)
        assert np.allclose(result, 0.0)

    def test_nested_structure_high_echo_density(self):
        """Gardner's nested allpasses should produce dense reflections."""
        impulse = np.zeros(3000)
        impulse[0] = 1.0
        result = GardnerReverb(room_size=0.7, wet=1.0, dry=0.0).process(impulse)
        # Count samples with significant energy in the tail
        tail = result[500:, 0]
        significant = np.abs(tail) > 1e-4
        # Nested structure should produce many significant samples
        assert np.sum(significant) > 100


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
