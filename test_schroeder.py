"""Tests for the Schroeder reverb implementation."""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from schroeder import AllpassFilter, CombFilter, SchroederReverb, apply_reverb


# ---------------------------------------------------------------------------
# CombFilter tests
# ---------------------------------------------------------------------------
class TestCombFilter:
    def test_delays_signal(self):
        """With zero feedback the comb filter acts as a pure delay."""
        delay = 10
        comb = CombFilter(delay, feedback=0.0, damping=0.0)
        impulse = np.zeros(25)
        impulse[0] = 1.0
        out = np.array([comb.process(s) for s in impulse])

        assert np.all(out[:delay] == 0.0)
        assert out[delay] == 1.0
        # No feedback → silence after the single echo
        assert np.all(out[delay + 1 :] == 0.0)

    def test_feedback_produces_decaying_echoes(self):
        """Non-zero feedback should create repeated, decaying echoes."""
        delay = 10
        comb = CombFilter(delay, feedback=0.5, damping=0.0)
        impulse = np.zeros(50)
        impulse[0] = 1.0
        out = np.array([comb.process(s) for s in impulse])

        assert out[delay] == 1.0
        assert abs(out[2 * delay]) > 0
        assert abs(out[2 * delay]) < abs(out[delay])

    def test_damping_attenuates_highs(self):
        """Higher damping should reduce the energy in the feedback path."""
        delay = 10
        comb_no_damp = CombFilter(delay, feedback=0.7, damping=0.0)
        comb_hi_damp = CombFilter(delay, feedback=0.7, damping=0.9)

        impulse = np.zeros(60)
        impulse[0] = 1.0
        out_no = np.array([comb_no_damp.process(s) for s in impulse])
        out_hi = np.array([comb_hi_damp.process(s) for s in impulse])

        # Tail energy should be lower with heavy damping
        tail = slice(30, None)
        assert np.sum(out_hi[tail] ** 2) < np.sum(out_no[tail] ** 2)


# ---------------------------------------------------------------------------
# AllpassFilter tests
# ---------------------------------------------------------------------------
class TestAllpassFilter:
    def test_bounded_energy(self):
        """Output energy should stay within a reasonable range (no blow-up)."""
        ap = AllpassFilter(100, feedback=0.7)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, 10000)
        out = np.array([ap.process(s) for s in noise])

        e_in = np.mean(noise[1000:] ** 2)
        e_out = np.mean(out[1000:] ** 2)
        # Allpass should not blow up; allow up to 5x energy
        assert e_out < e_in * 5.0
        assert e_out > e_in * 0.1

    def test_output_differs_from_input(self):
        """The allpass filter should alter the phase / waveform shape."""
        ap = AllpassFilter(50, feedback=0.7)
        rng = np.random.default_rng(7)
        sig = rng.normal(0, 0.3, 500)
        out = np.array([ap.process(s) for s in sig])
        assert not np.allclose(sig, out)


# ---------------------------------------------------------------------------
# SchroederReverb processor tests
# ---------------------------------------------------------------------------
class TestSchroederReverb:
    def test_mono_produces_stereo(self):
        mono = np.zeros(1000)
        mono[0] = 1.0
        result = SchroederReverb().process(mono)
        assert result.shape == (1000, 2)

    def test_stereo_produces_stereo(self):
        stereo = np.zeros((1000, 2))
        stereo[0, 0] = 1.0
        result = SchroederReverb().process(stereo)
        assert result.shape == (1000, 2)

    def test_dry_only_preserves_input(self):
        """wet=0 → output equals dry input on both channels."""
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.3, 500)
        result = SchroederReverb(wet=0.0, dry=1.0).process(mono)
        np.testing.assert_allclose(result[:, 0], mono, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], mono, atol=1e-12)

    def test_wet_adds_reverb_tail(self):
        """Reverb tail should contain non-trivial energy after the impulse."""
        impulse = np.zeros(5000)
        impulse[0] = 0.8
        result = SchroederReverb(room_size=0.8, wet=0.5, dry=0.5).process(impulse)
        tail_energy = np.mean(result[2000:] ** 2)
        assert tail_energy > 1e-6

    def test_silence_in_silence_out(self):
        silence = np.zeros(500)
        result = SchroederReverb().process(silence)
        assert np.allclose(result, 0.0)

    def test_stereo_width_zero_gives_mono_reverb(self):
        """With width=0 both channels of the wet signal should be identical."""
        impulse = np.zeros(3000)
        impulse[0] = 0.5
        result = SchroederReverb(wet=1.0, dry=0.0, width=0.0).process(impulse)
        np.testing.assert_allclose(result[:, 0], result[:, 1], atol=1e-12)

    def test_fewer_filters_than_freeverb(self):
        """Schroeder uses 4 combs and 2 allpasses (fewer than Freeverb's 8+4)."""
        reverb = SchroederReverb()
        assert len(reverb.combs_l) == 4
        assert len(reverb.allpasses_l) == 2


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
