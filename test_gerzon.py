"""Tests for the Gerzon Reverb implementation."""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from gerzon import DampingFilter, ModulatedDelayLine, GerzonReverb, apply_reverb


# ---------------------------------------------------------------------------
# DampingFilter tests
# ---------------------------------------------------------------------------
class TestDampingFilter:
    def test_passthrough_at_zero_damping(self):
        """With damping=0 the filter passes the input through unchanged."""
        df = DampingFilter(damping=0.0)
        sig = [0.5, -0.3, 0.8, 0.0, -0.2]
        out = [df.process(s) for s in sig]
        np.testing.assert_allclose(out, sig)

    def test_high_damping_smooths_more(self):
        """Higher damping should reduce output energy for noisy input."""
        df_lo = DampingFilter(damping=0.2)
        df_hi = DampingFilter(damping=0.9)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, 500)
        out_lo = np.array([df_lo.process(s) for s in noise])
        out_hi = np.array([df_hi.process(s) for s in noise])
        # Heavier smoothing → lower energy
        assert np.mean(out_hi ** 2) < np.mean(out_lo ** 2)


# ---------------------------------------------------------------------------
# ModulatedDelayLine tests
# ---------------------------------------------------------------------------
class TestModulatedDelayLine:
    def test_basic_delay(self):
        """With zero modulation the delay line acts as a pure delay."""
        delay = 10
        dl = ModulatedDelayLine(delay_length=delay, max_mod_samples=0)
        impulse = np.zeros(25)
        impulse[0] = 1.0
        out = []
        for s in impulse:
            out.append(dl.read(0.0))
            dl.write(s)
        out = np.array(out)

        assert np.all(out[:delay] == 0.0)
        assert out[delay] == 1.0
        assert np.all(out[delay + 1 :] == 0.0)

    def test_modulation_shifts_read_position(self):
        """Positive modulation should increase the effective delay."""
        dl_a = ModulatedDelayLine(delay_length=50, max_mod_samples=10)
        dl_b = ModulatedDelayLine(delay_length=50, max_mod_samples=10)
        impulse = np.zeros(100)
        impulse[0] = 1.0

        out_a, out_b = [], []
        for s in impulse:
            out_a.append(dl_a.read(0.0))
            dl_a.write(s)
            out_b.append(dl_b.read(0.5))  # positive modulation → longer delay
            dl_b.write(s)
        out_a = np.array(out_a)
        out_b = np.array(out_b)

        peak_a = np.argmax(np.abs(out_a))
        peak_b = np.argmax(np.abs(out_b))
        assert peak_b > peak_a


# ---------------------------------------------------------------------------
# GerzonReverb processor tests
# ---------------------------------------------------------------------------
class TestGerzonReverb:
    def test_mono_produces_stereo(self):
        mono = np.zeros(1000)
        mono[0] = 1.0
        result = GerzonReverb().process(mono)
        assert result.shape == (1000, 2)

    def test_stereo_produces_stereo(self):
        stereo = np.zeros((1000, 2))
        stereo[0, 0] = 1.0
        result = GerzonReverb().process(stereo)
        assert result.shape == (1000, 2)

    def test_dry_only_preserves_input(self):
        """wet=0 → output equals dry input on both channels."""
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.3, 500)
        result = GerzonReverb(wet=0.0, dry=1.0).process(mono)
        np.testing.assert_allclose(result[:, 0], mono, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], mono, atol=1e-12)

    def test_wet_adds_reverb_tail(self):
        """Reverb tail should contain non-trivial energy after the impulse."""
        impulse = np.zeros(5000)
        impulse[0] = 0.8
        result = GerzonReverb(room_size=0.8, wet=0.5, dry=0.5).process(impulse)
        tail_energy = np.mean(result[2000:] ** 2)
        assert tail_energy > 1e-6

    def test_silence_in_silence_out(self):
        silence = np.zeros(500)
        result = GerzonReverb().process(silence)
        assert np.allclose(result, 0.0)

    def test_modulation_changes_output(self):
        """Non-zero modulation depth should alter the reverb output."""
        impulse = np.zeros(3000)
        impulse[0] = 0.5
        r_no_mod = GerzonReverb(
            modulation_depth=0.0, wet=1.0, dry=0.0
        ).process(impulse)
        r_mod = GerzonReverb(
            modulation_depth=0.5, modulation_rate=0.8, wet=1.0, dry=0.0
        ).process(impulse)
        assert not np.allclose(r_no_mod, r_mod, atol=1e-10)


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
