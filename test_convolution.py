"""Tests for the Convolution Reverb implementation."""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from convolution import (
    ConvolutionReverb,
    ImpulseResponse,
    MIN_IR_SECONDS,
    MAX_IR_SECONDS,
    apply_reverb,
)


# ---------------------------------------------------------------------------
# ImpulseResponse tests
# ---------------------------------------------------------------------------
class TestImpulseResponse:
    def test_generates_correct_length(self):
        """Synthetic IR length should match the expected duration."""
        ir = ImpulseResponse(44100, ir_length=0.0, decay=0.5)
        expected_length = int(MIN_IR_SECONDS * 44100)
        assert len(ir.ir_left) == expected_length
        assert len(ir.ir_right) == expected_length

    def test_max_ir_length(self):
        """ir_length=1.0 should produce an IR of MAX_IR_SECONDS duration."""
        ir = ImpulseResponse(44100, ir_length=1.0, decay=0.5)
        expected_length = int(MAX_IR_SECONDS * 44100)
        assert len(ir.ir_left) == expected_length
        assert len(ir.ir_right) == expected_length

    def test_decay_reduces_tail_energy(self):
        """Higher decay should yield less energy in the tail."""
        ir_low = ImpulseResponse(44100, ir_length=0.5, decay=0.1)
        ir_high = ImpulseResponse(44100, ir_length=0.5, decay=0.9)

        # Compare tail energy (last quarter)
        tail_start = len(ir_low.ir_left) * 3 // 4
        energy_low = np.sum(ir_low.ir_left[tail_start:] ** 2)
        energy_high = np.sum(ir_high.ir_left[tail_start:] ** 2)
        assert energy_high < energy_low

    def test_normalized_peak(self):
        """Synthetic IR should be normalized to unit peak."""
        ir = ImpulseResponse(44100, ir_length=0.5, decay=0.5)
        peak = max(np.max(np.abs(ir.ir_left)), np.max(np.abs(ir.ir_right)))
        assert abs(peak - 1.0) < 1e-10

    def test_loads_from_file(self):
        """Loading an IR from a WAV file should produce valid arrays."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ir.wav")
            rng = np.random.default_rng(0)
            ir_data = (rng.normal(0, 0.3, 1000) * 16000).astype(np.int16)
            wavfile.write(path, 44100, ir_data)

            ir = ImpulseResponse(44100, ir_file=path)
            assert len(ir.ir_left) == 1000
            assert len(ir.ir_right) == 1000


# ---------------------------------------------------------------------------
# ConvolutionReverb processor tests
# ---------------------------------------------------------------------------
class TestConvolutionReverb:
    def test_mono_produces_stereo(self):
        mono = np.zeros(1000)
        mono[0] = 1.0
        result = ConvolutionReverb(ir_length=0.0).process(mono)
        assert result.shape == (1000, 2)

    def test_stereo_produces_stereo(self):
        stereo = np.zeros((1000, 2))
        stereo[0, 0] = 1.0
        result = ConvolutionReverb(ir_length=0.0).process(stereo)
        assert result.shape == (1000, 2)

    def test_dry_only_preserves_input(self):
        """wet=0 → output equals dry input on both channels."""
        rng = np.random.default_rng(42)
        mono = rng.normal(0, 0.3, 500)
        result = ConvolutionReverb(wet=0.0, dry=1.0, ir_length=0.0).process(mono)
        np.testing.assert_allclose(result[:, 0], mono, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], mono, atol=1e-12)

    def test_wet_adds_reverb_tail(self):
        """Reverb tail should contain non-trivial energy after the impulse."""
        impulse = np.zeros(5000)
        impulse[0] = 0.8
        result = ConvolutionReverb(
            ir_length=0.2, decay=0.3, wet=0.5, dry=0.5
        ).process(impulse)
        tail_energy = np.mean(result[2000:] ** 2)
        assert tail_energy > 1e-6

    def test_silence_in_silence_out(self):
        silence = np.zeros(500)
        result = ConvolutionReverb(ir_length=0.0).process(silence)
        assert np.allclose(result, 0.0)

    def test_delta_ir_preserves_signal_shape(self):
        """An IR that is a unit impulse should preserve the input signal."""
        with tempfile.TemporaryDirectory() as tmp:
            ir_path = os.path.join(tmp, "delta.wav")
            delta = np.zeros(100, dtype=np.float32)
            delta[0] = 1.0
            wavfile.write(ir_path, 44100, delta)

            rng = np.random.default_rng(99)
            mono = rng.normal(0, 0.3, 1000)
            reverb = ConvolutionReverb(wet=1.0, dry=0.0, width=1.0, ir_file=ir_path)
            result = reverb.process(mono)
            # With a delta IR, the convolution output ≈ input
            np.testing.assert_allclose(result[:, 0], mono, atol=1e-6)
            np.testing.assert_allclose(result[:, 1], mono, atol=1e-6)


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

            sr, data = apply_reverb(inp, out, ir_length=0.1, wet=0.3, dry=0.7)
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
