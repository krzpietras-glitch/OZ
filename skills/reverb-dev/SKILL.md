---
name: reverb-dev
description: >
  Guide for implementing reverb algorithms in the OZ audio project.
  Use when building new reverb processors (Dattorro, Gardner, Gerzon, Schroeder, Moorer, FDN, Convolution).
  Defines the required file structure, class patterns, GUI layout, test patterns, and coding conventions
  that every reverb implementation must follow.
---

# Reverb Development Guide

All reverb implementations in this project follow the same architecture established by `freeverb.py`. Read `references/freeverb_example.py` for the concrete reference implementation.

## Required Files Per Reverb

For a reverb named `<name>` (e.g. `dattorro`, `gardner`, `gerzon`):

```
<name>.py          # Processor + CLI
<name>_gui.py      # Tkinter GUI
test_<name>.py     # Pytest tests
```

## Processor File (`<name>.py`)

Follow this structure exactly:

1. **Module docstring** — algorithm name, brief description, academic reference
2. **Imports** — only `argparse`, `numpy`, `scipy.io.wavfile`; no external DSP libraries
3. **Constants** — reference delay lengths at 44100 Hz, default parameter values (all floats 0-1 where possible)
4. **Filter building-block classes** — small classes with `__init__` + `process(sample) -> float` method, sample-by-sample
5. **Main processor class** — named after the algorithm (e.g. `DattorroPlate`, `GardnerReverb`, `GerzonReverb`):
   - `__init__(self, sample_rate, **params)` — scale delays to sample rate, instantiate filters
   - `process(self, audio: np.ndarray) -> np.ndarray` — accepts mono `(N,)` or stereo `(N,2)`, always returns stereo `(N,2)` float64
   - Wet/dry mixing and stereo width applied at the end of `process()`
6. **`apply_reverb()` function** — reads WAV, normalizes to float64 [-1,1], runs processor, clips, writes WAV. Returns `(sample_rate, output_array)`. Must handle int16, int32, float32, float64 WAV formats.
7. **CLI via `argparse`** — `input` and `output` positional args, `--param` optional args for each parameter with defaults

### Key conventions
- All delay lengths defined as constants for 44100 Hz, scaled by `sample_rate / 44100`
- Parameter defaults as module-level `DEFAULT_*` constants (exported for GUI import)
- Processing is sample-by-sample in a Python loop (matching freeverb style)
- Output always clipped to [-1, 1] before dtype conversion

## GUI File (`<name>_gui.py`)

Tkinter GUI matching `freeverb_gui.py` layout exactly:

1. Import `DEFAULT_*` constants and `apply_reverb` from the processor module
2. `_play_wav_file(path)` helper — simpleaudio with platform fallback (Linux/macOS/Windows)
3. Main GUI class inheriting `tk.Tk`:
   - Window title: algorithm display name (e.g. "Dattorro Plate Reverb")
   - File section: LabelFrame with filename label + Browse button
   - Parameters section: LabelFrame with one `tk.Scale` slider per parameter (0.0-1.0, resolution 0.01)
   - Buttons: Apply Reverb, Play Input, Play Output
   - Status bar at bottom
   - Processing runs in a background `threading.Thread`
4. `main()` function + `if __name__ == "__main__"` block

## Test File (`test_<name>.py`)

Pytest tests matching `test_freeverb.py` structure:

1. **Filter unit tests** — one `Test*` class per filter building block:
   - Test basic delay behavior
   - Test feedback/decay
   - Test parameter effects (e.g. damping reduces tail energy)
2. **Processor tests** (`TestProcessorName`):
   - `test_mono_produces_stereo` — mono (N,) input → (N,2) output
   - `test_stereo_produces_stereo` — stereo (N,2) input → (N,2) output
   - `test_dry_only_preserves_input` — wet=0, dry=1 → output equals input
   - `test_wet_adds_reverb_tail` — impulse with wet>0 → tail energy > 1e-6
   - `test_silence_in_silence_out` — zeros in → zeros out
   - At least one algorithm-specific test
3. **WAV round-trip tests** (`TestWavRoundTrip`):
   - `_make_sine_wav` helper (440 Hz sine, int16)
   - `test_mono_wav` — mono WAV in → stereo int16 WAV out, file exists
   - `test_stereo_wav` — stereo WAV in → stereo out
   - `test_output_not_silent` — output has non-zero samples

## Algorithm References

### Dattorro Plate Reverb
Jon Dattorro, "Effect Design Part 1: Reverberator and Other Filters" (1997). Topology: input → pre-delay → bandwidth filter → 4 input diffusers (series allpass) → tank with two cross-coupled decay paths, each having decay diffuser + delay + damping. Tap outputs from multiple points in the tank for stereo. Key parameters: pre_delay, bandwidth, damping, decay, input_diffusion, stereo_width.

### Gardner Reverb
Bill Gardner, "The Virtual Acoustic Room" (1992). Nested allpass structures where allpass filters contain other allpass filters in their delay lines. Creates efficient high-density reverberation. Key parameters: room_size, damping, wet, dry, width.

### Gerzon Reverb
Michael Gerzon's randomized FDN. Multiple delay lines (4+) mixed through a unitary matrix (Hadamard or Householder). Delay lengths slowly modulated by LFOs to prevent metallic artifacts. Key parameters: room_size, damping, modulation_depth, modulation_rate, wet, dry, width.

## Dependencies

Only `numpy` and `scipy`. These are already in `requirements.txt`. Do not add any other packages for the processor or tests. The GUI optionally uses `simpleaudio` for playback but must have a fallback.
