#!/usr/bin/env python3
"""Tkinter GUI for the Gardner reverb processor."""

import os
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from gardner import (
    DEFAULT_DAMPING,
    DEFAULT_DRY,
    DEFAULT_ROOM_SIZE,
    DEFAULT_WET,
    DEFAULT_WIDTH,
    apply_reverb,
)

# Try to import simpleaudio for cross-platform playback
try:
    import simpleaudio as sa

    HAS_SIMPLEAUDIO = True
except ImportError:
    HAS_SIMPLEAUDIO = False


def _play_wav_file(path: str) -> None:
    """Play a WAV file using simpleaudio or a platform-specific fallback."""
    if HAS_SIMPLEAUDIO:
        wave_obj = sa.WaveObject.from_wave_file(path)
        wave_obj.play()
        return

    system = platform.system()
    if system == "Linux":
        subprocess.Popen(["aplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif system == "Darwin":
        subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif system == "Windows":
        # powershell one-liner that works without extra deps
        subprocess.Popen(
            ["powershell", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        raise RuntimeError(f"No audio playback method available on {system}")


class GardnerGUI(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Gardner Reverb")
        self.resizable(False, False)

        self._input_path: str = ""
        self._output_path: str = ""

        self._build_ui()

    # ---- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- File selection --------------------------------------------------
        file_frame = tk.LabelFrame(self, text="File", padx=8, pady=4)
        file_frame.pack(fill="x", **pad)

        self._file_var = tk.StringVar(value="No file selected")
        tk.Label(file_frame, textvariable=self._file_var, anchor="w", width=50).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(file_frame, text="Browse…", command=self._browse).pack(side="right")

        # --- Parameter sliders -----------------------------------------------
        slider_frame = tk.LabelFrame(self, text="Parameters", padx=8, pady=4)
        slider_frame.pack(fill="x", **pad)

        self._sliders: dict[str, tk.Scale] = {}
        params = [
            ("room_size", "Room Size", DEFAULT_ROOM_SIZE),
            ("damping", "Damping", DEFAULT_DAMPING),
            ("wet", "Wet", DEFAULT_WET),
            ("dry", "Dry", DEFAULT_DRY),
            ("width", "Width", DEFAULT_WIDTH),
        ]
        for key, label, default in params:
            row = tk.Frame(slider_frame)
            row.pack(fill="x")
            tk.Label(row, text=label, width=12, anchor="w").pack(side="left")
            scale = tk.Scale(
                row,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                orient="horizontal",
                length=300,
            )
            scale.set(default)
            scale.pack(side="left", fill="x", expand=True)
            self._sliders[key] = scale

        # --- Buttons ---------------------------------------------------------
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", **pad)

        self._apply_btn = tk.Button(
            btn_frame, text="Apply Reverb", command=self._on_apply
        )
        self._apply_btn.pack(side="left", padx=4)

        self._play_in_btn = tk.Button(
            btn_frame, text="Play Input", command=self._play_input, state="disabled"
        )
        self._play_in_btn.pack(side="left", padx=4)

        self._play_out_btn = tk.Button(
            btn_frame, text="Play Output", command=self._play_output, state="disabled"
        )
        self._play_out_btn.pack(side="left", padx=4)

        # --- Status bar ------------------------------------------------------
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(
            self,
            textvariable=self._status_var,
            anchor="w",
            relief="sunken",
            padx=4,
        ).pack(fill="x", side="bottom", **pad)

    # ---- Callbacks ----------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self._input_path = path
            self._file_var.set(os.path.basename(path))
            self._play_in_btn.config(state="normal")
            self._play_out_btn.config(state="disabled")
            self._output_path = ""
            self._set_status(f"Loaded: {os.path.basename(path)}")

    def _on_apply(self) -> None:
        if not self._input_path:
            messagebox.showwarning("No file", "Please select an input WAV file first.")
            return

        # Build output path next to the input file
        base, ext = os.path.splitext(self._input_path)
        self._output_path = f"{base}_reverb{ext}"

        params = {key: scale.get() for key, scale in self._sliders.items()}

        self._apply_btn.config(state="disabled")
        self._set_status("Applying reverb…")

        # Run the (potentially slow) processing in a background thread so
        # the UI stays responsive.
        thread = threading.Thread(
            target=self._apply_worker, args=(self._input_path, self._output_path, params), daemon=True
        )
        thread.start()

    def _apply_worker(self, input_path: str, output_path: str, params: dict) -> None:
        try:
            apply_reverb(
                input_path,
                output_path,
                room_size=params["room_size"],
                damping=params["damping"],
                wet=params["wet"],
                dry=params["dry"],
                width=params["width"],
            )
            self.after(0, self._apply_done, output_path, None)
        except Exception as exc:
            self.after(0, self._apply_done, output_path, exc)

    def _apply_done(self, output_path: str, error: Exception | None) -> None:
        self._apply_btn.config(state="normal")
        if error is not None:
            self._set_status(f"Error: {error}")
            messagebox.showerror("Error", str(error))
        else:
            self._play_out_btn.config(state="normal")
            self._set_status(f"Saved: {os.path.basename(output_path)}")

    def _play_input(self) -> None:
        self._play(self._input_path)

    def _play_output(self) -> None:
        self._play(self._output_path)

    def _play(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Missing file", f"File not found:\n{path}")
            return
        self._set_status(f"Playing: {os.path.basename(path)}")
        try:
            _play_wav_file(path)
        except Exception as exc:
            self._set_status(f"Playback error: {exc}")
            messagebox.showerror("Playback error", str(exc))

    # ---- Helpers ------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)


def main() -> None:
    app = GardnerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
