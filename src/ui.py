from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict

from .pipeline import LANGUAGE_OPTIONS, PRESETS, SharedState


class TranslatorUI:
    def __init__(
        self,
        state: SharedState,
        on_change_input: Callable[[], None],
        on_change_output: Callable[[], None],
        on_change_kokoro: Callable[[], None],
        on_close: Callable[[], None],
    ) -> None:
        self.state = state
        self._input_callback = on_change_input
        self._output_callback = on_change_output
        self._kokoro_callback = on_change_kokoro
        self._close_callback = on_close

        self.root = tk.Tk()
        self.root.title("Realtime Translator")
        self.root.geometry("420x320")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)

        self._language_labels: Dict[str, str] = {option.code: option.label for option in LANGUAGE_OPTIONS}
        self._language_codes: Dict[str, str] = {option.label: option.code for option in LANGUAGE_OPTIONS}
        self._preset_labels: Dict[str, str] = {preset.key: preset.label for preset in PRESETS.values()}

        self.input_label_var = tk.StringVar(value="")
        self.output_label_var = tk.StringVar(value="")
        self.kokoro_label_var = tk.StringVar(value="")
        self.language_var = tk.StringVar(value=self._language_labels.get("ko", "한국어 → EN"))
        self.preset_var = tk.StringVar(value=self._preset_labels.get("latency", "지연 우선"))
        self.latency_value_var = tk.StringVar(value="0 ms")

        self._build()
        self._refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=(20, 16))
        frame.pack(fill="both", expand=True)

        # Devices
        device_frame = ttk.LabelFrame(frame, text="오디오 장치", padding=(12, 8))
        device_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(device_frame, text="입력 장치").grid(row=0, column=0, sticky="w")
        ttk.Label(device_frame, textvariable=self.input_label_var, width=28).grid(row=0, column=1, padx=(12, 8), sticky="w")
        ttk.Button(device_frame, text="변경", command=self._handle_input_change).grid(row=0, column=2)

        ttk.Label(device_frame, text="출력 장치").grid(row=1, column=0, pady=(6, 0), sticky="w")
        ttk.Label(device_frame, textvariable=self.output_label_var, width=28).grid(
            row=1, column=1, padx=(12, 8), pady=(6, 0), sticky="w"
        )
        ttk.Button(device_frame, text="변경", command=self._handle_output_change).grid(row=1, column=2, pady=(6, 0))

        ttk.Label(device_frame, text="Kokoro 출력").grid(row=2, column=0, pady=(6, 0), sticky="w")
        ttk.Label(device_frame, textvariable=self.kokoro_label_var, width=28).grid(
            row=2, column=1, padx=(12, 8), pady=(6, 0), sticky="w"
        )
        ttk.Button(device_frame, text="변경", command=self._handle_kokoro_change).grid(row=2, column=2, pady=(6, 0))

        # Language selection
        lang_frame = ttk.LabelFrame(frame, text="언어 고정", padding=(12, 8))
        lang_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(lang_frame, text="입력 언어").grid(row=0, column=0, sticky="w")
        lang_box = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=[option.label for option in LANGUAGE_OPTIONS],
            state="readonly",
            width=18,
        )
        lang_box.grid(row=0, column=1, padx=(12, 0), sticky="w")
        lang_box.bind("<<ComboboxSelected>>", self._on_language_selected)

        # Presets
        preset_frame = ttk.LabelFrame(frame, text="세팅 프리셋", padding=(12, 8))
        preset_frame.pack(fill="x", pady=(0, 12))
        for idx, preset in enumerate(PRESETS.values()):
            ttk.Radiobutton(
                preset_frame,
                text=preset.label,
                value=preset.label,
                variable=self.preset_var,
                command=self._on_preset_changed,
            ).grid(row=0, column=idx, padx=(0, 16), sticky="w")

        # Latency gauge
        latency_frame = ttk.LabelFrame(frame, text="지연 (ms)", padding=(12, 12))
        latency_frame.pack(fill="x")
        self.latency_bar = ttk.Progressbar(latency_frame, maximum=3000, length=320)
        self.latency_bar.grid(row=0, column=0, sticky="we")
        ttk.Label(latency_frame, textvariable=self.latency_value_var, width=10, anchor="e").grid(
            row=0, column=1, padx=(12, 0)
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_language_selected(self, _event: object) -> None:
        label = self.language_var.get()
        code = self._language_codes.get(label)
        if code:
            self.state.request_language(code)

    def _on_preset_changed(self) -> None:
        label = self.preset_var.get()
        for key, display in self._preset_labels.items():
            if display == label:
                self.state.request_preset(key)
                return

    def _handle_input_change(self) -> None:
        self._input_callback()

    def _handle_output_change(self) -> None:
        self._output_callback()

    def _handle_kokoro_change(self) -> None:
        self._kokoro_callback()

    def _handle_close(self) -> None:
        self._close_callback()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        snap = self.state.snapshot()
        self.input_label_var.set(snap.get("input_label", ""))
        self.output_label_var.set(snap.get("output_label", ""))
        self.kokoro_label_var.set(snap.get("kokoro_label", ""))
        latency = float(snap.get("latency_ms", 0.0))
        self.latency_bar["value"] = max(0.0, min(latency, float(self.latency_bar["maximum"])))
        self.latency_value_var.set(f"{latency:.0f} ms")

        lang_label = self._language_labels.get(snap.get("language", "ko"), self.language_var.get())
        if self.language_var.get() != lang_label:
            self.language_var.set(lang_label)

        preset_label = self._preset_labels.get(snap.get("preset", "latency"), self.preset_var.get())
        if self.preset_var.get() != preset_label:
            self.preset_var.set(preset_label)

        while True:
            alert = self.state.pop_alert()
            if alert is None:
                break
            try:
                messagebox.showerror("Kokoro 모델 다운로드 필요", alert)
            except Exception:
                print(alert)

        self.root.after(200, self._refresh)

    def run(self) -> None:
        self.root.mainloop()
