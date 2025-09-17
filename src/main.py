from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional

import sounddevice as sd
import tomli
import tomli_w

from .pipeline import LANGUAGE_OPTIONS, PRESETS, SharedState, TranslatorPipeline
from .ui import TranslatorUI
from .utils import parse_sd_device


logger = logging.getLogger("vc-translator.main")


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set(a.keys()) | set(b.keys())
    for key in keys:
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out[key] = _deep_merge(va, vb)
        elif vb is None:
            out[key] = va
        else:
            out[key] = vb if vb is not None else va
    return out


def select_input_device_gui(current: Optional[object] = None) -> Optional[int]:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as exc:  # pragma: no cover - GUI fallback
        print(f"GUI not available for device selection: {exc}")
        return None

    devices = sd.query_devices()
    candidates: list[tuple[int, str]] = []
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            name = info.get("name", f"Device {idx}")
            label = f"[{idx}] {name} (in={info['max_input_channels']}, out={info['max_output_channels']})"
            candidates.append((idx, label))

    if not candidates:
        print("No input devices found.")
        return None

    root = tk.Tk()
    root.title("Select Input Device")
    root.geometry("640x360")
    root.resizable(True, True)

    tk.Label(root, text="Choose microphone / input device:").pack(anchor="w", padx=12, pady=(12, 6))
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=12)

    listbox = tk.Listbox(frame, selectmode=tk.SINGLE)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for _, label in candidates:
        listbox.insert(tk.END, label)

    selected_index: Optional[int] = None
    current_val = parse_sd_device(current)
    if isinstance(current_val, int):
        for pos, (idx, _) in enumerate(candidates):
            if idx == current_val:
                selected_index = pos
                break
    elif isinstance(current_val, str):
        for pos, (_, label) in enumerate(candidates):
            if current_val in label:
                selected_index = pos
                break
    if selected_index is None:
        selected_index = 0
    listbox.selection_set(selected_index)
    listbox.see(selected_index)

    result: dict[str, Optional[int]] = {"idx": None}

    def on_ok() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("Select device", "Please select an input device.")
            return
        result["idx"] = candidates[selection[0]][0]
        root.destroy()

    def on_cancel() -> None:
        result["idx"] = None
        root.destroy()

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=12)
    ttk.Button(buttons, text="OK", command=on_ok).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right", padx=(0, 8))

    root.mainloop()
    return result["idx"]


def select_output_device_gui(
    current: Optional[object] = None,
    title: str = "Select Output Device",
    prompt: str = "Choose speaker / output device:",
) -> Optional[int]:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as exc:  # pragma: no cover - GUI fallback
        print(f"GUI not available for device selection: {exc}")
        return None

    devices = sd.query_devices()
    candidates: list[tuple[int, str]] = []
    for idx, info in enumerate(devices):
        if info.get("max_output_channels", 0) > 0:
            name = info.get("name", f"Device {idx}")
            label = f"[{idx}] {name} (in={info['max_input_channels']}, out={info['max_output_channels']})"
            candidates.append((idx, label))

    if not candidates:
        print("No output devices found.")
        return None

    root = tk.Tk()
    root.title(title)
    root.geometry("640x360")
    root.resizable(True, True)

    tk.Label(root, text=prompt).pack(anchor="w", padx=12, pady=(12, 6))
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=12)

    listbox = tk.Listbox(frame, selectmode=tk.SINGLE)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for _, label in candidates:
        listbox.insert(tk.END, label)

    selected_index: Optional[int] = None
    current_val = parse_sd_device(current)
    if isinstance(current_val, int):
        for pos, (idx, _) in enumerate(candidates):
            if idx == current_val:
                selected_index = pos
                break
    elif isinstance(current_val, str):
        for pos, (_, label) in enumerate(candidates):
            if current_val in label:
                selected_index = pos
                break
    if selected_index is None:
        selected_index = 0
    listbox.selection_set(selected_index)
    listbox.see(selected_index)

    result: dict[str, Optional[int]] = {"idx": None}

    def on_ok() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("Select device", "Please select an output device.")
            return
        result["idx"] = candidates[selection[0]][0]
        root.destroy()

    def on_cancel() -> None:
        result["idx"] = None
        root.destroy()

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=12)
    ttk.Button(buttons, text="OK", command=on_ok).pack(side="right")
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(side="right", padx=(0, 8))

    root.mainloop()
    return result["idx"]


def describe_device(device: Optional[object]) -> str:
    if device in (None, "", "default"):
        return "기본 장치"
    parsed = parse_sd_device(device)
    try:
        devices = sd.query_devices()
    except Exception:
        return str(parsed)
    if isinstance(parsed, int) and 0 <= parsed < len(devices):
        name = devices[parsed].get("name", f"Device {parsed}")
        return f"[{parsed}] {name}"
    if isinstance(parsed, str):
        return parsed
    return str(parsed)


def describe_optional_device(device: Optional[object], empty_label: str = "사용 안 함") -> str:
    if device in (None, "", "default"):
        return empty_label
    return describe_device(device)


def _save_devices_local(
    cfg_path: str,
    input_dev: Optional[object],
    output_dev: Optional[object],
    kokoro_dev: Optional[object],
) -> None:
    base_dir = os.path.dirname(cfg_path)
    local_path = os.path.join(base_dir, "local.toml")
    data: Dict[str, Dict[str, Any]] = {}
    device_section: Dict[str, Any] = {}
    if input_dev is not None:
        device_section["input_device"] = parse_sd_device(input_dev)
    if output_dev is not None:
        device_section["output_device"] = parse_sd_device(output_dev)
    if device_section:
        data["device"] = device_section
    if kokoro_dev is not None:
        data.setdefault("kokoro", {})["passthrough_input_device"] = parse_sd_device(kokoro_dev)
    if not data:
        return
    try:
        os.makedirs(base_dir, exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(tomli_w.dumps(data).encode("utf-8"))
        logger.debug("Saved device selection to %s", local_path)
    except Exception as exc:
        logger.warning("Failed to save device selection: %s", exc)


def load_cfg(path: str = "config/settings.toml") -> Dict[str, Any]:
    with open(path, "rb") as fh:
        base = tomli.load(fh)
    local_path = os.path.join(os.path.dirname(path), "local.toml")
    if os.path.isfile(local_path):
        try:
            with open(local_path, "rb") as lf:
                local_cfg = tomli.load(lf)
            base = _deep_merge(base, local_cfg)
        except Exception:
            logger.warning("Failed to read %s", local_path, exc_info=True)
    return base


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cfg.items()}
    device_cfg = merged.setdefault("device", {})
    asr_cfg = merged.setdefault("asr", {})
    app_cfg = merged.setdefault("app", {})

    if getattr(args, "input_device", None) is not None:
        device_cfg["input_device"] = parse_sd_device(args.input_device)
    if getattr(args, "output_device", None) is not None:
        device_cfg["output_device"] = parse_sd_device(args.output_device)
    if getattr(args, "language", None):
        asr_cfg["language"] = args.language
    if getattr(args, "preset", None):
        app_cfg["preset"] = args.preset
    return merged


def print_devices() -> None:
    devices = sd.query_devices()
    print("Input Devices:")
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            name = info.get("name", f"Device {idx}")
            print(f"  [{idx}] {name}  (in={info['max_input_channels']}, out={info['max_output_channels']})")
    print("")
    print("Output Devices:")
    for idx, info in enumerate(devices):
        if info.get("max_output_channels", 0) > 0:
            name = info.get("name", f"Device {idx}")
            print(f"  [{idx}] {name}  (in={info['max_input_channels']}, out={info['max_output_channels']})")


async def list_edge_voices() -> None:
    try:
        import edge_tts
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"edge-tts not installed or failed to import: {exc}")
        return
    voices = await edge_tts.list_voices()
    print("Edge TTS Voices (English locales):")
    for voice in voices:
        if voice.get("Locale", "").startswith("en-"):
            print(f"  {voice['ShortName']}  ({voice['Gender']}, {voice['Locale']})")


def run_app(cfg_path: str, args: Optional[argparse.Namespace]) -> None:
    cfg = load_cfg(cfg_path)
    if args is not None:
        cfg = apply_overrides(cfg, args)

    device_cfg = cfg.setdefault("device", {})
    asr_cfg = cfg.setdefault("asr", {})
    app_cfg = cfg.setdefault("app", {})
    kokoro_cfg = cfg.setdefault("kokoro", {})

    language = asr_cfg.get("language", "ko")
    supported_languages = {option.code for option in LANGUAGE_OPTIONS}
    if language not in supported_languages:
        language = "ko"
        asr_cfg["language"] = language

    preset_key = app_cfg.get("preset", "latency")
    if preset_key not in PRESETS:
        preset_key = "latency"
        app_cfg["preset"] = preset_key

    input_device = device_cfg.get("input_device")
    output_device = device_cfg.get("output_device")
    kokoro_passthrough = kokoro_cfg.get("passthrough_input_device")

    if input_device in (None, ""):
        chosen = select_input_device_gui()
        if chosen is None:
            print("Input selection cancelled. Exiting.")
            return
        input_device = chosen
        device_cfg["input_device"] = chosen

    if output_device in (None, ""):
        chosen = select_output_device_gui()
        if chosen is None:
            print("Output selection cancelled. Exiting.")
            return
        output_device = chosen
        device_cfg["output_device"] = chosen

    input_label = describe_device(input_device)
    output_label = describe_device(output_device)
    kokoro_label = describe_optional_device(kokoro_passthrough)
    kokoro_save = kokoro_passthrough if kokoro_passthrough not in ("",) else None
    _save_devices_local(cfg_path, input_device, output_device, kokoro_save)

    state = SharedState(
        language,
        preset_key,
        input_device,
        output_device,
        input_label,
        output_label,
        kokoro_passthrough,
        kokoro_label,
    )
    state.set_labels(input_label, output_label, kokoro_label)

    def save_devices_callback(
        inp: Optional[object], out: Optional[object], kokoro: Optional[object]
    ) -> None:
        _save_devices_local(cfg_path, inp, out, kokoro)

    pipeline = TranslatorPipeline(cfg, state, save_devices_callback)

    def on_change_input() -> None:
        selected = select_input_device_gui(current=state.get_active_input_device())
        if selected is None:
            return
        state.request_input_device(selected, describe_device(selected))

    def on_change_output() -> None:
        selected = select_output_device_gui(current=state.get_active_output_device())
        if selected is None:
            return
        state.request_output_device(selected, describe_device(selected))

    def on_change_kokoro() -> None:
        selected = select_output_device_gui(
            current=state.get_active_kokoro_device(),
            title="Select Kokoro Output Mirror",
            prompt="Choose the device that should receive Kokoro TTS audio:",
        )
        if selected is None:
            return
        label = describe_device(selected)
        state.request_kokoro_device(selected, label)

    def on_close() -> None:
        pipeline.stop()

    ui = TranslatorUI(state, on_change_input, on_change_output, on_change_kokoro, on_close)
    pipeline.start()
    try:
        ui.run()
    finally:
        pipeline.stop()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime speech translator")
    parser.add_argument("--config", default="config/settings.toml", help="Path to settings TOML")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit")
    parser.add_argument("--list-voices", action="store_true", help="List Edge TTS voices (network) and exit")
    parser.add_argument("--input-device", help="Input device index or name")
    parser.add_argument("--output-device", help="Output device index or name")
    parser.add_argument(
        "--language",
        choices=[option.code for option in LANGUAGE_OPTIONS],
        help="Force input language",
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Select processing preset")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = _build_argparser()
    ns = parser.parse_args()
    if ns.list_devices:
        print_devices()
        sys.exit(0)
    if ns.list_voices:
        try:
            asyncio.run(list_edge_voices())
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    try:
        run_app(cfg_path=ns.config, args=ns)
    except KeyboardInterrupt:
        pass
