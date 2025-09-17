import asyncio
import argparse
import logging
import os
import sys
import time
from collections import deque
from typing import Any, Dict, Optional

import tomli
import tomli_w
import sounddevice as sd

from .audio_io import MicReader
from .vad import VADSegmenter
from .asr import ASR
from .tts_edge import EdgeTTS
from .tts_piper import PiperTTS
from .voice_changer_client import VoiceChangerClient, VoiceChangerConfig
from .utils import parse_sd_device


logger = logging.getLogger("vc-translator.main")


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out[k] = _deep_merge(va, vb)
        elif vb is None:
            out[k] = va
        else:
            out[k] = vb if vb is not None else va
    return out


def select_input_device_gui(current: Optional[object] = None) -> Optional[int]:
    """
    Show a small Tkinter window to select an input device.
    Returns the selected device index (int) or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as e:
        print(f"GUI not available for device selection: {e}")
        return None

    devices = sd.query_devices()
    candidates: list[tuple[int, str]] = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            name = d.get("name", f"Device {idx}")
            label = f"[{idx}] {name} (in={d['max_input_channels']}, out={d['max_output_channels']})"
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

    # preselect
    pre_idx: Optional[int] = None
    cur = parse_sd_device(current)
    if isinstance(cur, int):
        for i, (idx, _) in enumerate(candidates):
            if idx == cur:
                pre_idx = i
                break
    if pre_idx is None and isinstance(cur, str):
        for i, (idx, _) in enumerate(candidates):
            if cur in candidates[i][1]:
                pre_idx = i
                break
    if pre_idx is None:
        pre_idx = 0
    listbox.selection_set(pre_idx)
    listbox.see(pre_idx)

    selected: dict[str, Optional[int]] = {"idx": None}

    def on_ok():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("Select device", "Please select an input device.")
            return
        pos = sel[0]
        selected["idx"] = candidates[pos][0]
        root.destroy()

    def on_cancel():
        selected["idx"] = None
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=12, pady=12)
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right")
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=(0, 8))

    root.mainloop()
    return selected["idx"]


def select_output_device_gui(current: Optional[object] = None, title: str = "Select Output Device", prompt: str = "Choose speaker / output device:") -> Optional[int]:
    """
    Show a small Tkinter window to select an output device.
    Returns the selected device index (int) or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as e:
        print(f"GUI not available for device selection: {e}")
        return None

    devices = sd.query_devices()
    candidates: list[tuple[int, str]] = []
    for idx, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            name = d.get("name", f"Device {idx}")
            label = f"[{idx}] {name} (in={d['max_input_channels']}, out={d['max_output_channels']})"
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

    # preselect
    pre_idx: Optional[int] = None
    cur = parse_sd_device(current)
    if isinstance(cur, int):
        for i, (idx, _) in enumerate(candidates):
            if idx == cur:
                pre_idx = i
                break
    if pre_idx is None and isinstance(cur, str):
        for i, (idx, _) in enumerate(candidates):
            if cur in candidates[i][1]:
                pre_idx = i
                break
    if pre_idx is None:
        pre_idx = 0
    listbox.selection_set(pre_idx)
    listbox.see(pre_idx)

    selected: dict[str, Optional[int]] = {"idx": None}

    def on_ok():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("Select device", "Please select an output device.")
            return
        pos = sel[0]
        selected["idx"] = candidates[pos][0]
        root.destroy()

    def on_cancel():
        selected["idx"] = None
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=12, pady=12)
    ttk.Button(btns, text="OK", command=on_ok).pack(side="right")
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=(0, 8))

    root.mainloop()
    return selected["idx"]


def _save_devices_local(cfg_path: str, input_dev: Optional[int], output_dev: Optional[int]) -> None:
    """Persist selected devices to config/local.toml next to cfg_path."""
    base_dir = os.path.dirname(cfg_path)
    local_path = os.path.join(base_dir, "local.toml")
    data: Dict[str, Any] = {"device": {}}
    if input_dev is not None:
        data["device"]["input_device"] = input_dev
    if output_dev is not None:
        data["device"]["output_device"] = output_dev
    try:
        os.makedirs(base_dir, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(tomli_w.dumps(data).encode("utf-8"))
        print(f"Saved device selection to {local_path}")
    except Exception as e:
        print(f"Failed to save device selection: {e}")


def load_cfg(path: str = "config/settings.toml") -> Dict[str, Any]:
    with open(path, "rb") as f:
        base = tomli.load(f)
    local_path = os.path.join(os.path.dirname(path), "local.toml")
    if os.path.isfile(local_path):
        try:
            with open(local_path, "rb") as lf:
                local = tomli.load(lf)
            base = _deep_merge(base, local)
        except Exception:
            pass
    return base


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Overlay selected CLI overrides onto loaded config dict."""
    c = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cfg.items()}

    if getattr(args, 'engine', None):
        c.setdefault("tts", {})["engine"] = args.engine
    if getattr(args, 'voice', None):
        c.setdefault("tts", {})["voice"] = args.voice
    if getattr(args, 'piper_model', None):
        c.setdefault("tts", {})["piper_model"] = args.piper_model
    if getattr(args, 'pace', None) is not None:
        c.setdefault("tts", {})["pace"] = float(args.pace)
    if getattr(args, 'volume_db', None) is not None:
        c.setdefault("tts", {})["volume_db"] = float(args.volume_db)

    if getattr(args, 'input_device', None):
        c.setdefault("device", {})["input_device"] = args.input_device
    if getattr(args, 'output_device', None):
        c.setdefault("device", {})["output_device"] = args.output_device

    return c


def print_devices() -> None:
    devs = sd.query_devices()
    print("Input Devices:")
    for idx, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            name = d.get("name", f"Device {idx}")
            print(f"  [{idx}] {name}  (in={d['max_input_channels']}, out={d['max_output_channels']})")
    print("")
    print("Output Devices:")
    for idx, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0:
            name = d.get("name", f"Device {idx}")
            print(f"  [{idx}] {name}  (in={d['max_input_channels']}, out={d['max_output_channels']})")


async def list_edge_voices() -> None:
    try:
        import edge_tts
    except Exception as e:
        print(f"edge-tts not installed or failed to import: {e}")
        return
    voices = await edge_tts.list_voices()
    print("Edge TTS Voices (English locales):")
    for v in voices:
        if v.get("Locale", "").startswith("en-"):
            print(f"  {v['ShortName']}  ({v['Gender']}, {v['Locale']})")


async def run(cfg_path: str = "config/settings.toml", args: Optional[argparse.Namespace] = None):
    cfg = load_cfg(cfg_path)
    if args is not None:
        cfg = apply_overrides(cfg, args)

    # Prompt for input device via GUI unless provided via CLI
    if not (args and getattr(args, "input_device", None)):
        chosen = select_input_device_gui(current=cfg["device"].get("input_device"))
        if chosen is None:
            print("Input selection cancelled. Exiting.")
            return
        cfg.setdefault("device", {})["input_device"] = chosen
    # Prompt for output device via GUI unless provided via CLI
    if not (args and getattr(args, "output_device", None)):
        chosen_out = select_output_device_gui(current=cfg["device"].get("output_device"))
        if chosen_out is None:
            print("Output selection cancelled. Exiting.")
            return
        cfg.setdefault("device", {})["output_device"] = chosen_out

    # Offer to persist selection unless --no-save-devices
    auto_save = bool(args and getattr(args, "save_devices", False))
    if auto_save:
        _save_devices_local(cfg_path, cfg["device"].get("input_device"), cfg["device"].get("output_device"))
    else:
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk(); r.withdraw()
            if messagebox.askyesno("Save devices", "Save selected devices for next run?"):
                _save_devices_local(cfg_path, cfg["device"].get("input_device"), cfg["device"].get("output_device"))
            r.destroy()
        except Exception:
            pass

    # 1) IO
    mic = MicReader(
        sr=cfg["device"]["input_samplerate"],
        block_ms=cfg["vad"]["frame_ms"],
        input_device=cfg["device"].get("input_device") or None,
    )
    vad = VADSegmenter(
        sr=cfg["device"]["input_samplerate"],
        frame_ms=cfg["vad"]["frame_ms"],
        aggressiveness=cfg["vad"]["aggressiveness"],
        min_speech_sec=cfg["vad"]["min_speech_sec"],
        max_utt_sec=cfg["vad"]["max_utterance_sec"],
        silence_end_ms=cfg["vad"]["silence_end_ms"],
    )

    # 2) ASR
    asr = ASR(
        model=cfg["asr"]["whisper_model"],
        device=cfg["asr"]["device"],
        compute_type=cfg["asr"]["compute_type"],
        task=cfg["asr"]["task"],
        language=cfg["asr"]["language"],
        beam_size=cfg["asr"]["beam_size"],
        input_sr=cfg["device"]["input_samplerate"],
    )

    vc_cfg = cfg.get("voice_changer", {}) or {}
    voice_changer_client: Optional[VoiceChangerClient] = None

    def _maybe_sr(key: str) -> Optional[int]:
        value = vc_cfg.get(key)
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    wants_vc = bool(vc_cfg.get("enabled")) or bool(vc_cfg.get("save_original_path")) or bool(vc_cfg.get("save_converted_path"))
    fallback_device_cfg = vc_cfg.get("fallback_output_device")
    if wants_vc:
        if not fallback_device_cfg:
            chosen_fallback = select_output_device_gui(
                title="Select Fallback Output Device",
                prompt="Choose fallback output device (optional, cancel to skip):"
            )
            if chosen_fallback is not None:
                fallback_device_cfg = chosen_fallback
        voice_changer_client = VoiceChangerClient(
            VoiceChangerConfig(
                enabled=bool(vc_cfg.get("enabled", False)),
                base_url=str(vc_cfg.get("base_url", "http://localhost:18000")),
                endpoint=str(vc_cfg.get("endpoint", "/api/voice-changer/convert_chunk")),
                input_sample_rate=_maybe_sr("input_sample_rate"),
                output_sample_rate=_maybe_sr("output_sample_rate"),
                timeout_sec=float(vc_cfg.get("timeout_sec", 5.0) or 5.0),
                save_original_path=vc_cfg.get("save_original_path") or None,
                save_converted_path=vc_cfg.get("save_converted_path") or None,
                fallback_endpoint=str(vc_cfg.get("fallback_endpoint", "/api/voice-changer/convert_chunk_bulk")),
                fallback_output_device=fallback_device_cfg if fallback_device_cfg not in ("", None) else None,
            )
        )

    # 3) TTS
    output_device = cfg["device"].get("output_device") or None
    if cfg["tts"]["engine"].lower() == "edge":
        normalize_dbfs = cfg.get("stream", {}).get("normalize_dbfs")
        tts = EdgeTTS(
            voice=cfg["tts"]["voice"],
            pace=cfg["tts"]["pace"],
            volume_db=cfg["tts"]["volume_db"],
            out_sr=cfg["device"]["output_samplerate"],
            output_device=output_device,
            normalize_dbfs=normalize_dbfs,
            voice_changer=voice_changer_client,
        )

        async def speak(txt: str):
            return await tts.synth_to_play(txt)

    else:
        tts = PiperTTS(
            model_path=cfg["tts"]["piper_model"],
            out_sr=cfg["device"]["output_samplerate"],
            pace=cfg["tts"]["pace"],
            volume_db=cfg["tts"]["volume_db"],
            output_device=output_device,
            voice_changer=voice_changer_client,
        )

        async def speak(txt: str):
            return tts.synth_to_play(txt)

    print('Starting real-time KO->EN translation. Ctrl+C to stop.')
    mic.start()
    vc_stats = deque(maxlen=50)
    tts_stats = deque(maxlen=50)
    try:
        while True:
            frame = mic.read()
            segment = vad.push(frame)
            if segment is None:
                continue
            segment_samples = len(segment) // 2
            segment_ms = 0.0
            input_sr = cfg["device"]["input_samplerate"]
            if segment_samples > 0 and input_sr > 0:
                segment_ms = (segment_samples / float(input_sr)) * 1000.0
            asr_start = time.perf_counter()
            text_en = asr.transcribe_translate(segment)
            asr_elapsed = time.perf_counter() - asr_start
            logger.debug("ASR took %.3f s for %.1f ms segment", asr_elapsed, segment_ms)
            if text_en:
                print(">>", text_en)
                tts_start = time.perf_counter()
                vc_result = await speak(text_en)
                tts_elapsed = time.perf_counter() - tts_start
                tts_stats.append(tts_elapsed)
                avg_tts = sum(tts_stats) / len(tts_stats)
                if voice_changer_client is not None:
                    if vc_result is not None:
                        vc_stats.append(1 if vc_result else 0)
                    total_chunks, failed_chunks = voice_changer_client.stats()
                    success_pct = (sum(vc_stats) / len(vc_stats) * 100.0) if vc_stats else None
                    logger.debug(
                        "TTS took %.3f s (avg %.3f s); VC chunks total=%d failed=%d success=%s",
                        tts_elapsed,
                        avg_tts,
                        total_chunks,
                        failed_chunks,
                        f"{success_pct:.1f}%" if success_pct is not None else "n/a",
                    )
                else:
                    logger.debug("TTS took %.3f s (avg %.3f s)", tts_elapsed, avg_tts)
            else:
                logger.debug("ASR returned empty text (%.3f s)", asr_elapsed)
    finally:
        mic.stop()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Realtime Korean->English speech translator')
    p.add_argument('--config', default='config/settings.toml', help='Path to settings TOML')
    p.add_argument('--list-devices', action='store_true', help='List audio input/output devices and exit')
    p.add_argument('--list-voices', action='store_true', help='List Edge TTS voices (network) and exit')
    p.add_argument('--engine', choices=['edge', 'piper'], help='Override TTS engine')
    p.add_argument('--voice', help='Edge TTS voice name (e.g. en-US-AriaNeural)')
    p.add_argument('--piper-model', help='Path to Piper .onnx or .tar model')
    p.add_argument('--input-device', help='Input device name or index')
    p.add_argument('--output-device', help='Output device name or index')
    p.add_argument('--save-devices', action='store_true', help='Persist selected devices to config/local.toml')
    p.add_argument('--pace', type=float, help='TTS pace multiplier (1.0 = normal)')
    p.add_argument('--volume-db', type=float, help='TTS gain in dB e.g. -6.0 or +3.0')
    return p


if __name__ == "__main__":
    ap = _build_argparser()
    ns = ap.parse_args()
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
        asyncio.run(run(cfg_path=ns.config, args=ns))
    except KeyboardInterrupt:
        pass
