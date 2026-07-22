#!/usr/bin/env python3
"""
export_to_ollama.py -- add a trained LoRA adapter onto the SAME Ollama model you
serve for inference, so a fine-tune actually reaches production.

Weaver (agent/weaver_node.py) generates only through Ollama, using OLLAMA_MODEL.
Training produces a PEFT LoRA adapter on a Hugging Face base. This script bridges
the two: it builds a new Ollama model that is your inference model + the adapter --

    FROM     <OLLAMA_MODEL>        # e.g. qwen3:8b, taken from your inference .env
    ADAPTER  <adapter.gguf>        # your trained LoRA, converted to GGUF

then 'ollama create's it. Point OLLAMA_MODEL at the new tag and Weaver serves the
fine-tuned model -- same engine, same model, now carrying your weights.

ALIGNMENT (the one rule): the adapter must have been trained on the HF base that
matches OLLAMA_MODEL's architecture+size (serve qwen3:8b -> train Qwen3-8B). A
LoRA only loads into the architecture it was trained on. Training and inference
use separate, isolated venvs and are not run at the same time, so the whole GPU
is free for whichever is active -- pick the training base to match what you serve.

Run OUTSIDE Docker, in the training venv, on the training box.

Prereqs
  - llama.cpp built, path in $LLAMA_CPP
      (convert_lora_to_gguf.py; also convert_hf_to_gguf.py + llama-quantize for --mode merge)
  - ollama installed and running

Usage -- add the adapter onto your inference model (default):
  python -m training.export_to_ollama --adapter training/export/final_adapter --name weaver-ft
  # --from defaults to $OLLAMA_MODEL; override with --from qwen3:8b

Usage -- bake a self-contained merged model instead of an adapter layer:
  python -m training.export_to_ollama --mode merge \
      --adapter ... --base Qwen/Qwen3-8B --name weaver-ft

Preview every step without executing:  add --dry-run
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd, dry: bool) -> None:
    print("  $ " + " ".join(shlex.quote(str(c)) for c in cmd))
    if dry:
        return
    if subprocess.run([str(c) for c in cmd]).returncode != 0:
        sys.exit(f"ERROR: command failed: {cmd[0]}")


def _llama_cpp(dry: bool = False) -> Path:
    p = os.getenv("LLAMA_CPP")
    if not p:
        if dry:
            return Path("$LLAMA_CPP")
        sys.exit("ERROR: set $LLAMA_CPP to your built llama.cpp checkout (see module docstring).")
    return Path(p).expanduser()


# -- Default mode: adapter layered onto the Ollama inference model ----------------
def convert_lora(adapter: Path, base: str | None, out_gguf: Path, dry: bool) -> None:
    """Convert a PEFT LoRA directory into a GGUF adapter (llama.cpp)."""
    script = _llama_cpp(dry) / "convert_lora_to_gguf.py"
    if not dry and not script.exists():
        sys.exit(f"ERROR: {script} not found -- update llama.cpp.")
    if not dry and not (adapter / "adapter_config.json").exists():
        sys.exit(f"ERROR: {adapter} is not a PEFT adapter (no adapter_config.json).")
    cmd = [sys.executable, script, str(adapter), "--outfile", str(out_gguf), "--outtype", "f16"]
    if base:  # some llama.cpp versions need the base to resolve the architecture
        cmd += ["--base", base]
    print(f"[1/2] Convert LoRA -> GGUF   ({adapter} -> {out_gguf})")
    _run(cmd, dry)


def create_adapter_model(name: str, from_model: str, adapter_gguf: Path, modelfile: Path, dry: bool) -> None:
    print(f"[2/2] ollama create '{name}'   (FROM {from_model} + ADAPTER)")
    content = f"FROM {from_model}\nADAPTER {adapter_gguf.resolve()}\nPARAMETER temperature 0.2\n"
    print("      Modelfile:\n" + "".join("        " + ln + "\n" for ln in content.splitlines()))
    if not dry:
        modelfile.write_text(content, encoding="utf-8")
    _run(["ollama", "create", name, "-f", str(modelfile)], dry)


# -- Alternative mode: merge into a standalone model -----------------------------
def merge_and_convert(adapter: Path, base: str, name: str, outdir: Path, quant: str, dry: bool) -> Path:
    merged = outdir / f"{name}-merged"
    gguf_f16 = outdir / f"{name}-f16.gguf"
    gguf_out = outdir / f"{name}-{quant}.gguf"
    print(f"[1/3] Merge {adapter} into '{base}' (CPU/fp16) -> {merged}")
    if not dry:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map={"": "cpu"}, trust_remote_code=True)
        m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload()
        merged.mkdir(parents=True, exist_ok=True)
        m.save_pretrained(str(merged), safe_serialization=True)
        tsrc = str(adapter) if (adapter / "tokenizer_config.json").exists() else base
        AutoTokenizer.from_pretrained(tsrc, trust_remote_code=True).save_pretrained(str(merged))
    llama = _llama_cpp(dry)
    print(f"[2/3] Convert -> GGUF f16 -> {gguf_f16}")
    _run([sys.executable, llama / "convert_hf_to_gguf.py", str(merged),
          "--outfile", str(gguf_f16), "--outtype", "f16"], dry)
    qbin = llama / "build" / "bin" / ("llama-quantize" + (".exe" if os.name == "nt" else ""))
    print(f"[3/3] Quantize -> {quant} -> {gguf_out}")
    _run([str(qbin), str(gguf_f16), str(gguf_out), quant], dry)
    return gguf_out


def create_merged_model(name: str, gguf_out: Path, modelfile: Path, dry: bool) -> None:
    content = f"FROM {gguf_out.resolve()}\nPARAMETER temperature 0.2\n"
    if not dry:
        modelfile.write_text(content, encoding="utf-8")
    _run(["ollama", "create", name, "-f", str(modelfile)], dry)


def main() -> None:
    ap = argparse.ArgumentParser(description="Add a trained LoRA adapter onto your Ollama inference model.")
    ap.add_argument("--adapter", required=True, type=Path, help="path to the trained final_adapter/")
    ap.add_argument("--name", required=True, help="new Ollama model name to create")
    ap.add_argument("--mode", choices=["adapter", "merge"], default="adapter",
                    help="adapter: layer LoRA onto --from model (default); merge: bake a standalone model")
    ap.add_argument("--from", dest="from_model", default=os.getenv("OLLAMA_MODEL"),
                    help="adapter mode: Ollama model to add weight onto (default $OLLAMA_MODEL)")
    ap.add_argument("--base", help="HF base: convert hint (adapter mode) or full-precision merge target (merge mode)")
    ap.add_argument("--quantize", default="q4_K_M", help="merge mode GGUF quantization (default q4_K_M)")
    ap.add_argument("--outdir", type=Path, default=Path("training/export"))
    ap.add_argument("--dry-run", action="store_true", help="print every step without executing")
    a = ap.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    modelfile = a.outdir / "Modelfile"

    print("=" * 72)
    print(f"Export LoRA -> Ollama   mode={a.mode}  adapter={a.adapter}  name={a.name}")
    print("=" * 72)

    if a.mode == "adapter":
        if not a.from_model:
            sys.exit("ERROR: --from not set and OLLAMA_MODEL not in env -- give the inference model to add weight onto.")
        gguf = a.outdir / f"{a.name}-lora.gguf"
        convert_lora(a.adapter, a.base, gguf, a.dry_run)
        create_adapter_model(a.name, a.from_model, gguf, modelfile, a.dry_run)
    else:
        if not a.base:
            sys.exit("ERROR: --mode merge needs --base (full-precision HF base to merge into).")
        gguf = merge_and_convert(a.adapter, a.base, a.name, a.outdir, a.quantize, a.dry_run)
        create_merged_model(a.name, gguf, modelfile, a.dry_run)

    print("\nDry-run complete." if a.dry_run else "\nDone.")
    print(f"  Next: set OLLAMA_MODEL={a.name} in the inference .env and restart the backend --")
    print(f"  Weaver then serves your fine-tuned model.   Verify:  ollama run {a.name}")


if __name__ == "__main__":
    main()
