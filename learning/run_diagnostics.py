"""Run the CPU inspection suite and emit a versioned, machine-readable report."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    ("tokenizer_contract", "inspect_tokenizer.py"),
    ("pretrain_supervision", "inspect_pretrain_labels.py"),
    ("sft_supervision", "inspect_sft_labels.py"),
    ("tensor_shapes", "inspect_model_shapes.py"),
    ("gradient_update", "one_step_training.py"),
    ("prompt_protocol", "inspect_prompts.py"),
    ("model_census", "profile_model.py"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/diagnostics/latest.json")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per experiment")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    environment = os.environ.copy()
    environment.update({"HF_HOME": str(ROOT / ".cache/huggingface"),
                        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"})
    packages = {}
    for package in ("torch", "transformers", "numpy", "datasets"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    sources = ["model/model_minimind.py", "model/tokenizer.json", "model/tokenizer_config.json",
               "dataset/lm_dataset.py", "learning/sample_pretrain.jsonl", "learning/sample_sft.jsonl",
               "requirements-learning.txt", "learning/run_diagnostics.py"]
    sources += [f"learning/{filename}" for _, filename in EXPERIMENTS]
    report = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU mechanism checks on toy samples; no trained checkpoints or quality benchmarks",
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "architecture": platform.machine(), "packages": packages},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "experiments": [],
    }
    def redact(value):
        # Avoid embedding a workstation username or absolute local paths in a public report.
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return (value or "").replace(str(ROOT), "<repo>").replace(str(Path.home()), "<home>")

    for identifier, filename in EXPERIMENTS:
        script = f"learning/{filename}"
        try:
            process = subprocess.run([sys.executable, script], cwd=ROOT, env=environment,
                                     capture_output=True, text=True, timeout=args.timeout)
            item = {"id": identifier, "script": script, "status": "passed" if process.returncode == 0 else "failed",
                    "returncode": process.returncode, "stdout": redact(process.stdout), "stderr": redact(process.stderr)}
        except subprocess.TimeoutExpired as exc:
            item = {"id": identifier, "script": script, "status": "timeout", "returncode": None,
                    "stdout": redact(exc.stdout), "stderr": redact(exc.stderr)}
        report["experiments"].append(item)
        print(f"{item['status']:7} {identifier}", flush=True)
    report["passed"] = all(item["status"] == "passed" for item in report["experiments"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
