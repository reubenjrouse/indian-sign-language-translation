"""One command for the whole pipeline, per config: download -> preprocess -> train ->
evaluate -> export. Used by notebooks/kaggle_train.ipynb.

Downloading is network-bound and preprocessing is CPU-bound, so while one config
is being preprocessed, the next config's videos download in the background
(log: ``<out>/<name>/download.log``).

Each config gets ``<out>/<name>/`` with the checkpoint, metrics, confusion matrices,
training curves, ONNX model for the web app, and a zip of the extracted landmarks
(so later experiments can retrain without re-running MediaPipe).

    python scripts/run_all.py --config configs/isl60.yaml configs/include50.yaml \
        --data-dir C:\\isl-data --out C:\\isl-data\\runs --workers 8
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def step(name, *args):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)
    print(f"--- {name} done in {(time.time() - start) / 60:.1f} min", flush=True)


def prepare(config_path, data_dir: Path, out: Path):
    """Write a copy of the config with absolute data paths into <out>/<name>/."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run_dir = out / raw["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    raw["data"]["include_root"] = str(data_dir / "include")
    raw["data"]["processed_dir"] = str(data_dir / "processed" / raw["name"])
    cfg = run_dir / "config.yaml"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return raw, run_dir, cfg


def start_background_download(cfg: Path, log: Path):
    print(f"(downloading {cfg.parent.name} in the background -> {log})", flush=True)
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", "scripts/download_include.py", "--config", str(cfg)],
                            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return proc, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--data-dir", required=True, help="where videos and landmarks go (needs a few GB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    data_dir, out = Path(args.data_dir).resolve(), Path(args.out).resolve()
    runs = [prepare(c, data_dir, out) for c in args.config]
    summary, background = {}, None
    for i, (raw, run_dir, cfg) in enumerate(runs):
        name = raw["name"]
        print(f"\n######## {name}: {len(raw['signs'])} signs ########")

        if not args.skip_download:
            if background:  # this config's download was started during the previous one
                proc, log = background
                print("\n=== download (waiting for background download) ===", flush=True)
                proc.wait()
                log.close()
                background = None
                if proc.returncode:
                    print(f"background download failed, retrying in the foreground (see {log.name})")
                    step("download", "scripts/download_include.py", "--config", cfg)
            else:
                step("download", "scripts/download_include.py", "--config", cfg)
            if i + 1 < len(runs):
                next_cfg = runs[i + 1][2]
                background = start_background_download(next_cfg, runs[i + 1][1] / "download.log")

        step("preprocess", "scripts/preprocess.py", "--config", cfg, "--workers", args.workers)
        step("train", "scripts/train.py", "--config", cfg, "--out", run_dir)
        for split in ("val", "test"):
            step(f"evaluate {split}", "scripts/evaluate.py", "--config", cfg,
                 "--checkpoint", run_dir / "best.pt", "--split", split)
        step("export", "scripts/export_onnx.py", "--config", cfg,
             "--checkpoint", run_dir / "best.pt", "--metrics", run_dir / "metrics_test.json",
             "--out", run_dir / "web" / name)

        shutil.make_archive(str(run_dir / "landmarks"), "zip", raw["data"]["processed_dir"])
        metrics = {s: json.loads((run_dir / f"metrics_{s}.json").read_text(encoding="utf-8")) for s in ("val", "test")}
        summary[name] = {s: {k: m[k] for k in ("accuracy", "macro_f1", "samples")} for s, m in metrics.items()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n######## summary ########")
    for name, splits in summary.items():
        t = splits["test"]
        print(f"{name:12s} test accuracy {t['accuracy']:.1%}  macro F1 {t['macro_f1']:.3f}  ({t['samples']} videos)")


if __name__ == "__main__":
    main()
