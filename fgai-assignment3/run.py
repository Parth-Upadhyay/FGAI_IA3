"""Run from any working directory: paths are resolved relative to this file."""
import argparse
from mt.common import ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description="ICT4403 Assignment 3 translation comparison")
    parser.add_argument("command", choices=("prepare", "preflight", "run", "report", "verify"))
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--models", nargs="+", choices=("basic", "mt5", "llama", "gemma"), default=["basic", "mt5", "llama", "gemma"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse-checkpoints", action="store_true", help="Evaluate existing matching checkpoints instead of retraining")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "prepare":
        from mt.data import prepare
        print(prepare(config)["split_sizes"])
    elif args.command == "preflight":
        from huggingface_hub import hf_hub_download
        import torch
        print("CUDA available:", torch.cuda.is_available())
        failed = []
        for name in args.models:
            if name == "basic":
                continue
            try:
                hf_hub_download(config[name]["model_id"], "config.json", revision=config[name]["revision"])
                print(name, "access OK")
            except Exception as exc:
                print(name, type(exc).__name__, "- accept model terms and run huggingface-cli login locally")
                failed.append(name)
        if failed:
            raise SystemExit(1)
    elif args.command == "run":
        import torch
        from mt.experiment import run_model
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        for name in args.models:
            run_model(name, config, device, args.reuse_checkpoints)
        from mt.reporting import report
        report()
    elif args.command == "report":
        from mt.reporting import report
        report()
    elif args.command == "verify":
        from mt.reporting import verify
        verify(config)


if __name__ == "__main__":
    main()
