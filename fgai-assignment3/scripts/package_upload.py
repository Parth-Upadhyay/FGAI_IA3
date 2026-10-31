"""Build a clean ZIP without credentials, model weights or full dataset files."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".cache", "checkpoints", ".ipynb_checkpoints"}


def included(path):
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.parts[:2] in (("data", "raw"), ("data", "processed")):
        return False
    if path.name.startswith(".env") or path.suffix in {".pyc", ".log", ".zip"}:
        return False
    return True


def package(destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and included(path) and path.resolve() != destination:
                if path.stat().st_size >= 95 * 1024**2:
                    raise ValueError(f"File exceeds a safe GitHub upload size: {path.name}")
                archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        print(f"Created {destination} with {len(archive.namelist())} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT.parent / "fgai-assignment3-upload.zip"))
    package(parser.parse_args().output)
