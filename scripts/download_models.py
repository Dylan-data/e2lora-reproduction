"""Download the Hugging Face Transformers files for one model into the project."""
import argparse
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


def curl(*args):
    return subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location",
         "--connect-timeout", "20", "--retry", "3", *args],
        check=True, stdout=subprocess.PIPE,
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face model ID, e.g. FacebookAI/roberta-base")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    repo = quote(args.repo, safe="/")
    revision = quote(args.revision, safe="")
    api = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    model = json.loads(curl(api))
    if model.get("gated"):
        parser.exit(2, f"{args.repo} requires Hugging Face account access.\n")
    sha = model["sha"]
    root = Path(__file__).resolve().parents[1] / "src/e2lora/model/weights"
    output = root / args.repo / sha
    output.mkdir(parents=True, exist_ok=True)
    files = {
        entry["rfilename"]: entry.get("size")
        for entry in model["siblings"] if "/" not in entry["rfilename"]
    }
    base_url = f"https://huggingface.co/{repo}/resolve/{sha}"

    def fetch(name):
        target = output / name
        size = files[name]
        if target.exists() and size is not None and target.stat().st_size == size:
            return
        print(f"{args.repo}: {name}", flush=True)
        partial = target.with_name(target.name + ".part")
        if not (size and partial.exists() and partial.stat().st_size == size):
            curl("--continue-at", "-", "--output", str(partial), f"{base_url}/{quote(name)}")
        if size is not None and partial.stat().st_size != size:
            raise ValueError(f"File size differs from the official listing: {name}")
        partial.replace(target)

    for name in files:
        if name.endswith((".json", ".txt", ".jinja", ".model", ".tiktoken")) or name in {
            "LICENSE", "README.md", "USAGE_POLICY",
        }:
            fetch(name)
    index = output / "model.safetensors.index.json"
    if index.exists():
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif "model.safetensors" in files:
        weights = ["model.safetensors"]
    elif "pytorch_model.bin" in files:
        weights = ["pytorch_model.bin"]
    else:
        raise ValueError(f"No supported Transformers weights in {args.repo}")
    for name in weights:
        fetch(name)
    (output / ".complete").touch()
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
