"""Export the trained Keras (HDF5) checkpoints in model/ to JSON for the web app.

The browser app in site/ re-implements the dueling Q-network forward pass in
JavaScript, so it only needs the raw layer weights. Run after retraining:

    pip install h5py numpy
    python scripts/export_weights.py
"""
import json
import pathlib
import re

import h5py

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "model"
OUT = ROOT / "site" / "models"
LAYERS = ["dense", "dense_1", "dense_2", "dense_3", "dense_4", "dense_5"]


def export(path: pathlib.Path) -> dict:
    with h5py.File(path, "r") as f:
        mw = f["model_weights"]
        layers = {}
        for name in LAYERS:
            g = mw[name][name]
            layers[name] = {
                "kernel": [[float(f"{v:.7g}") for v in row] for row in g["kernel:0"][()]],
                "bias": [float(f"{v:.7g}") for v in g["bias:0"][()]],
            }
    return {"architecture": "dueling-dqn", "inputs": 15, "layers": layers}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = []
    for path in sorted(SRC.glob("alien_soil_model_ep*.keras")):
        ep = int(re.search(r"ep(\d+)", path.name).group(1))
        (OUT / f"ep{ep}.json").write_text(json.dumps(export(path), separators=(",", ":")))
        episodes.append(ep)
    episodes.sort()
    (OUT / "index.json").write_text(json.dumps({"episodes": episodes, "default": max(episodes)}))
    print(f"Exported {len(episodes)} checkpoints to {OUT}")


if __name__ == "__main__":
    main()
