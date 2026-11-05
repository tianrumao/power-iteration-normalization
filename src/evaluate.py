import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data import splits
from src.model import Model, direction_error, readout
from src.train import tensor


def save_table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(run, device):
    config = json.loads((run / "config.json").read_text())
    samples = splits(config)["test"]
    target = tensor(samples.target, device)
    records = []

    def record(condition, mode, layer, error, valid):
        for sample, (value, ok) in enumerate(zip(error.tolist(), valid.tolist())):
            records.append({"seed": config["seed"], "condition": condition, "mode": mode,
                            "layer": layer, "sample": sample, "error": value, "failure": int(not ok)})

    power = tensor(samples.power, device)
    for step in range(5):
        valid = torch.ones(len(power), dtype=torch.bool, device=device)
        record("power", "intact", step, direction_error(power[:, step], target, valid), valid)

    for condition in ("norm_on", "norm_off"):
        checkpoint = run / condition / "model.pt"
        if not checkpoint.exists():
            continue
        model = Model(config["d"], condition == "norm_on", config["init_std"]).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        tokens = tensor(samples.tokens, device)
        states, _ = model(tokens)
        for layer, state in enumerate(states, 1):
            pred, valid = readout(state)
            record(condition, "intact", layer, direction_error(pred, target, valid), valid)
        for index in range(4):
            for mode in ("bypass", "matched"):
                changed, possible = model(tokens, index, states[index] if mode == "matched" else None)
                pred, valid = readout(changed[-1])
                valid = valid & possible
                record(condition, mode, index + 1, direction_error(pred, target, valid), valid)

    save_table(run / "per_example.csv", records)
    groups = {}
    for row in records:
        key = row["condition"], row["mode"], row["layer"]
        groups.setdefault(key, []).append(row)
    means = {key: sum(row["error"] for row in rows) / len(rows) for key, rows in groups.items()}
    summary = [
        {"seed": config["seed"], "condition": c, "mode": m, "layer": l,
         "mean_error": means[c, m, l], "failures": sum(row["failure"] for row in rows),
         "samples": len(rows),
         "delta_final_error": means[c, m, l] - means[c, "intact", 4] if m != "intact" else ""}
        for (c, m, l), rows in groups.items()
    ]
    save_table(run / "summary.csv", summary)
    save_table(run / "final_performance.csv", [r for r in summary if r["mode"] == "intact" and r["layer"] == 4])
    print((run / "final_performance.csv").read_text(), end="")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for condition, color in (("power", "gray"), ("norm_on", "tab:blue"), ("norm_off", "tab:orange")):
        layers = [l for c, m, l in means if c == condition and m == "intact"]
        if not layers:
            continue
        axes[0].plot(layers, [means[condition, "intact", l] for l in layers], "o-", color=color, label=condition)
        if condition == "power":
            continue
        for mode, style in (("bypass", "-"), ("matched", "--")):
            delta = [means[condition, mode, l] - means[condition, "intact", 4] for l in layers]
            axes[1].plot(layers, delta, marker="o", linestyle=style, color=color, label=f"{condition} {mode}")
    axes[0].set(xlabel="Layer or power step", ylabel="Mean directional error", xticks=range(5))
    axes[1].set(xlabel="Bypassed layer", ylabel="Increase in final error", xticks=range(1, 5))
    axes[1].axhline(0, color="gray", linewidth=0.8)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.savefig(run / "layer_profiles.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    evaluate(args.run, torch.device(args.device))
