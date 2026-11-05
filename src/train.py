import argparse
import copy
import json
import platform
import traceback
from pathlib import Path

import numpy as np
import torch

from src.data import splits
from src.model import Model, direction_error, readout


def tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


@torch.no_grad()
def validation(model, samples, device, batch_size):
    errors = []
    failures = 0
    for start in range(0, len(samples.tokens), batch_size):
        batch = slice(start, start + batch_size)
        states, _ = model(tensor(samples.tokens[batch], device))
        pred, valid = readout(states[-1])
        errors.extend(direction_error(pred, tensor(samples.target[batch], device), valid).tolist())
        failures += (~valid).sum().item()
    return {"mean_error": sum(errors) / len(errors), "failures": failures}


def train_one(config, data, model, batches, device, directory):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    x = tensor(data["train"].tokens, device)
    y = tensor(data["train"].target, device)
    with (directory / "metrics.jsonl").open("w") as log:
        for step in range(len(batches) + 1):
            if step % config["log_every"] == 0 or step == len(batches):
                metrics = validation(model, data["validation"], device, config["batch_size"])
                print(json.dumps({"step": step, **metrics}), file=log, flush=True)
            if step == len(batches):
                break
            ids = batches[step]
            optimizer.zero_grad(set_to_none=True)
            states, _ = model(x[ids])
            pred, valid = readout(states[-1])
            if not valid.all():
                raise FloatingPointError(f"Invalid training readout at step {step + 1}")
            loss = direction_error(pred, y[ids], valid).mean()
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite gradient at step {step + 1}")
            optimizer.step()
    torch.save(model.state_dict(), directory / "model.pt")


def run(config, output, device):
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
    }, indent=2))
    data = splits(config)
    torch.manual_seed(config["seed"])
    initial = Model(config["d"], False, config["init_std"])
    torch.save(initial.state_dict(), output / "initial.pt")
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1]))
    batches = rng.integers(0, config["train_size"], (config["steps"], config["batch_size"]))
    np.save(output / "batches.npy", batches)
    statuses = []
    for norm_on in (True, False):
        name = "norm_on" if norm_on else "norm_off"
        directory = output / name
        directory.mkdir()
        model = copy.deepcopy(initial)
        model.norm_on = norm_on
        status = {"condition": name, "status": "running"}
        path = directory / "status.json"
        path.write_text(json.dumps(status))
        try:
            train_one(config, data, model, batches, device, directory)
            status["status"] = "complete"
        except Exception:
            status["status"] = "failed"
            (directory / "failure.txt").write_text(traceback.format_exc())
        finally:
            path.write_text(json.dumps(status, indent=2))
        statuses.append(status)
    print(json.dumps(statuses, indent=2))
    return all(status["status"] == "complete" for status in statuses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device == "cpu":
        torch.set_num_threads(1)
    raise SystemExit(0 if run(config, args.output, torch.device(args.device)) else 1)


if __name__ == "__main__":
    main()
