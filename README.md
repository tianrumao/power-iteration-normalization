# Power iteration and normalization

Does internal token normalization change which layers help a four-layer linear attention model recover the leading eigenvector of XXᵀ?

This is a research pilot. We use the input encoding and residual attention block from [Wu, Zhang, and Cao, §3.2](https://arxiv.org/pdf/2606.00605v1#page=5) with independent layer weights. The study takes inspiration from [Which Layer Runs the Program?](https://akyrillidis.github.io/aiowls/which_layer_runs_the_program.html). The experimental details are in [the protocol](docs/protocol.md).

## Run in VS Code

Open the project folder and use a Python environment with PyTorch available. On the pilot Mac, /usr/bin/python3 (3.9.6) was used. For a new environment, Python 3.11 or 3.12 is a practical choice.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train --output runs/my-pilot
python -m src.evaluate runs/my-pilot
```

Use a fresh output directory for each run. The evaluator reloads available trained checkpoints and writes the final-performance table, per-example scores, layer summaries, and plots inside that directory. CPU is the default. Training and evaluation accept `--device cuda` on an NVIDIA runtime.

## Run in Colab

[Open the Colab notebook](https://colab.research.google.com/github/tianrumao/power-iteration-normalization/blob/main/notebooks/run_colab.ipynb). Run the cells from top to bottom. The notebook clones this repository and runs its Python modules. Change the ref in the setup cell to a commit SHA for a repeatable run. The notebook prints the checked-out commit. Its optional final cell copies results to Google Drive after you authorize the mount.

## Share updates

Commit changes from VS Code and push them to GitHub. A fresh Colab runtime pulls the updated code. Update the notebook's ref when you intend to use a new version. The synthetic data can be regenerated from the saved config. Run folders and checkpoints stay outside Git.
