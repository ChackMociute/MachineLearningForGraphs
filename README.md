# Machine Learning For Graphs Project

### Setup:
1. cd graphcast
2. pip install -e .
3. pip install torch

### The files:
- updated_model.py – For debugging the updated GraphCast model
- updated_model.ipynb – Similar to the Python file, but also has test MLP and saves the acquired data
- getting_data.ipynb – Loading in example data, model weights, and streaming ERA5 dataset
- full.py and half.py – For running the experiments and generating data. Use respective .job files with Snellius
- plots.ipynb – Generating plots and analysis

### Modified GraphCast files:
- deep_typed_graph_net.py
- graphcast.py
- casting.py
- normalization.py
- autoregressive.py
- rollout.py
