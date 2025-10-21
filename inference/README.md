# NOBLE Example Notebooks

This folder contains five Python notebooks designed to help you explore and test the trained $\texttt{NOBLE}$ model.

---

### 1. `compute_ephys_feature.ipynb`

Demonstrates how to compute electrophysiological features from voltage traces predicted by $\texttt{NOBLE}$.

---

### 2. `compare_ephys_features_experiments.ipynb`

Compares electrophysiological features extracted from $\texttt{NOBLE}$-predicted voltage traces with those obtained from experimental recordings.

---

### 3. `ensemble_generation.ipynb`

Shows how to generate ensemble predictions using $\texttt{NOBLE}$ by sampling the neuron models used during training, denoted as $\{\text{HoF}^{train}\}$.

---

### 4. `arbitrary_ensemble_generation.ipynb`

Illustrates how to generate ensemble predictions by sampling arbitrary synthetic neuron models from the latent space, $\mathcal{CH}_{train}$.

---

### 5. `generate_FI_curve.ipynb`

Shows how to compute and plot frequency–current (F–I) curves using $\texttt{NOBLE}$, sampling from the training neuron models $\{\text{HoF}^{train}\}$.

> **Note:** Due to the large number of inferences required to generate the frequency–current curves, a `GPU` with `CUDA` compatibility is needed to run this notebook efficiently.

---

Each notebook includes detailed comments and configurable parameters to facilitate customization and experimentation.
