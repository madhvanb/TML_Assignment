# TML Assignment 1: Membership Inference Attack

This repository contains the code that produced our best leaderboard result
on the Membership Inference Attack task. The attack combines **LiRA**
(Carlini et al. 2022) with **RMIA** (Zarifzadeh et al. 2023): we train shadow
models on random halves of `pub.pt ∪ priv.pt`, fit per-sample Gaussians to
their signals, compute log-likelihood ratios for the target model, then rank
those ratios against pub non-members of the same class.

The single command [`python lira_rmia.py`](lira_rmia.py) reproduces the result.




## Setup

```bash
git clone <https://github.com/madhvanb/TML_Assignment.git>
cd TML_Assignment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset and pretrained model into `data/`:

```bash
mkdir -p data
cd data
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
cd ..
```

---

## Reproducing the leaderboard result

```bash
python lira_rmia.py
```

That's it. All hyperparameters are set as constants at the top of
`lira_rmia.py` edit them there if you want to change anything.

This trains 32 ResNet-18 shadow models on random halves of the combined
`pub and priv` pool (28 000 samples. each shadow sees ≈ 14 000 samples for
training), evaluates each shadow on the full pool, fits per-sample Gaussians
 and writes the final ranked scores to
`outputs/submission.csv`.

Runtime: roughly **1.5 hours on a single Tesla P100** (16 GB VRAM).
Shadow weights and signals are cached under `outputs/shadows/`.


---

## Submitting

```bash

export TML_API_KEY='your_api_key_here'
python task_template.py --file outputs/submission.csv
```

---

## Reproducing on an HPC cluster

The `cluster/` directory contains the Condor scripts we used on the
Saarland HPC.

```bash
chmod +x cluster/run_job.sh
condor_submit cluster/mia.sub
```

---

## References

1. Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P 2017.
2. Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022.
3. Zarifzadeh et al., *Low-Cost High-Power Membership Inference Attacks*, ICML 2024.
