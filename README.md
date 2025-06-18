# MentorPDM: Learning Data-Driven Curriculum for Multi-Modal Predictive Maintenance

This repository contains utilities and scripts to train simple models on the Paderborn bearing dataset. The original exploratory work was provided in `MentorPDM.ipynb`. The repository now exposes a lightweight Python package with data loading, model definition, training and evaluation helpers.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Dataset

Download the Paderborn Bearing dataset from the [official website](https://mb.uni-paderborn.de/konstruktions-und-antriebstechnik-kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download) and extract it locally. Organise the files so that each bearing class is stored in a separate folder containing the `.mat` files.

## Training

Run training by pointing to the root directory containing the dataset:

```bash
python -m mentorpdm.train /path/to/dataset --epochs 20
```

This will save the trained model weights to `model.pth`.

## Evaluation

To evaluate a saved model run:

```bash
python -m mentorpdm.evaluate /path/to/dataset model.pth
```

This reports the validation loss and accuracy.
