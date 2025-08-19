# MentorPDM: Learning Data-Driven Curriculum for Multi-Modal Predictive Maintenance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **MentorPDM** for the KDD paper: *"MentorPDM: Learning Data-Driven Curriculum for Multi-Modal Predictive Maintenance"*

## Overview

MentorPDM is a novel multi-modal graph neural network-based curriculum learning framework designed to address critical challenges in predictive maintenance systems for industrial assets such as bearings in rotating machinery. 

### Key Features

- **Multi-modal Learning**: Processes multiple sensor modalities simultaneously
- **Graph Neural Networks**: Leverages graph structures for better representation learning
- **Multi-head Attention**: Implements attention mechanisms for feature fusion
- **Contrastive Learning**: Includes contrastive pretraining for better representations
- **Curriculum Learning**: Adaptive training strategy for improved performance
- **Comprehensive Evaluation**: Extensive baseline comparisons and ablation studies

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/zshuai8/MentorPDM.git
cd MentorPDM
```

2. Create a virtual environment:
```bash
python -m venv mentorpdm_env
source mentorpdm_env/bin/activate  # On Windows: mentorpdm_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

The code is designed to work with the Paderborn University bearing dataset. Download the dataset and organize it as follows:

```
BearingDataCenter/
├── K001/
├── K002/
├── ...
└── KA04/
```

## Quick Start

### Training

Train the MentorPDM model with default configuration:

```bash
python main.py --mode train --config configs/default_config.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python main.py --mode evaluate --model_path checkpoints/best_model.pth
```

### Inference

Run inference on new data:

```bash
python main.py --mode inference --model_path checkpoints/best_model.pth --data_path data/test.csv
```

## Configuration

All hyperparameters and settings can be configured via YAML files. See `configs/default_config.yaml` for the full configuration options.

Key configuration sections:

- **Data**: Data loading and preprocessing settings
- **Model**: Model architecture and hyperparameters
- **Training**: Training procedure and optimization settings
- **Evaluation**: Evaluation metrics and visualization options

## Model Architecture

MentorPDM consists of several key components:

1. **Multi-modal Input Processing**: Handles different sensor modalities
2. **Graph Construction**: Creates graph representations from sensor data
3. **Multi-head Attention**: Attends to relevant features across modalities
4. **Graph Neural Network**: SAGE convolution for graph-based learning
5. **Classification Head**: Final prediction layer with curriculum learning

## Experimental Results

## Repository Structure

```
MentorPDM/
├── configs/                 # Configuration files
│   └── default_config.yaml
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   ├── data_loader.py
│   └── preprocessing.py
├── models/                  # Model definitions
│   ├── __init__.py
│   └── mentorpdm.py
├── training/               # Training scripts
│   ├── __init__.py
│   └── train_mentorpdm.py
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── losses.py
│   └── metrics.py
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{zhang2025mentorpdm,
  title={MentorPDM: Learning Data-Driven Curriculum for Multi-Modal Predictive Maintenance},
  author={Zhang, Shuaicheng and Wang, Tuo and Adams, Stephen and Bhattacharya, Sanmitra and Tiyyagura, Sunil Reddy and Bowen, Edward and Veeramani, Balaji and Zhou, Dawei},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={2837--2847},
  year={2025}
}
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Paderborn University for providing the bearing dataset
- PyTorch Geometric team for the excellent graph neural network library
- The open-source community for various tools and libraries used in this project

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors at [zshuai8@vt.edu]


### Environment Details

- Python: 3.8+
- PyTorch: 1.12+
- PyTorch Geometric: 2.1+
- CUDA: 11.6+ (for GPU support)

