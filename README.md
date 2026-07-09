# <div align="center">**Towards Resilient Transportation: A Conditional Transformer for Accident-Informed Traffic Forecasting**</div>

> **Accepted to KDD 2026**

## Abstract

Traffic prediction remains a fundamental challenge in spatiotemporal data mining, with a critical yet underexplored dimension: modeling the disruptive impact of accidents. While existing approaches excel at capturing recurring patterns, they falter when confronted with the non-stationary perturbations induced by traffic accidents, which create distinctive directional shockwaves through transportation networks. We propose ConFormer (Conditional Transformer), which addresses this limitation through two key innovations: 1) an accident-aware graph propagation mechanism that models how disruptions spread asymmetrically through traffic networks, and 2) Guided Layer Normalization (GLN) that dynamically modulates internal representations based on traffic conditions. We contribute two enriched large-scale benchmark datasets from Tokyo and California highways with detailed accident annotations. Theoretically, we establish how GLN enables adaptive feature transformations through condition-dependent affine parameters, allowing ConFormer to maintain coherent representations across both normal and accident-induced states. Empirically, ConFormer consistently outperforms state-of-the-art models, with improvements of up to 10.7% in accident scenarios, demonstrating that explicitly modeling directional accident propagation substantially enhances predictive performance in complex traffic networks.

## Framework Overview

<div align="center">
  <img src="img/framework_new_v2.jpg" alt="ConFormer Framework" width="800"/>
</div>

## Features

* **Accident-Aware Graph Propagation**: Models how traffic disruptions spread asymmetrically through transportation networks
* **Guided Layer Normalization (GLN)**: Dynamically modulates internal representations based on traffic conditions
* **Dynamic Path Discovery**: Automatically discovers influential paths in the traffic network
* **Multi-scale Temporal Modeling**: Captures both short-term and long-term traffic patterns
* **Comprehensive Embeddings**: Supports time-of-day, day-of-week, accident, and regional embeddings

## Installation

### Requirements

* Python >= 3.7
* PyTorch >= 1.8.0
* NumPy
* Pandas
* PyYAML
* Matplotlib
* torchinfo

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd conformer

# Install dependencies
pip install torch numpy pandas pyyaml matplotlib torchinfo
```

## Project Structure

```text
conformer/
├── model/
│   ├── ConFormer.py          # Main model implementation
│   ├── ConFormer.yaml        # Model configuration files
│   └── train.py              # Training and evaluation script
├── lib/
│   ├── data_prepare.py       # Data loading and preprocessing
│   ├── metrics.py            # Evaluation metrics (RMSE, MAE, MAPE)
│   └── utils.py              # Utility functions
├── data/                     # Dataset directory (to be created)
│   ├── TKY/                  # Tokyo dataset
│   ├── BA/                   # Bay Area dataset
│   └── SD/                   # San Diego dataset
├── logs/                     # Training logs (auto-created)
├── saved_models/             # Saved model checkpoints (auto-created)
└── README.md
```

## Quick Start

### Data Preparation

1. Download the datasets from the [Google Drive link](#datasets).
2. Extract and place the datasets in the `data/` directory with the following structure:

```text
data/
├── TKY/
│   ├── data.npz
│   └── index.npz
├── BA/
│   ├── data.npz
│   └── index.npz
└── SD/
    ├── data.npz
    └── index.npz
```

Depending on the dataset release, additional files such as `adj.npy`, `data.h5`, and `accident.h5` may also be included.

### Training

Train the model on a specific dataset:

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

**Parameters:**

* `-d, --dataset`: Dataset name, e.g., `tky`, `ba`, `sd`
* `-g, --gpu_num`: GPU ID to use, default: `1`
* `-m, --mode`: Running mode, either `train` or `test`, default: `train`

**Examples:**

```bash
# Train on the Tokyo dataset using GPU 0
python train.py -d tky -g 0

# Test on the Bay Area dataset using GPU 0
python train.py -d ba -g 0 -m test
```

### Configuration

Model hyperparameters can be configured in `model/ConFormer.yaml`.

Key parameters include:

* `in_steps`: Input sequence length
* `out_steps`: Output prediction horizon
* `num_heads`: Number of attention heads
* `num_layers`: Number of transformer layers
* `lr`: Learning rate
* `batch_size`: Batch size
* `max_epochs`: Maximum number of training epochs
* `early_stop`: Early stopping patience

## Datasets

The accident-aware traffic forecasting datasets are available at the following link:

**Dataset Download**: [Google Drive](https://drive.google.com/drive/folders/1aHxrooo0WV3k-2u_PTDnWpTBGsbOUH6a?usp=sharing)

The released datasets include:

* **BA**: Bay Area dataset
* **SD**: San Diego dataset
* **TKY**: Tokyo dataset

## Data Format

Each dataset directory contains the following files.

### Required Files

#### `data.npz`

`data.npz` contains the traffic data and auxiliary features.

* Key: `"data"`
* Shape: `(num_samples, num_nodes, num_features)`
* Data type: `float32`
* Features may include:

  * traffic flow
  * time-of-day
  * day-of-week
  * accident annotations

The exact feature layout may vary slightly across datasets. In the default setting, the first feature corresponds to the main traffic measurement, such as traffic flow.

If `data.npz` does not exist but `data.h5` is available, the released code will automatically generate `data.npz` from `data.h5` during preprocessing.

#### `index.npz`

`index.npz` contains the train/validation/test splits.

* Keys: `"train"`, `"val"`, `"test"`
* Shape: `(num_samples, 3)` for each split
* Each row contains `[x_start, x_end, y_end]`

The indices are interpreted as follows:

* `x_start` to `x_end`: input sequence
* `x_end` to `y_end`: prediction target sequence

### Optional or Additional Files

#### `data.h5`

`data.h5` is an alternative HDF5 format for traffic data.

* It is used as a fallback when `data.npz` is not present.
* The code can automatically convert it to `data.npz` on the first run.

#### `adj.npy`

`adj.npy` contains the adjacency matrix of the traffic network graph.

* Shape: `(num_nodes, num_nodes)`
* It represents spatial relationships between traffic sensors or road segments.

#### `accident.h5`

`accident.h5` contains accident annotation data.

* Shape: `(num_time_steps, num_nodes)`
* Each entry corresponds to one sensor and one time slot.
* A value of `0` means that no accident record is associated with this sensor-time slot.
* A non-zero value indicates that one or more accident records, or their aggregated accident-impact contributions, are matched to this sensor-time slot.

## Accident Annotation Format

The accident annotation is generated through a preprocessing pipeline that aligns accident records with the traffic sensor network in both space and time.

Please note that the values in `accident.h5` or in the accident channel of `data.npz` should **not** be directly interpreted as raw accident severity labels or ordinal accident categories.

For example, values such as `10` or `12` do not represent new severity classes. They are produced by aggregation when multiple accident records or multiple severity-related contributions are assigned to the same sensor-time bin.

In other words:

* `0`: no matched accident record
* `> 0`: at least one accident-related record or contribution is associated with the corresponding sensor-time slot

### San Diego Accident Annotation

For the San Diego dataset, the accident annotations are generated by matching records from the US-Accidents dataset to the LargeST-SD traffic sensors.

The preprocessing follows three main steps.

#### 1. Spatial and Temporal Filtering

Accident records are first filtered to match the spatial and temporal coverage of the LargeST-SD dataset.

#### 2. Spatial Matching

Each accident record is assigned to the nearest corresponding LargeST-SD sensor according to the geographic coordinates of the accident record and the sensor metadata.

#### 3. Temporal Alignment

Accident records are aligned to the same 15-minute time grid used by LargeST-SD.

After this process, the accident annotation represents an aggregated accident-impact signal for each sensor and each time slot.

### Usage in ConFormer

In the released ConFormer implementation, the accident annotation is mainly used as an **accident event indicator**.

Specifically, the accident embedding is generated by checking whether the accumulated accident value in the input window is greater than zero. Therefore, for reproducing the results reported in the paper, the most important distinction is accident vs. non-accident rather than the exact magnitude of the aggregated value.

The default experimental setting should be interpreted as follows:

* `0`: non-accident condition
* `> 0`: accident-related condition

The exact magnitude of the accident value is not intended to be used as a fine-grained severity category in the default implementation.

## Citation

If you find this repository, code, or dataset useful for your research, please consider citing our paper.

```bibtex
@inproceedings{wang2026conformer,
  title={Towards Resilient Transportation: A Conditional Transformer for Accident-Informed Traffic Forecasting},
  author={Wang, Hongjun and others},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

## Acknowledgements

We thank the providers of the original traffic and accident datasets. We also thank the research community for their interest in accident-aware traffic forecasting and for helpful feedback on the released datasets and implementation.
