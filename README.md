# <div align="center">**Towards Resilient Transportation: A Conditional Transformer for Accident-Informed Traffic Forecasting**</div>

## Abstract

Traffic prediction remains a fundamental challenge in spatiotemporal data mining, with a critical yet underexplored dimension: modeling the disruptive impact of accidents. While existing approaches excel at capturing recurring patterns, they falter when confronted with the non-stationary perturbations induced by traffic accidents, which create distinctive directional shockwaves through transportation networks. We propose ConFormer (Conditional Transformer), which addresses this limitation through two key innovations: 1) an accident-aware graph propagation mechanism that models how disruptions spread asymmetrically through traffic networks, and 2) Guided Layer Normalization (GLN) that dynamically modulates internal representations based on traffic conditions. We contribute two enriched large-scale benchmark datasets from Tokyo and California highways with detailed accident annotations. Theoretically, we establish how GLN enables adaptive feature transformations through condition-dependent affine parameters, allowing ConFormer to maintain coherent representations across both normal and accident-induced states. Empirically, ConFormer consistently outperforms state-of-the-art models, with improvements of up to 10.7\% in accident scenarios, demonstrating that explicitly modeling directional accident propagation substantially enhances predictive performance in complex traffic networks.



## Quick Start

### Training Commands

```bash
cd model/
python train.py -d <dataset> -g <gpu_id>
```

> **Note**: The accident datasets will be released upon paper acceptance. The complete datasets and preprocessing code will be available in the project repository.
