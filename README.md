
# RECRAFT

Rethinking Cross-Modal Fine-Tuning: Optimizing the Interaction between Feature Alignment and Target Fitting

[![Main Figure](RECRAFT.pdf)]

## Overview

RECRAFT is a principled two-stage algorithm for cross-modal fine-tuning. It optimizes the interaction between **feature alignment** and **target fitting** to effectively adapt pre-trained models to unseen feature modalities.

The method minimizes the semantic gap by performing:
- **Stage 1**: Feature alignment (FA) + feature-label distortion (FLD)
- **Stage 2**: Target predictor optimization

The approach is supported by a theoretical generalization bound and achieves state-of-the-art performance on NAS-Bench-360 and PDEBench benchmarks.

Paper: [Rethinking Cross-Modal Fine-Tuning: Optimizing the Interaction between Feature Alignment and Target Fitting](https://arxiv.org/abs/2601.18231)
## Installation
1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/RECRAFT.git
   cd RECRAFT
2. Set up environment 
   ```bash
   ./startup_hook.sh

##   Usage
1. Run NAS-Bench-360 baseline
   ```bash
   ./run_NasBench.sh
2. Run PDEBench baseline
   ```bash
   ./run_PDE.sh   
## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
tran2026rethinking,
title={Rethinking Cross-Modal Fine-Tuning: Optimizing the Interaction between Feature Alignment and Target Fitting},
author={T. Khiem Tran and Manh Cuong Dao and Phi Le Nguyen and Thao Nguyen Truong and Trong Nghia Hoang},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=YXPoM9GI12}
}

