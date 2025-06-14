# CLaSP (Contrastive Language-Structure Pre-training)

## NOTICE

This repository is provided in a tentative form and will be updated soon.


## Overview
This repository contains the author’s implementation of the paper “Bridging Text and Crystal Structures: Literature-Driven Contrastive Learning for Materials Science.” 
It includes all training and evaluation scripts for CLaSP (Contrastive  Language-Structure Pre-training), proposed in the publication.

## Requirements
Please refer to `docker/Dockerfile` for the complete software environment. All required Python versions and libraries are defined within the Dockerfile.

## Installation
```bash
# From repository root:

# 1. Build Docker image
docker build -t clasp:latest -f docker/Dockerfile .

# 2. Start container (with GPU support if available)
docker run --gpus all \
  -v $(pwd):/workspace/clasp \
  -w /workspace/clasp \
  -it clasp:latest bash
```

## Usage
to be updated

<!-- 1. Data Preparation  

2. Training Example  
   ```bash
   python train_finetuning.py --config configs/clasp_finetune.yaml
   ```

3. Inference Example  
   ```bash
   bash run_experiments.sh
   ``` -->

## Citation
If you use this code, please cite the paper using the following BibTeX entry:

```bibtex
@misc{suzuki2025contrastivelanguagestructurepretrainingdriven,
      title={Contrastive Language-Structure Pre-training Driven by Materials Science Literature}, 
      author={Yuta Suzuki and Tatsunori Taniai and Ryo Igarashi and Kotaro Saito and Naoya Chiba and Yoshitaka Ushiku and Kanta Ono},
      year={2025},
      eprint={2501.12919},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.12919}, 
}
```

## TODO
- [ ] Publish training and evaluation scripts  
- [ ] Release pre-trained model weights  
- [ ] Release dataset
- [ ] Add examples