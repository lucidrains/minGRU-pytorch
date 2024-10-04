## minGRU

Implementation of the proposed <a href="https://arxiv.org/abs/2410.01201v1">minGRU</a> in Pytorch, only the log-space numerically stable version.

## Install

```bash
$ pip install minGRU-pytorch
```

## Usage

```python
import torch
from minGRU_pytorch import minGRU

min_gru = minGRU(512)

x = torch.randn(2, 1024, 512)

out = min_gru(x)

assert x.shape == out.shape
```

## Test

enwik8

```bash
$ python train.py
```

## Citations

```bibtex
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
```
