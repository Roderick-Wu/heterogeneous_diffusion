# bagel_take_home

## Usage

config is backed directly into the scripts. In practice you can load it from a yaml or something

Training:
`python train.py`

Inference:
`python inference.py`

Evaluation (Needs images to be generated first in some directory):
`python evaluate.py`


MNIST dataset under dataset/

Final checkpoint and training log under mnist_flow_matching/

Images under inference/*/    (see report)