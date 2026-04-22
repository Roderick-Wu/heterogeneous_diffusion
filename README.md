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


## Orchestration mode (heterogeneous experts over SSH)

This repo now includes a simple central orchestrator using Fabric in [fabfile.py](fabfile.py).

### 1) Install Fabric

```bash
pip install -r requirements.txt
```

### 2) Create expert config

```bash
mkdir -p orchestration
cp orchestration/experts.example.json orchestration/experts.json
```

Edit `orchestration/experts.json` with your GPU and TPU node info.

Optional SSH key override:

```bash
export ORCH_SSH_KEY=~/.ssh/id_rsa
```

### 3) Launch asynchronous expert training

```bash
fab train-all
```

Each expert trains independently on its own dataset shard (`shard_index`/`num_shards`).
Jobs are launched with `nohup` so they keep running after orchestration exits.
Accelerator is chosen per expert from `accelerator` (or falls back to `role` if `accelerator` is `auto`).

### 4) Launch asynchronous expert inference

```bash
fab infer-all
```

### 5) Check status/logs

```bash
fab status
fab logs --expert-name=gpu_expert --mode=train --lines=100
fab logs --expert-name=tpu_expert --mode=infer --lines=100
```

### Notes

- Training and inference scripts now support `--accelerator {auto,gpu,tpu,cpu}`.
- TPU mode requires `torch_xla` on the TPU node environment.
- This setup intentionally runs experts independently (no synchronous gradient exchange).

### Manual launch examples

```bash
# GPU expert
python train.py --expert-name gpu_expert --num-shards 2 --shard-index 0 --accelerator gpu

# TPU expert
python train.py --expert-name tpu_expert --num-shards 2 --shard-index 1 --accelerator tpu

# Inference from a trained expert checkpoint
python inference.py --checkpoint-path ./runs/tpu_expert/step_010000.pt --accelerator tpu --expert-name tpu_expert
```