import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from fabric import Connection, task


@dataclass
class ExpertConfig:
    name: str
    host: str
    user: str
    role: str
    project_dir: str
    python_bin: str
    train_work_dir: str
    infer_work_dir: str
    checkpoint_path: str
    dataset_dir: str = "./dataset"
    train_steps: int = 10000
    train_batch_size: int = 64
    inference_samples: int = 64
    accelerator: str = "auto"


def _expert_accelerator(expert: ExpertConfig) -> str:
    if expert.accelerator != "auto":
        return expert.accelerator
    role = expert.role.lower()
    if role in {"gpu", "tpu", "cpu"}:
        return role
    return "auto"


DEFAULT_CONFIG_PATH = "orchestration/experts.json"


def _load_experts(config_path: str = DEFAULT_CONFIG_PATH) -> List[ExpertConfig]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy orchestration/experts.example.json to orchestration/experts.json and edit it."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    experts = []
    for item in payload.get("experts", []):
        experts.append(ExpertConfig(**item))

    if not experts:
        raise ValueError("No experts configured in orchestration config.")
    return experts


def _conn_kwargs() -> Dict[str, object]:
    key_path = os.getenv("ORCH_SSH_KEY")
    kwargs: Dict[str, object] = {}
    if key_path:
        kwargs["connect_kwargs"] = {"key_filename": key_path}
    return kwargs


def _run_detached(c, command: str) -> None:
    # Uses nohup so central orchestrator can exit while worker keeps training.
    c.run(f"nohup bash -lc '{command}' > /dev/null 2>&1 &", pty=False)


def _connection_for_expert(ctx, expert: ExpertConfig) -> Connection:
    connect_kwargs = _conn_kwargs().get("connect_kwargs", {})
    return Connection(
        host=expert.host,
        user=expert.user,
        connect_kwargs=connect_kwargs,
        config=ctx.config,
    )


def _train_command(expert: ExpertConfig, shard_index: int, num_shards: int) -> str:
    log_path = f"{expert.train_work_dir}/train.log"
    accelerator = _expert_accelerator(expert)
    return (
        f"cd {expert.project_dir} && "
        f"mkdir -p {expert.train_work_dir} && "
        f"{expert.python_bin} train.py "
        f"--work-dir {expert.train_work_dir} "
        f"--dataset-dir {expert.dataset_dir} "
        f"--steps {expert.train_steps} "
        f"--batch-size {expert.train_batch_size} "
        f"--num-shards {num_shards} "
        f"--shard-index {shard_index} "
        f"--expert-name {expert.name} "
        f"--accelerator {accelerator} "
        f">> {log_path} 2>&1"
    )


def _infer_command(expert: ExpertConfig) -> str:
    log_path = f"{expert.infer_work_dir}/inference.log"
    accelerator = _expert_accelerator(expert)
    return (
        f"cd {expert.project_dir} && "
        f"mkdir -p {expert.infer_work_dir} && "
        f"{expert.python_bin} inference.py "
        f"--checkpoint-path {expert.checkpoint_path} "
        f"--work-dir {expert.infer_work_dir} "
        f"--num-samples {expert.inference_samples} "
        f"--expert-name {expert.name} "
        f"--accelerator {accelerator} "
        f">> {log_path} 2>&1"
    )


@task
def train_all(ctx, config_path=DEFAULT_CONFIG_PATH):
    """Kick off independent expert training jobs on all configured nodes."""
    experts = _load_experts(config_path)
    num_shards = len(experts)

    for shard_index, expert in enumerate(experts):
        print(f"Launching train job for {expert.name} ({expert.role}) on {expert.host}")
        c = _connection_for_expert(ctx, expert)
        cmd = _train_command(expert, shard_index, num_shards)
        _run_detached(c, cmd)

    print("Train jobs launched.")


@task
def infer_all(ctx, config_path=DEFAULT_CONFIG_PATH):
    """Kick off independent inference jobs on all configured nodes."""
    experts = _load_experts(config_path)

    for expert in experts:
        print(f"Launching inference job for {expert.name} ({expert.role}) on {expert.host}")
        c = _connection_for_expert(ctx, expert)
        cmd = _infer_command(expert)
        _run_detached(c, cmd)

    print("Inference jobs launched.")


@task
def status(ctx, config_path=DEFAULT_CONFIG_PATH):
    """Show running train/inference processes on each configured node."""
    experts = _load_experts(config_path)

    for expert in experts:
        print(f"\n[{expert.name} @ {expert.host}]")
        c = _connection_for_expert(ctx, expert)
        c.run("ps -ef | grep -E 'train.py|inference.py' | grep -v grep", warn=True)


@task
def logs(ctx, expert_name, lines=50, mode="train", config_path=DEFAULT_CONFIG_PATH):
    """Tail train or inference logs for a specific expert."""
    experts = _load_experts(config_path)
    by_name = {e.name: e for e in experts}

    if expert_name not in by_name:
        raise ValueError(f"Unknown expert_name={expert_name}. Available: {sorted(by_name.keys())}")

    expert = by_name[expert_name]
    c = _connection_for_expert(ctx, expert)

    if mode == "train":
        log_path = f"{expert.train_work_dir}/train.log"
    elif mode == "infer":
        log_path = f"{expert.infer_work_dir}/inference.log"
    else:
        raise ValueError("mode must be one of: train, infer")

    c.run(f"tail -n {int(lines)} {log_path}")
