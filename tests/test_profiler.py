"""Test PyTorch profiler integration."""

import gzip
import json
from pathlib import Path

import pytest
import torch
from mjlab.tasks.velocity.go1_velocity import Go1VelocityCfg

from mjlab.envs import ManagerBasedRlEnv, ProfilerConfig

TEST_REF = "test_trace"


def _trace_path() -> Path:
  return Path.cwd().parent / "tmp" / f"{TEST_REF}.json.gz"


@pytest.fixture
def cleanup_profiler_output():
  """Cleanup profiler output files after test."""
  yield
  path = _trace_path()
  if path.exists():
    path.unlink()


def test_profiler_disabled_by_default():
  """Test that profiler is disabled by default."""
  env_cfg = Go1VelocityCfg()
  assert env_cfg.profiler.enabled is False


def test_profiler_config():
  """Test profiler configuration."""
  profiler_cfg = ProfilerConfig(
    enabled=True,
    wait_steps=2,
    warmup_steps=2,
    active_steps=3,
    repeat=1,
    ref="my_run",
  )

  assert profiler_cfg.enabled is True
  assert profiler_cfg.wait_steps == 2
  assert profiler_cfg.warmup_steps == 2
  assert profiler_cfg.active_steps == 3
  assert profiler_cfg.repeat == 1
  assert profiler_cfg.ref == "my_run"


def test_profiler_integration(cleanup_profiler_output):
  """Test that profiler can be enabled and runs without errors."""
  env_cfg = Go1VelocityCfg()
  env_cfg.scene.num_envs = 4

  env_cfg.profiler = ProfilerConfig(
    enabled=True,
    wait_steps=2,
    warmup_steps=2,
    active_steps=3,
    repeat=1,
    ref=TEST_REF,
    with_stack=False,
  )

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  assert env._profiler is not None

  obs, _ = env.reset()

  num_steps = 10
  for _ in range(num_steps):
    action = torch.randn(
      env.num_envs, env.action_manager.total_action_dim, device=device
    )
    obs, reward, terminated, truncated, info = env.step(action)

  env.close()

  trace_path = _trace_path()
  assert trace_path.exists()
  assert trace_path.is_file()
  assert trace_path.stat().st_size > 0
  with gzip.open(trace_path, "rt") as f:
    data = json.loads(f.read())
  assert isinstance(data, (dict, list))


def test_profiler_disabled(cleanup_profiler_output):
  """Test that environment works normally with profiler disabled."""
  env_cfg = Go1VelocityCfg()
  env_cfg.scene.num_envs = 4

  env_cfg.profiler.enabled = False

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  assert env._profiler is None

  obs, _ = env.reset()
  action = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=device)
  obs, reward, terminated, truncated, info = env.step(action)

  env.close()

  trace_path = _trace_path()
  assert not trace_path.exists()


def test_profiler_step_counter(cleanup_profiler_output):
  """Test that profiler step counter increments correctly."""
  env_cfg = Go1VelocityCfg()
  env_cfg.scene.num_envs = 4

  env_cfg.profiler = ProfilerConfig(
    enabled=True,
    wait_steps=1,
    warmup_steps=1,
    active_steps=2,
    repeat=1,
    ref=TEST_REF,
  )

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  env.reset()

  assert env._profiler_step == 0

  action = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=device)
  env.step(action)

  assert env._profiler_step == 1

  env.step(action)
  assert env._profiler_step == 2

  env.close()
