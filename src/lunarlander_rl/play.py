"""Watch a trained agent land, and optionally save a GIF.

    lunarlander-play runs/ppo --episodes 3                 # prints the returns
    lunarlander-play runs/ppo --gif assets/ppo.gif         # records the first episode
    lunarlander-play runs/ppo --human                      # opens a window (needs a display)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .envs import make_env


def record_episode(agent, continuous: bool, seed: int = 0, policy=None, max_steps: int = 1000):
    """Play one greedy episode with rendering; returns (frames as uint8 arrays, return)."""
    env = make_env(continuous, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, ret, done, t = [env.render()], 0.0, False, 0
    while not done and t < max_steps:
        action = policy(env, obs) if policy is not None else agent.act(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ret += r
        done, t = term or trunc, t + 1
        frames.append(env.render())
    env.close()
    return frames, ret


def save_gif(frames, path, every: int = 2, scale: float = 0.5, fps: int = 25) -> Path:
    """Write every ``every``-th frame, downscaled by ``scale``, as an animated GIF (Pillow)."""
    from PIL import Image

    imgs = []
    for f in frames[::every]:
        img = Image.fromarray(np.asarray(f))
        if scale != 1:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        imgs.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
    imgs += [imgs[-1]] * 10  # hold the last frame
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 * every / (fps * every)), loop=0)
    return path


def main(argv=None) -> None:
    from .train import load_agent

    p = argparse.ArgumentParser(description="Play a trained LunarLander agent.")
    p.add_argument("run_dir")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", help="save the first episode as a GIF here")
    p.add_argument("--human", action="store_true", help="render in a window")
    a = p.parse_args(argv)

    agent, cfg = load_agent(a.run_dir)
    if a.gif:
        frames, ret = record_episode(agent, cfg.continuous, a.seed)
        print(f"GIF episode return {ret:.1f} -> {save_gif(frames, a.gif)}")
    env = make_env(cfg.continuous, render_mode="human" if a.human else None)
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        done, ret, t = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.act(obs))
            ret, done, t = ret + r, term or trunc, t + 1
        outcome = "landed" if ret >= 200 else ("crashed" if term and ret < 0 else "")
        print(f"episode {ep}: return {ret:7.1f} in {t} steps {outcome}")
    env.close()
