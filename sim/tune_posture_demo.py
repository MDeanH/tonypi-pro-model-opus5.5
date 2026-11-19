"""Deterministic tiny controller search exercising the fixed-base training API.

The twin's leg servos are position servos with a finite stiffness, so under gravity a leg settles short of its
command. This searches one outer-loop gain k for the synthetic posture task:

    command = target + k * (target - measured angle)

k = 0 sends the target unchanged; larger k pushes the command past the target to cancel the sag, until the loop
overshoots. A "hold home" candidate (command 0) is the do-nothing baseline. It is an integration example, not a
learned policy, a walking controller or a proposed setting for a physical TonyPi Pro.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leg_posture_env import ACTION_SCALE_RAD, LegPostureEnv, MODEL  # noqa: E402


def evaluate(env, k, seeds: range) -> tuple[float, float]:
    """Mean episode return and mean final RMS error; k=None holds the home pose."""
    returns, errors = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        n = env.n
        total = 0.0
        while True:
            q, target = obs[:n], obs[2 * n:]
            command = np.zeros(n) if k is None else target + k * (target - q)
            obs, reward, terminated, truncated, state = env.step(command / ACTION_SCALE_RAD)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        errors.append(state["error_rms_rad"])
    return float(np.mean(returns)), float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1:
        parser.error("--episodes and --horizon must be positive")
    env = LegPostureEnv(horizon=args.horizon)
    candidates = [None, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    results = []
    for k in candidates:
        ret, err = evaluate(env, k, range(args.episodes))
        results.append({"outer_loop_gain": "hold home" if k is None else k, "mean_return": ret,
                        "mean_final_rms_error_rad": err})
    best = max(results, key=lambda result: result["mean_return"])
    here = Path(__file__).resolve().parent
    print(json.dumps({
        "status": "synthetic fixed-base software demonstration only",
        "model": "build/tonypi_pro_as_is.xml", "episodes_per_candidate": args.episodes,
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256((here / "leg_posture_env.py").read_bytes()).hexdigest(),
        "search_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episode_seeds": list(range(args.episodes)),
        "horizon_steps": args.horizon, "frame_skip": env.frame_skip, "control_period_s": env.control_period_s,
        "candidates": results, "best": best,
    }, indent=2))


if __name__ == "__main__":
    main()
