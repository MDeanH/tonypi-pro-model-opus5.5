"""Small fixed-base MuJoCo task for checking a training pipeline.

This commands the 10 leg servos of the fixed-base TonyPi Pro twin; the arm, hand and head servos hold their home
targets. TonyPi Pro's servos are position-controlled, so actions are normalized POSITION commands (the A2 Ultra
version of this task commands torques because its leg actuators are motors). It is not a walking or balance task,
does not model Hiwonder's on-board motion software, and is not a deployable robot policy.
"""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "build/tonypi_pro_as_is.xml"
LEG_JOINTS = tuple(f"{side}_{j}" for side in ("l", "r")
                   for j in ("hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"))
ACTION_SCALE_RAD = 0.5  # full-scale position command around the home pose (clipped to each servo's ctrlrange)
TARGET_SPAN_RAD = 0.15  # synthetic targets are drawn uniformly within +-this of the home pose


class LegPostureEnv:
    """Gym-shaped reset/step API; no Gym dependency or hardware transport."""

    def __init__(self, frame_skip: int = 5, horizon: int = 100):
        if frame_skip < 1 or horizon < 1:
            raise ValueError("frame_skip and horizon must be positive")
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        if self.model.nu != 20 or self.model.nq != 20:
            raise ValueError("unexpected baseline model dimensions (expected the fixed-base twin)")
        self.data = mujoco.MjData(self.model)
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG_JOINTS]
        act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_JOINTS]
        if min(joint_ids + act_ids) < 0:
            raise ValueError("leg joints or servos missing from the model")
        if [int(self.model.actuator_trnid[a, 0]) for a in act_ids] != joint_ids:
            raise ValueError("each leg servo must drive the joint of the same name")
        self.act_ids = np.array(act_ids)
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]
        self.joint_names = LEG_JOINTS
        self.ctrl_ranges = self.model.actuator_ctrlrange[act_ids].copy()
        self.joint_ranges = self.model.jnt_range[joint_ids].copy()
        self.frame_skip = frame_skip
        self.horizon = horizon
        self.control_period_s = frame_skip * float(self.model.opt.timestep)
        self.steps = 0
        self.done = False
        self.target = np.zeros(len(LEG_JOINTS), dtype=np.float64)
        self.rng = np.random.default_rng(0)

    @property
    def n(self) -> int:
        return len(self.joint_names)

    def observation(self) -> np.ndarray:
        return np.concatenate((
            self.data.qpos[self.qpos_ids],
            self.data.qvel[self.qvel_ids],
            self.target,
        )).copy()

    def reset(self, seed: int | None = None, target: np.ndarray | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)  # home pose, every servo commanded to 0
        target_source = "synthetic" if target is None else "supplied"
        if target is None:
            # Small synthetic task around the CAD home pose; not a checked-safe range for a physical robot.
            sampled = self.rng.uniform(-TARGET_SPAN_RAD, TARGET_SPAN_RAD, size=self.n)
            self.target = np.clip(sampled, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        else:
            target = np.asarray(target, dtype=np.float64)
            if target.shape != (self.n,) or not np.all(np.isfinite(target)):
                raise ValueError(f"target must be {self.n} finite joint coordinates")
            if np.any(target < self.joint_ranges[:, 0]) or np.any(target > self.joint_ranges[:, 1]):
                raise ValueError("target outside the twin's joint ranges")
            self.target = target.copy()
        self.steps = 0
        self.done = False
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), {"joint_names": self.joint_names, "target_source": target_source,
                                    "control_period_s": self.control_period_s}

    def command(self, action: np.ndarray) -> np.ndarray:
        """Normalized action -> absolute servo position command (rad), clipped to each servo's ctrlrange."""
        return np.clip(np.clip(action, -1.0, 1.0) * ACTION_SCALE_RAD, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.n,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be {self.n} finite normalized position commands")
        if self.done:
            raise RuntimeError("episode ended; call reset")
        bounded = np.clip(action, -1.0, 1.0)
        self.data.ctrl[self.act_ids] = self.command(bounded)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        obs = self.observation()
        n = self.n
        finite = bool(np.all(np.isfinite(obs)))
        error = (obs[:n] - self.target) / (self.joint_ranges[:, 1] - self.joint_ranges[:, 0])
        reward = (float(-np.mean(error**2) - 0.001 * np.mean(obs[n:2 * n]**2) - 0.0001 * np.mean(bounded**2))
                  if finite else -1e6)
        info = {
            "time_s": float(self.data.time),
            "error_rms_rad": float(np.sqrt(np.mean((obs[:n] - self.target)**2))) if finite else float("inf"),
            "finite": finite,
        }
        terminated, truncated = not finite, self.steps >= self.horizon
        self.done = terminated or truncated
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = LegPostureEnv()
    for seed in range(3):
        obs, info = env.reset(seed=seed)
        n = env.n
        while True:
            obs, reward, terminated, truncated, state = env.step(obs[2 * n:] / ACTION_SCALE_RAD)
            if terminated or truncated:
                break
        print(f"seed {seed}: command = target, final RMS joint error {state['error_rms_rad']:.4f} rad "
              f"after {state['time_s']:.2f} s")
