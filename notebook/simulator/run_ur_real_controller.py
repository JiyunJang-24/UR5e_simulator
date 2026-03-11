import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SIM_ROOT = THIS_DIR.parents[1]
WORKSPACE_ROOT = SIM_ROOT.parent
SERL_INFRA_ROOT = WORKSPACE_ROOT / "serl_vai" / "serl_robot_infra"

sys.path.append(str(SIM_ROOT / "package" / "kinematics_helper"))
sys.path.append(str(SIM_ROOT))
sys.path.append(str(SERL_INFRA_ROOT))

from transforms import rpy2r  # noqa: E402
from ur5e_ik_env import UR5eIKEnv  # noqa: E402
from ur_env.envs.camera_env.config import UR5CameraConfigFinal  # noqa: E402
from robot_controllers.ur5_controller_thread import UrImpedanceController_Thread  # noqa: E402
from scipy.spatial.transform import Rotation as R

class RuntimeConfig(UR5CameraConfigFinal):
    pass


class UR5RealControllerRunner:
    def __init__(
        self,
        robot_ip: str,
        frequency_hz: int = 100,
        mirror_sim: bool = False,
    ):
        RuntimeConfig.ROBOT_IP = robot_ip
        RuntimeConfig.CONTROLLER_HZ = frequency_hz
        self.config = RuntimeConfig

        self.curr_pos = np.zeros(7, dtype=np.float64)
        self.curr_pos_euler = np.zeros(6, dtype=np.float64)
        self.curr_pos_rv = np.zeros(6, dtype=np.float64)
        self.curr_vel = np.zeros(6, dtype=np.float64)
        self.curr_force = np.zeros(3, dtype=np.float64)
        self.curr_torque = np.zeros(3, dtype=np.float64)
        self.curr_q = np.zeros(6, dtype=np.float64)
        self.gripper_state = np.zeros(2, dtype=np.float64)
        self.gripper_working = True

        self.controller = UrImpedanceController_Thread(
            robot_ip=robot_ip,
            config=self.config,
            state_update_callback=self._update_state_from_controller,
        )
        self.controller.start()
        while not self.controller.is_ready():
            time.sleep(0.1)
        time.sleep(0.2)
        self._update_state_from_controller()

        self.sim = None
        self.sim_env = None
        if mirror_sim:
            self._init_sim()

    def _init_sim(self):
        self.sim = UR5eIKEnv(verbose=False)
        q0 = self.curr_q.copy()
        if not np.isfinite(q0).all():
            raise RuntimeError("Failed to read current joint state from UR controller.")
        self.sim.reset(qpos=q0)
        self.sim_env = self.sim.env
        self.sim_env.init_viewer(
            title="UR5e Real Controller Mirror",
            azimuth=170,
            distance=1.8,
            elevation=-20,
            lookat=[0.35, 0.0, 0.4],
        )
        self.sync_sim()

    def _update_state_from_controller(self):
        state = self.controller.get_state()
        self.curr_pos[:] = state["pos"]
        self.curr_vel[:] = state["vel"]
        self.curr_q[:] = state["Q"]
        self.curr_force[:] = state["force"][:3]
        self.curr_torque[:] = state["force"][3:]
        self.gripper_state[:] = state["gripper"]
        self.curr_pos_rv[:] = state["pos_rv"]
        self.curr_pos_euler[:] = state["pos_euler"]
        self.gripper_working = state["gripper_working"]

    def reset(self):
        self._update_state_from_controller()
        self.sync_sim()
        return self.curr_q.copy()

    def move_to_pose(self, pose_euler: np.ndarray, wait: bool = False, timeout: float = 5.0):
        target = np.asarray(pose_euler, dtype=np.float64).reshape(6).copy()
        self.controller.set_target_pose(target)
        if wait:
            self.wait_until_pose(target, timeout=timeout)
        self.sync_sim(target_pose=target)
        return target

    def move_by_delta(self, delta_pose: np.ndarray, wait: bool = False, timeout: float = 5.0):
        self._update_state_from_controller()
        p_curr = self.curr_pos_euler[:3]
        R_curr = rpy2r(self.curr_pos_euler[3:])
        dp = delta_pose[:3]
        drpy = delta_pose[3:]

        p_trgt = p_curr + dp

        R_trgt_mat = rpy2r(drpy) @ R_curr

        euler_trgt = R.from_matrix(R_trgt_mat).as_euler("xyz")

        target = np.concatenate([p_trgt, euler_trgt])

        return self.move_to_pose(target, wait=wait, timeout=timeout)

    def set_gripper(self, closed: bool):
        self.controller.set_gripper_pos(1.0 if closed else -1.0)

    def run_x_axis_scenario(self, step_size: float = 0.02, dwell: float = 2.0, wait: bool = True):
        self.reset()
        start_pose = self.curr_pos_euler.copy()
        pose_plus = start_pose.copy()
        pose_plus[0] += step_size
        pose_minus = start_pose.copy()
        pose_minus[0] -= step_size

        sent_plus = self.move_to_pose(pose_plus, wait=wait, timeout=max(dwell, 5.0))
        print(f"sent pose (+x): {np.array2string(sent_plus, precision=5)}")
        time.sleep(dwell)

        sent_minus = self.move_to_pose(pose_minus, wait=wait, timeout=max(dwell, 5.0))
        print(f"sent pose (-x): {np.array2string(sent_minus, precision=5)}")
        time.sleep(dwell)

        return {
            "start_pose": start_pose,
            "pose_plus": sent_plus,
            "pose_minus": sent_minus,
        }

    def wait_until_pose(self, target_pose: np.ndarray, timeout: float = 5.0, pos_tol: float = 0.01, rot_tol: float = 0.1):
        t0 = time.time()
        while True:
            self._update_state_from_controller()
            pos_err = np.linalg.norm(self.curr_pos_euler[:3] - target_pose[:3])
            rot_err = np.linalg.norm(self.curr_pos_euler[3:] - target_pose[3:])
            if pos_err <= pos_tol and rot_err <= rot_tol:
                return
            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"Target not reached in time. pos_err={pos_err:.4f}, rot_err={rot_err:.4f}"
                )
            time.sleep(0.05)

    def sync_sim(self, target_pose: Optional[np.ndarray] = None):
        if self.sim is None or self.sim_env is None:
            return
        self._update_state_from_controller()
        self.sim.reset(qpos=self.curr_q.copy())
        if target_pose is not None:
            p_trgt = np.asarray(target_pose[:3], dtype=float)
            rpy_trgt = np.asarray(target_pose[3:], dtype=float)
            self.sim_env.plot_sphere(p=p_trgt, r=0.012, rgba=[1.0, 0.2, 0.2, 0.8])
            self.sim_env.plot_T(
                p=p_trgt,
                R=rpy2r(rpy_trgt),
                plot_axis=True,
                axis_len=0.12,
                axis_width=0.004,
                label="Target",
            )
        self.sim_env.viewer_overlay(
            loc="top left",
            text1="TCP xyzrpy",
            text2=np.array2string(self.curr_pos_euler, precision=4),
        )
        self.sim_env.viewer_overlay(
            loc="top left",
            text1="TCP force",
            text2=np.array2string(self.curr_force, precision=3),
        )
        self.sim_env.render()

    def print_status(self):
        self._update_state_from_controller()
        print(f"tcp_euler : {np.array2string(self.curr_pos_euler, precision=5)}")
        print(f"tcp_rotvec: {np.array2string(self.curr_pos_rv, precision=5)}")
        print(f"q         : {np.array2string(self.curr_q, precision=5)}")
        print(f"force     : {np.array2string(self.curr_force, precision=4)}")
        print(f"gripper   : width={self.gripper_state[0]:.3f}, status={self.gripper_state[1]:.3f}")
        print(f"gripper_ok: {self.gripper_working}")

    def close(self):
        if self.controller is not None:
            self.controller.stop()
        if self.sim_env is not None and self.sim_env.use_mujoco_viewer and self.sim_env.is_viewer_alive():
            self.sim_env.close_viewer()


def parse_pose(values, expected_len: int, name: str) -> np.ndarray:
    if len(values) != expected_len:
        raise ValueError(f"{name} requires {expected_len} values, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def build_argparser():
    parser = argparse.ArgumentParser(description="UR5e real controller runner with safe pose streaming.")
    parser.add_argument("--ur-ip", default=UR5CameraConfigFinal.ROBOT_IP, help="UR controller IP")
    parser.add_argument("--hz", type=int, default=100, help="Controller frequency")
    parser.add_argument("--mirror-sim", action="store_true", help="Mirror the real robot state in MuJoCo")
    parser.add_argument("--wait", action="store_true", help="Wait until each target pose is approximately reached")
    parser.add_argument("--pose", nargs=6, type=float, metavar=("X", "Y", "Z", "R", "P", "Y"), help="One-shot absolute target pose in xyz+rpy [m, rad]")
    parser.add_argument("--delta", nargs=6, type=float, metavar=("DX", "DY", "DZ", "DR", "DP", "DY"), help="One-shot delta pose in xyz+rpy [m, rad]")
    parser.add_argument("--grip", choices=["open", "close"], help="One-shot gripper command")
    parser.add_argument("--scenario", choices=["x"], default="x", help="Built-in scenario to run when no explicit pose/delta/grip is given")
    parser.add_argument("--step-size", type=float, default=0.02, help="Scenario translation size in meters")
    parser.add_argument("--dwell", type=float, default=2.0, help="Scenario dwell time at each point in seconds")
    return parser


def run_repl(runner: UR5RealControllerRunner, wait: bool):
    help_text = (
        "Commands:\n"
        "  status\n"
        "  reset\n"
        "  scenario\n"
        "  pose x y z r p y\n"
        "  delta dx dy dz dr dp dy\n"
        "  grip open|close\n"
        "  sync\n"
        "  quit\n"
    )
    print(help_text)
    while True:
        try:
            raw = input("ur5e> ").strip()
        except EOFError:
            break
        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()
        try:
            if cmd == "quit" or cmd == "exit":
                break
            if cmd == "help":
                print(help_text)
            elif cmd == "status":
                runner.print_status()
            elif cmd == "reset":
                runner.reset()
                runner.print_status()
            elif cmd == "scenario":
                runner.run_x_axis_scenario(wait=wait)
                runner.print_status()
            elif cmd == "pose":
                target = parse_pose(parts[1:], 6, "pose")
                clipped = runner.move_to_pose(target, wait=wait)
                print(f"sent pose: {np.array2string(clipped, precision=5)}")
            elif cmd == "delta":
                delta = parse_pose(parts[1:], 6, "delta")
                clipped = runner.move_by_delta(delta, wait=wait)
                print(f"sent pose: {np.array2string(clipped, precision=5)}")
            elif cmd == "grip":
                if len(parts) != 2 or parts[1] not in {"open", "close"}:
                    raise ValueError("grip command must be 'grip open' or 'grip close'")
                runner.set_gripper(closed=(parts[1] == "close"))
            elif cmd == "sync":
                runner.sync_sim()
            else:
                print("Unknown command. Type 'help'.")
        except Exception as exc:
            print(f"[ERROR] {exc}")


def main():
    args = build_argparser().parse_args()
    runner = UR5RealControllerRunner(
        robot_ip=args.ur_ip,
        frequency_hz=args.hz,
        mirror_sim=args.mirror_sim,
    )
    try:
        runner.reset()
        runner.print_status()

        ran_one_shot = False
        if args.pose is not None:
            runner.move_to_pose(np.asarray(args.pose, dtype=np.float64), wait=args.wait)
            runner.print_status()
            ran_one_shot = True

        if args.delta is not None:
            runner.move_by_delta(np.asarray(args.delta, dtype=np.float64), wait=args.wait)
            runner.print_status()
            ran_one_shot = True

        if args.grip is not None:
            runner.set_gripper(closed=(args.grip == "close"))
            ran_one_shot = True

        if not ran_one_shot:
            if args.scenario == "x":
                runner.run_x_axis_scenario(
                    step_size=args.step_size,
                    dwell=args.dwell,
                    wait=args.wait,
                )
                runner.print_status()
            run_repl(runner, wait=args.wait)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
