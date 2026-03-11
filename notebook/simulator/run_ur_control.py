import argparse
from time import sleep

import numpy as np

from run_ur_real_controller import UR5RealControllerRunner
from ur5e_ik_env import UR5eIKEnv


class RealURControlEnv:
    def __init__(
        self,
        robot_ip: str,
        controller_hz: int = 100,
        wait: bool = False,
        dwell: float = 2.0,
        use_sim: bool = True,
    ):
        self.wait = wait
        self.dwell = dwell
        self.use_sim = use_sim
        self.runner = UR5RealControllerRunner(
            robot_ip=robot_ip,
            frequency_hz=controller_hz,
            mirror_sim=False,
        )

        q0 = self.runner.reset()
        self.robot = None
        self.env = None
        if self.use_sim:
            self.robot = UR5eIKEnv(verbose=True)
            self.robot.reset(qpos=q0)

            self.env = self.robot.env
            self.env.init_viewer(
                title="UR5e Real Control Mirror",
                azimuth=170,
                distance=1.8,
                elevation=-20,
                lookat=[0.35, 0.0, 0.4],
            )

    def _sync_sim_from_real(self, action: np.ndarray | None = None):
        q = self.runner.reset()
        if not self.use_sim or self.robot is None or self.env is None:
            return

        self.robot.reset(qpos=q)

        self.env.viewer_overlay(
            loc="top left",
            text1="real q",
            text2=np.array2string(q, precision=5),
        )
        self.env.viewer_overlay(
            loc="top left",
            text1="real tcp xyzrpy",
            text2=np.array2string(self.runner.curr_pos_euler, precision=5),
        )
        self.env.viewer_overlay(
            loc="top left",
            text1="real force",
            text2=np.array2string(self.runner.curr_force, precision=4),
        )
        if action is not None:
            self.env.viewer_overlay(
                loc="top left",
                text1="last action",
                text2=np.array2string(action, precision=5),
            )
        self.env.render()

    def action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(6)
        target = self.runner.move_by_delta(
            action,
            wait=self.wait,
            timeout=max(self.dwell, 5.0),
        )
        self._sync_sim_from_real(action=action)
        return {
            "action": action.copy(),
            "target_pose": target.copy(),
            "real_q": self.runner.curr_q.copy(),
            "real_tcp_pose": self.runner.curr_pos_euler.copy(),
        }

    def close(self):
        self.runner.close()
        if self.env is not None and self.env.use_mujoco_viewer and self.env.is_viewer_alive():
            self.env.close_viewer()


def build_argparser():
    parser = argparse.ArgumentParser(description="Send actions directly to the UR controller and mirror the real robot in sim.")
    parser.add_argument("--ur-ip", default="192.168.0.251", help="UR controller IP")
    parser.add_argument("--hz", type=int, default=100, help="Real controller frequency")
    parser.add_argument("--x-action", type=float, default=0.02, help="X-axis delta action in meters")
    parser.add_argument("--dwell", type=float, default=1.0, help="Pause after each action")
    parser.add_argument("--wait", action="store_true", help="Wait until the robot approximately reaches each target")
    parser.add_argument("--use-sim", action="store_true", help="Mirror the real robot state in MuJoCo")
    return parser


def main():
    args = build_argparser().parse_args()
    env = RealURControlEnv(
        robot_ip=args.ur_ip,
        controller_hz=args.hz,
        wait=args.wait,
        dwell=args.dwell,
        use_sim=args.use_sim,
    )

    actions = [
        np.array([args.x_action, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([-args.x_action, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ]
    idx = 0

    try:
        env._sync_sim_from_real()
        while True:
            action = actions[idx % 2]
            idx += 1
            result = env.action(action)
            print(
                f"[INFO] action={np.array2string(result['action'], precision=5)} "
                f"target={np.array2string(result['target_pose'], precision=5)} "
                f"real_tcp={np.array2string(result['real_tcp_pose'], precision=5)}"
            )
            if env.use_sim and env.env is not None and not env.env.is_viewer_alive():
                break
            sleep(args.dwell)
    finally:
        env.close()


if __name__ == "__main__":
    main()
