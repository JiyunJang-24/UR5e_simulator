import argparse
from time import sleep

import numpy as np

from quest3 import Quest3InterventionUR5
from run_ur_real_controller import UR5RealControllerRunner
from ur5e_ik_env import UR5eIKEnv


class VRRealURControlEnv:
    def __init__(
        self,
        robot_ip: str,
        controller_hz: int = 100,
        wait: bool = False,
        use_sim: bool = True,
        loop_dt: float = 0.01,
    ):
        self.wait = wait
        self.use_sim = use_sim
        self.loop_dt = loop_dt
        self.runner = UR5RealControllerRunner(
            robot_ip=robot_ip,
            frequency_hz=controller_hz,
            mirror_sim=False,
        )
        self.vr = Quest3InterventionUR5(env=None, fake_env=use_sim, pos_action_gain=0.09, rot_action_gain=0.02)

        q0 = self.runner.reset()
        self.robot = None
        self.env = None
        if self.use_sim:
            self.robot = UR5eIKEnv(verbose=True)
            self.robot.reset(qpos=q0)
            self.env = self.robot.env
            self.env.init_viewer(
                title="UR5e VR Real Control Mirror",
                azimuth=170,
                distance=1.8,
                elevation=-20,
                lookat=[0.35, 0.0, 0.4],
            )

    def _sync_sim_from_real(self, action: np.ndarray | None = None, replaced: bool = False):
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
        self.env.viewer_overlay(
            loc="top left",
            text1="vr active",
            text2=str(replaced),
        )
        if action is not None:
            self.env.viewer_overlay(
                loc="top left",
                text1="last action",
                text2=np.array2string(action, precision=5),
            )
        self.env.render()

    def _build_robot_obs(self):
        self.runner.reset()
        return {
            "robot_state": {
                "cartesian_position": self.runner.curr_pos_euler.copy(),
                "gripper_position": float(self.runner.gripper_state[0]),
            }
        }

    def action(self):
        obs = self._build_robot_obs()
        vr_action, replaced = self.vr.action(obs=obs)
        vr_action = np.asarray(vr_action, dtype=np.float64).reshape(-1)
        pose_action = vr_action[:6]
        if replaced:
            target = self.runner.move_by_delta(
                pose_action,
                wait=self.wait,
                timeout=5.0,
            )
            gripper_target = self.runner.set_gripper(vr_action[6])
        else:
            target = self.runner.move_by_delta(
                np.zeros_like(pose_action),
                wait=self.wait,
                timeout=5.0,
            )

        # self._sync_sim_from_real(action=pose_action, replaced=replaced)
        return {
            "action": pose_action.copy(),
            "target_pose": target.copy(),
            "real_q": self.runner.curr_q.copy(),
            "real_tcp_pose": self.runner.curr_pos_euler.copy(),
            "replaced": replaced,
        }

    def close(self):
        self.runner.close()
        if self.env is not None and self.env.use_mujoco_viewer and self.env.is_viewer_alive():
            self.env.close_viewer()

    def reset(self):
        self.runner.reset()
        # self._sync_sim_from_real()

def build_argparser():
    parser = argparse.ArgumentParser(description="Use Quest3 VR actions to drive the real UR controller.")
    parser.add_argument("--ur-ip", default="192.168.0.251", help="UR controller IP")
    parser.add_argument("--hz", type=int, default=100, help="Real controller frequency")
    parser.add_argument("--wait", action="store_true", help="Wait until the robot approximately reaches each target")
    parser.add_argument("--use-sim", action="store_true", help="Mirror the real robot state in MuJoCo")
    parser.add_argument("--loop-dt", type=float, default=0.01, help="Control loop sleep in seconds")
    return parser


def main():
    args = build_argparser().parse_args()
    env = VRRealURControlEnv(
        robot_ip=args.ur_ip,
        controller_hz=args.hz,
        wait=args.wait,
        use_sim=args.use_sim,
        loop_dt=args.loop_dt,
    )

    try:
        env.reset()
        while True:
            result = env.action()
            print(
                f"[INFO] vr_active={result['replaced']} "
                f"action={np.array2string(result['action'], precision=5)} "
                f"real_tcp={np.array2string(result['real_tcp_pose'], precision=5)}"
            )
            if env.use_sim and env.env is not None and not env.env.is_viewer_alive():
                break
            sleep(env.loop_dt)
    finally:
        env.close()


if __name__ == "__main__":
    main()
