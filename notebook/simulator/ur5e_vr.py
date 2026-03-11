from time import sleep

import numpy as np
import mujoco
from ur5e_ik_env import UR5eIKEnv
from transforms import rpy2r
from quest3 import Quest3InterventionUR5
from scipy.spatial.transform import Rotation as R

def main():
    robot = UR5eIKEnv(verbose=True)
    q0 = np.array([-0.1406930128680628, -1.5176025193980713, 2.131890122090475, 4.033455534572266, -1.756136719380514, -0.6224458853351038])
    robot.reset(qpos=q0)
    
    env = robot.env
    env.init_viewer(
        title="UR5e IK",
        azimuth=0,
        distance=2.2,
        width=2560,
        height=2560,
        elevation=0,
        lookat=[0.0, 0.0, 0.4],
    )
    env = Quest3InterventionUR5(env)
    print("VR Intervention Mode: Move the target position with VR controller.")
    try:
        while env.is_viewer_alive():
            p_curr = env.get_p_body(robot.body_name_trgt)
            R_curr = env.get_R_body(robot.body_name_trgt)

            euler = R.from_matrix(R_curr).as_euler("xyz")
            jid = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "g2f85_right_driver_joint"
            )
            gripper = env.data.qpos[jid]
            robot_state = {
                "robot_state": {
                    "cartesian_position": np.concatenate([p_curr, euler]),
                    "gripper_position": gripper,
                }
            }
            # observation in forward () TODO yj

            # pos = obs["robot0_eef_pos"]
            # quat = obs["robot0_eef_quat"]
            # euler = R.from_quat(quat).as_euler("xyz")

            # gripper = obs["robot0_gripper_qpos"][0]

            # robot_state = {
            #     "cartesian_position": np.concatenate([pos, euler]),
            #     "gripper_position": gripper
            # }
            # {"robot_state": robot_state}

            delta, replaced = env.action(obs=robot_state)

            dp = delta[:3]
            drpy = delta[3:]
            # dgripper = delta[6]
            p_trgt = p_curr + dp
            R_trgt = rpy2r(drpy) @ R_curr
            # R_trgt = R_curr @ rpy2r(drpy)
            result = robot.action(
                p_trgt=p_trgt,
                R_trgt=R_trgt,
                q_init=None,
                apply=True,
                max_ik_tick=60,
                ik_err_th=1e-3,
                step_after=True,
                nstep=2,
                verbose=False,
            )
            
            # gripper_target = gripper + dgripper * 0.01
            # gripper_target = np.clip(gripper_target, 0.0, 0.8)

            # env.data.qpos[jid] = gripper_target
            # mujoco.mj_forward(env.model, env.data)
            # env.plot_sphere(p=p_trgt, r=0.012, rgba=[1.0, 0.2, 0.2, 0.8])
            # env.plot_T(
            #     p=p_trgt,
            #     R=R_trgt,
            #     plot_axis=True,
            #     axis_len=0.12,
            #     axis_width=0.004,
            #     label="Target",
            # )
            # env.plot_T(
            #     p=np.array([0.0, 0.0, 0.0]),
            #     R=np.eye(3),
            #     plot_axis=True,
            #     axis_len=0.45,
            #     axis_width=0.006,
            #     label="World",
            # )
            # env.viewer_overlay(
            #     loc="top left",
            #     text1="IK err max",
            #     text2=f"{result['ik_err_max']:.5f}",
            # )
            # env.viewer_overlay(
            #     loc="top left",
            #     text1="EEF_xyzrpy_deg",
            #     text2=f"{result['eef_xyzrpy_deg'][0]:.5f}, {result['eef_xyzrpy_deg'][1]:.5f}, {result['eef_xyzrpy_deg'][2]:.5f}, {result['eef_xyzrpy_deg'][3]:.1f}, {result['eef_xyzrpy_deg'][4]:.1f}, {result['eef_xyzrpy_deg'][5]:.1f}",
            # )
            # env.viewer_overlay(
            #     loc="top left",
            #     text1="qpos",
            #     text2=f"{result['qpos'][0]:.5f}, {result['qpos'][1]:.5f}, {result['qpos'][2]:.5f}, {result['qpos'][3]:.1f}, {result['qpos'][4]:.1f}, {result['qpos'][5]:.1f}",
            # )
            env.render()

            # tick += 1
            sleep(0.01)
    finally:
        if env.use_mujoco_viewer and env.is_viewer_alive():
            env.close_viewer()


if __name__ == "__main__":
    main()
