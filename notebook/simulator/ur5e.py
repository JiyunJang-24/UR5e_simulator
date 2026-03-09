from time import sleep

import numpy as np

from ur5e_ik_env import UR5eIKEnv
from transforms import rpy2r


def main():
    robot = UR5eIKEnv(verbose=True)
    q0 = np.array([-0.17189008394350225, -1.3012673419765015, 2.14858323732485, 3.9493061739155273, -1.5685656706439417, -0.1365898291217249])
    robot.reset(qpos=q0)

    env = robot.env
    env.init_viewer(
        title="UR5e IK",
        azimuth=170,
        distance=1.8,
        elevation=-20,
        lookat=[0.35, 0.0, 0.4],
    )

    p_center = env.get_p_body(body_name=robot.body_name_trgt).copy()
    R_center = env.get_R_body(body_name=robot.body_name_trgt).copy()
    dx = 0.06
    period = 160
    tick = 0

    try:
        while env.is_viewer_alive():
            phase = (tick % period) / period
            offset = dx * np.sin(2.0 * np.pi * phase)
            p_trgt = p_center + np.array([offset, 0.0, 0.0])
            yaw = np.radians(20.0) * np.sin(2.0 * np.pi * phase)
            R_trgt = R_center @ rpy2r(np.array([0.0, 0.0, yaw]))

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
            env.plot_sphere(p=p_trgt, r=0.012, rgba=[1.0, 0.2, 0.2, 0.8])
            env.plot_T(
                p=p_trgt,
                R=R_trgt,
                plot_axis=True,
                axis_len=0.12,
                axis_width=0.004,
                label="Target",
            )
            env.plot_T(
                p=np.array([0.0, 0.0, 0.0]),
                R=np.eye(3),
                plot_axis=True,
                axis_len=0.45,
                axis_width=0.006,
                label="World",
            )
            env.viewer_overlay(
                loc="top left",
                text1="IK err max",
                text2=f"{result['ik_err_max']:.5f}",
            )
            env.viewer_overlay(
                loc="top left",
                text1="EEF_xyzrpy_deg",
                text2=f"{result['eef_xyzrpy_deg'][0]:.5f}, {result['eef_xyzrpy_deg'][1]:.5f}, {result['eef_xyzrpy_deg'][2]:.5f}, {result['eef_xyzrpy_deg'][3]:.1f}, {result['eef_xyzrpy_deg'][4]:.1f}, {result['eef_xyzrpy_deg'][5]:.1f}",
            )
            env.viewer_overlay(
                loc="top left",
                text1="qpos",
                text2=f"{result['qpos'][0]:.5f}, {result['qpos'][1]:.5f}, {result['qpos'][2]:.5f}, {result['qpos'][3]:.1f}, {result['qpos'][4]:.1f}, {result['qpos'][5]:.1f}",
            )
            env.render()

            tick += 1
            sleep(0.01)
    finally:
        if env.use_mujoco_viewer and env.is_viewer_alive():
            env.close_viewer()


if __name__ == "__main__":
    main()
