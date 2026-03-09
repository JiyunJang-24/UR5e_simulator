from time import sleep
import argparse

import numpy as np
import asyncio

from ur5e_ik_env import UR5eIKEnv
from transforms import rpy2r, r2w

def interpolate(start, end, steps):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for i in range(1, steps + 1):
        alpha = i / steps
        yield (1 - alpha) * start + alpha * end

def get_actual_q_from_ur(ur_ip: str):
    """Read current 6-DoF joint state from real UR via RTDE receive."""
    try:
        import rtde_receive
    except ImportError as exc:
        raise RuntimeError(
            "rtde_receive import failed. Install ur_rtde in the running environment."
        ) from exc

    rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)
    try:
        q = np.array(rtde_r.getActualQ(), dtype=float)
    finally:
        try:
            rtde_r.disconnect()
        except Exception:
            pass

    if q.shape[0] != 6:
        raise RuntimeError(f"Unexpected UR q length: {q.shape[0]} (expected 6)")
    return q


def xyzrpy_to_pose6d(xyzrpy_rad: np.ndarray):
    """Convert [x,y,z,roll,pitch,yaw] to UR moveL pose [x,y,z,rx,ry,rz]."""
    xyz = xyzrpy_rad[:3]
    rpy = xyzrpy_rad[3:]
    R = rpy2r(rpy)
    rotvec = r2w(R)
    return np.concatenate([xyz, rotvec])


def run_sim_with_q(
    q0: np.ndarray,
    use_realworld: bool = False,
    ur_ip: str = "192.168.0.251",
    movel_speed: float = 0.15,
    movel_acc: float = 0.3,
    rtde_send_every: int = 5,
):
    robot = UR5eIKEnv(verbose=True)
    robot.reset(qpos=q0)

    rtde_c = None
    if use_realworld:
        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "rtde_control import failed. Install ur_rtde in the running environment."
            ) from exc
        rtde_c = rtde_control.RTDEControlInterface(ur_ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)
        print(f"[INFO] Real-world control enabled: moveL to {ur_ip}")

    env = robot.env
    env.init_viewer(
        title="UR5e IK (Real-Q Init)",
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

            if rtde_c is not None and (tick % max(1, rtde_send_every) == 0):
                target_pose6d = xyzrpy_to_pose6d(result["eef_xyzrpy_rad"])
                current_pose6d = np.asarray(rtde_r.getActualTCPPose(), dtype=float)
                for p in interpolate(current_pose6d, target_pose6d, steps=10):
                    ok = rtde_c.moveL(
                        p.tolist(),
                        movel_speed,
                        movel_acc,
                        True,
                    )
                # target_q = result["qpos"]
                # ok = rtde_c.moveJ(
                #     target_q.tolist(),
                #     movel_speed,
                #     movel_acc,
                #     True,
                # )

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
                text2=(
                    f"{result['eef_xyzrpy_deg'][0]:.5f}, {result['eef_xyzrpy_deg'][1]:.5f}, "
                    f"{result['eef_xyzrpy_deg'][2]:.5f}, {result['eef_xyzrpy_deg'][3]:.1f}, "
                    f"{result['eef_xyzrpy_deg'][4]:.1f}, {result['eef_xyzrpy_deg'][5]:.1f}"
                ),
            )
            env.viewer_overlay(
                loc="top left",
                text1="qpos",
                text2=(
                    f"{result['qpos'][0]:.5f}, {result['qpos'][1]:.5f}, {result['qpos'][2]:.5f}, "
                    f"{result['qpos'][3]:.1f}, {result['qpos'][4]:.1f}, {result['qpos'][5]:.1f}"
                ),
            )
            env.render()

            tick += 1
            sleep(0.01)
    finally:
        if rtde_c is not None:
            try:
                rtde_c.stopL(2.0)
            except Exception:
                pass
            try:
                rtde_c.stopScript()
            except Exception:
                pass
            try:
                rtde_c.disconnect()
            except Exception:
                pass
        if env.use_mujoco_viewer and env.is_viewer_alive():
            env.close_viewer()


def parse_args():
    parser = argparse.ArgumentParser(description="Run UR5e sim initialized from real UR joint state")
    parser.add_argument("--ur-ip", default="192.168.0.251", help="Real UR controller IP")
    parser.add_argument(
        "--use-realworld",
        action="store_true",
        help="If set, send IK result to real UR via RTDE moveL",
    )
    parser.add_argument("--movel-speed", type=float, default=0.15, help="RTDE moveL speed (m/s)")
    parser.add_argument("--movel-acc", type=float, default=0.3, help="RTDE moveL acceleration (m/s^2)")
    parser.add_argument(
        "--rtde-send-every",
        type=int,
        default=5,
        help="Send moveL command every N simulation ticks",
    )
    parser.add_argument(
        "--fallback-q",
        default="-0.17,-1.30,2.15,3.95,-1.57,-0.14",
        help="Comma-separated fallback q (used when RTDE read fails)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        q0 = get_actual_q_from_ur(args.ur_ip)
        print(f"[INFO] Read real UR q: {np.array2string(q0, precision=5)}")
    except Exception as exc:
        print(f"[WARN] Failed to read real UR q from {args.ur_ip}: {exc}")
        q0 = np.array([float(x.strip()) for x in args.fallback_q.split(",")], dtype=float)
        print(f"[INFO] Using fallback q: {np.array2string(q0, precision=5)}")

    run_sim_with_q(
        q0=q0,
        use_realworld=args.use_realworld,
        ur_ip=args.ur_ip,
        movel_speed=args.movel_speed,
        movel_acc=args.movel_acc,
        rtde_send_every=args.rtde_send_every,
    )


if __name__ == "__main__":
    main()
