import sys
from pathlib import Path

import mujoco
import numpy as np
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parents[1]  # UR5e_simulator
_PACKAGE_DIR = _ROOT_DIR / "package"

sys.path.append(str(_PACKAGE_DIR / "kinematics_helper"))
sys.path.append(str(_PACKAGE_DIR / "mujoco_helper"))

from ik import solve_ik  # noqa: E402
from mujoco_parser import MuJoCoParserClass  # noqa: E402
from transforms import r2rpy  # noqa: E402


class UR5eIKEnv:
    """Thin wrapper that provides reset-to-qpos and IK-based action for UR5e."""

    def __init__(
        self,
        scene_xml_path=None,
        joint_names=None,
        body_name_trgt="end_effector_target",
        name="UR5e-2F85",
        verbose=True,
    ):
        if scene_xml_path is None:
            scene_xml_path = _ROOT_DIR / "asset" / "universal_robots_ur5e" / "scene_2f85.xml"

        if joint_names is None:
            joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]

        self.joint_names = list(joint_names)
        self.body_name_trgt = body_name_trgt
        self.scene_xml_path = str(Path(scene_xml_path).resolve())

        self.env = MuJoCoParserClass(
            name=name,
            rel_xml_path=self.scene_xml_path,
            verbose=verbose,
        )

        self._idx_fwd = self.env.get_idxs_fwd(joint_names=self.joint_names)
        self._idx_ctrl = self.env.get_idxs_step(joint_names=self.joint_names)

    def _get_eef_pose_from_q(self, q):
        """Return end-effector pose (xyzrpy) for a given joint vector q."""
        self.env.store_state()
        self.env.forward(q=q, joint_idxs=self._idx_fwd, increase_tick=False)
        p_eef, R_eef = self.env.get_pR_body(body_name=self.body_name_trgt)
        rpy_rad = r2rpy(R_eef, unit="rad")
        self.env.restore_state()
        return p_eef, rpy_rad

    def reset(self, qpos, zero_vel=True, sync_ctrl=True, step=False):
        """Reset env and initialize robot joints to the given qpos."""
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.shape[0] != len(self.joint_names):
            raise ValueError(
                f"qpos length mismatch: got {qpos.shape[0]}, expected {len(self.joint_names)}"
            )

        self.env.reset(step=step)
        self.env.data.qpos[self._idx_fwd] = qpos
        if zero_vel:
            self.env.data.qvel[:] = 0.0
        if sync_ctrl:
            self.env.data.ctrl[self._idx_ctrl] = qpos
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env.get_qpos_joints(self.joint_names)

    def action(
        self,
        p_trgt,
        R_trgt=None,
        q_init=None,
        apply=True,
        max_ik_tick=100,
        ik_err_th=1e-3,
        ik_stepsize=1.0,
        ik_eps=1e-2,
        ik_th=np.radians(1.0),
        step_after=False,
        nstep=1,
        verbose=False,
    ):
        """Run IK from q_init (or current qpos) to target and optionally apply result."""
        p_trgt = np.asarray(p_trgt, dtype=float).reshape(3)
        if R_trgt is not None:
            R_trgt = np.asarray(R_trgt, dtype=float).reshape(3, 3)

        if q_init is None:
            q_init = self.env.get_qpos_joints(self.joint_names)
        q_init = np.asarray(q_init, dtype=float).reshape(-1)

        q_sol, ik_err_stack, _ = solve_ik(
            env=self.env,
            joint_names_for_ik=self.joint_names,
            body_name_trgt=self.body_name_trgt,
            q_init=q_init,
            p_trgt=p_trgt,
            R_trgt=R_trgt,
            max_ik_tick=max_ik_tick,
            ik_err_th=ik_err_th,
            restore_state=not apply,
            ik_stepsize=ik_stepsize,
            ik_eps=ik_eps,
            ik_th=ik_th,
            verbose=verbose,
            verbose_warning=verbose,
            reset_env=False,
            render=False,
        )

        if apply:
            self.env.forward(q=q_sol, joint_idxs=self._idx_fwd, increase_tick=False)
            self.env.data.ctrl[self._idx_ctrl] = q_sol
            if step_after:
                self.env.step(nstep=nstep)
            p_eef, R_eef = self.env.get_pR_body(body_name=self.body_name_trgt)
            rpy_rad = r2rpy(R_eef, unit="rad")
        else:
            p_eef, rpy_rad = self._get_eef_pose_from_q(q_sol)

        xyzrpy_rad = np.concatenate([p_eef, rpy_rad])
        xyzrpy_deg = np.concatenate([p_eef, np.degrees(rpy_rad)])

        return {
            "qpos": q_sol.copy(),
            "ik_err_max": float(np.abs(ik_err_stack).max()),
            "ik_err_norm": float(np.linalg.norm(ik_err_stack)),
            "ik_err_stack": ik_err_stack.copy(),
            "eef_xyzrpy_rad": xyzrpy_rad.copy(),  # [x,y,z,roll,pitch,yaw]
            "eef_xyzrpy_deg": xyzrpy_deg.copy(),  # [x,y,z,roll,pitch,yaw]
        }


if __name__ == "__main__":
    robot = UR5eIKEnv(verbose=True)

    q0 = np.array([-0.817, -0.691, 0.66, -1.63, -1.57, -3.2])
    robot.reset(qpos=q0)

    p_now = robot.env.get_p_body(body_name=robot.body_name_trgt)
    result = robot.action(
        p_trgt=p_now + np.array([0.02, 0.00, 0.00]),
        R_trgt=None,
        q_init=q0,
        apply=True,
        max_ik_tick=200,
        ik_err_th=1e-3,
    )

    print("IK done")
    print("qpos:", np.round(result["qpos"], 4))
    print("ik_err_max:", result["ik_err_max"])
