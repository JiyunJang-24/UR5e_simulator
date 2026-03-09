import asyncio
import rtde_control
import rtde_receive
import sys
sys.path.append("/Users/jangjiyun/Desktop/JY/code/UR5e_simulator")
from onrobot_rg.src.onrobot import RG
from scipy.spatial.transform import Rotation as R
import numpy as np
# UR5 IP 주소 설정
UR5_IP = "192.168.0.251"
RG2_IP = "192.168.0.72"
RG2_PORT = 502

# RTDE Control 및 Receive 인터페이스 초기화
rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
rg = RG('rg2', RG2_IP, RG2_PORT)

# 상태 변수
current_pose = None
current_Q = None
current_force = None
current_velocity = None
current_gripper_status = None

def compute_gripper_tip_pose(eef_pose, gripper_length=0.23):
    """
    eef_pose: list or np.ndarray of shape (6,) — [x, y, z, rx, ry, rz]
              where r* are axis-angle rotation vector
    gripper_length: distance from EEF to tip in meters
    Returns:
        tip_position: np.ndarray (3,) — [x, y, z] of the gripper tip
        tip_euler: np.ndarray (3,) — [roll, pitch, yaw] in radians
    """
    position = np.array(eef_pose[:3])  # x, y, z
    rotvec = np.array(eef_pose[3:])    # rx, ry, rz

    # Rotation: axis-angle → rotation matrix
    rotation = R.from_rotvec(rotvec)
    
    # Gripper offset in EEF frame (assuming forward along Z)
    gripper_offset = np.array([0, 0, gripper_length])
    
    # Apply rotation to the offset, then translate
    tip_position = position + rotation.apply(gripper_offset)

    # Get Euler angles from rotation
    tip_euler = rotation.as_euler('xyz', degrees=False)
    
    return np.concatenate((tip_position, tip_euler))

def pose_to_tip_pose_rv(rotvec_pose, gripper_length=0.23):
    """
    Converts UR5e EEF pose (pos + rotvec) to tip pose (pos + rotvec)
    Args:
        eef_pose: [x, y, z, rx, ry, rz]
        gripper_length: scalar (in meters)
    Returns:
        tip_pose: [tip_x, tip_y, tip_z, tip_rx, tip_ry, tip_rz]
    """
    position = np.array(rotvec_pose[:3])
    rotvec = np.array(rotvec_pose[3:])
    rotation = R.from_rotvec(rotvec)

    # Gripper offset along local z-axis
    offset = np.array([0, 0, gripper_length])
    rotated_offset = rotation.apply(offset)

    tip_position = position + rotated_offset
    tip_rotvec = rotvec  # Orientation is same as EEF

    tip_pose = np.concatenate([tip_position, tip_rotvec])
    return tip_pose

async def receive_robot_state(frequency=100):
    """RTDE Receive: 로봇 상태를 지속적으로 비동기 수신"""
    global current_pose, current_force, current_velocity, current_Q
    dt = 1 / frequency

    while True:
        current_pose = rtde_r.getActualTCPPose()
        current_Q = rtde_r.getActualQ()
        # current_pose = pose_to_tip_pose_rv(current_pose)
        current_force = rtde_r.getActualTCPForce()
        current_velocity = rtde_r.getActualTCPSpeed()
        await asyncio.sleep(dt)

async def control_robot(frequency=1):
    """RTDE Control: 로봇을 지속적으로 비동기 제어"""
    global current_pose
    dt = 1 / frequency
    while True:
        if current_pose:
            print(f"[Control] Current Pose: {current_pose}")
            # print(f"[Control] Current Force: {current_force}")
            # print(f"[Control] Current Velocity: {current_velocity}")
        # [0.17919724976642143, -0.5828446000209568, 0.2940166234704726, -3.128745750991788, 0.24643694191444748, -0.059692257337022527]
        # 로봇 제어 명령 (비동기)
        rtde_c.moveL(
            [0.17919724976642143, -0.5828446000209568, 0.2940166234704726, -3.128745750991788, 0.24643694191444748, -0.059692257337022527], 
            0.1, 0.1
        )
        # 이동 명령 (제어는 안전성을 위해 직접적으로 실행)
        rtde_c.moveL([0.20919724976642143, -0.6028446000209568, 0.2940166234704726, -3.128745750991788, 0.24643694191444748, -0.059692257337022527], 0.1, 0.1)
        await asyncio.sleep(dt)
        rtde_c.moveL([0.17919724976642143, -0.5828446000209568, 0.2940166234704726, -3.128745750991788, 0.24643694191444748, -0.059692257337022527], 0.1, 0.1)
        await asyncio.sleep(dt)

async def control_gripper(frequency=0.5):
    """Gripper Control: 그리퍼를 비동기적으로 제어"""
    dt = 1 / frequency
    while True:
        rg.close_gripper(force_val=400)
        await asyncio.sleep(dt)
        rg.open_gripper()
        await asyncio.sleep(dt)

async def monitor_status():
    """현재 상태 모니터링"""
    while True:
        if current_pose is not None:
            # current_pose_euler = compute_gripper_tip_pose(current_pose, 0.23)
            print(f"[Monitor] Pose: {current_pose} | Force: {current_force} | Velocity: {current_velocity}")
            print(f"[Monitor] Q: {current_Q}")
            # print(f"[Monitor] Euler Pose: {current_pose_euler} | Force: {current_force} | Velocity: {current_velocity}")
        await asyncio.sleep(0.5)

async def main():
    # 비동기적으로 모든 작업을 병렬로 실행
    await asyncio.gather(
        receive_robot_state(frequency=50),   # 50Hz로 상태 수신
        # control_robot(frequency=1),           # 1Hz로 제어 명령 전송
        # control_gripper(frequency=0.5),       # 0.5Hz로 그리퍼 제어
        monitor_status()                      # 상태 모니터링
    )

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("프로그램 종료 중...")
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("RTDE 인터페이스 안전하게 종료됨.")