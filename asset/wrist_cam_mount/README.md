# Wrist camera mount (RealSense D405)

연구실에서 쓰는 손목 카메라 마운트(Onshape `Part Studio 1.stl`, 원본은 `source/`에 있음)로 만든 모델입니다.
실물과 같은 순서로 **UR5e 플랜지 → 마운트(어댑터 플레이트 10 mm) → Quick Changer → RG2**를 쌓았습니다.

- 씬: `../universal_robots_ur5e/scene_rg2_tabletop.xml` (테이블과 물체 포함, 테이블 윗면이 z = 0), 로봇 모델은 `ur5e_rg2_wristcam.xml`
- 카메라: `wrist_cam` (MuJoCo camera). 크래들 크기(42×42×23 mm)로 보아 RealSense D405로 가정하고 `fovy` = 58°(D405 세로 FOV)로 설정했습니다. 16:10(예: 640×400)으로 렌더링하면 가로 FOV가 약 87°가 됩니다. 이미지 위쪽이 툴 축 방향입니다(실물 D405 화면 방향에 맞춰 180° 돌림).
- 렌즈는 크래들 바닥면 법선 방향이고, 툴 축에서 31.6° 기울어져 있습니다. 카메라 위치와 방향은 `wrist_cam_params.json`에 있습니다.
- 회전 정렬: 마운트 플레이트의 핀 구멍 (0, +25) mm를 UR 플랜지 핀 (0, -24.2) mm에 맞추려고 Z축으로 180° 돌렸습니다. Quick Changer 핀까지 0.8 mm 안에서 한 줄로 맞습니다.
- 충돌: 카메라 크래들 부분만 convex hull로 넣었습니다. 플레이트는 플랜지와 체인저 사이에 끼워져 있어 충돌 형상을 넣지 않았습니다.

```python
renderer = mujoco.Renderer(model, 400, 640)
renderer.update_scene(data, camera="wrist_cam")
rgb = renderer.render()
```

다시 만들기: `python tools/build_wristcam.py source/wrist_cam_mount_onshape.stl` (필요한 패키지: trimesh, fast-simplification)
