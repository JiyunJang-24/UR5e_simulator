# OnRobot Quick Changer for I/O + RG2 (MuJoCo)

OnRobot이 배포한 STEP 파일(`Quick_Changer_for_IO_v2.STEP`, `RG2_v2.STEP`)로 만든 MuJoCo 모델입니다.

- `rg2.xml`: Quick Changer(로봇쪽)와 RG2를 합친 독립 모델. `base_mount` 프레임이 UR 툴 플랜지와 같습니다 (z축이 플랜지 바깥 방향, 손가락은 y축으로 움직임).
- `../universal_robots_ur5e/ur5e_rg2.xml`, `scene_rg2.xml`: UR5e에 붙인 버전 (`ur5e_2f85.xml`과 같은 방식으로 `<attach>`, 이름 앞에 `rg2_`가 붙음).
- `rg2_utils.py`: 열림 폭 [m]과 액추에이터 목표값을 서로 바꿔 주는 함수.

## 제어

액추에이터 `rg2_fingers`는 바깥 암 각도 q [rad]를 목표로 하는 위치 서보입니다. **q = 0이면 완전히 열린 상태(~106 mm)이고, q = 1.5013이면 닫힌 상태입니다.**
RG2는 평행사변형 링크라서 열림 폭이 q에 비례하지 않으므로 아래처럼 변환해서 쓰세요.

```python
from rg2_utils import width_to_ctrl, ctrl_to_width
data.actuator("rg2_fingers").ctrl = width_to_ctrl(0.05)   # 50 mm로 열기
```

- 손가락 패드 중심 사이트: `rg2_right_tip_pad_center`, `rg2_left_tip_pad_center`
- `rg2_tcp` 및 `end_effector_target`: 닫혔을 때 패드 중심 위치 (플랜지에서 0.226 m). 손끝이 열린 상태에서 닫힌 상태로 가면서 툴 축 방향으로 약 5 cm 움직입니다(`rg2_utils.pad_depth(q)`).
- 그립 힘: forcerange를 패드 한쪽당 약 40 N(RG2 최대치)이 나오도록 맞췄습니다.

## 모델링 메모

- 바깥 암, 안쪽 링크, 손끝을 각각 hinge로 만들고, `<equality joint>`로 모두 `right_arm_joint`에 묶었습니다. 그래서 평행사변형 기구학이 정확히 재현되고 손끝은 항상 평행을 유지합니다.
- 충돌은 각 링크의 convex hull로 처리하고, 패드는 따로 hull을 만들어 마찰 1.0을 줬습니다. 시각용 메시는 데시메이션했고, 툴체인저 안쪽에 가려 보이지 않는 52핀 I/O 접점 블록은 뺐습니다.
- 질량은 근사치입니다 (베이스+체인저 0.72 kg, 링크 15~35 g).
- 최대 열림은 CAD 형상 기준으로 106 mm입니다 (공식 스펙은 110 mm).
- 플랜지 축을 기준으로 한 그리퍼의 회전 방향(손가락이 플랜지 y축으로 움직이게 함)은 임의로 정한 값입니다. 실제 장착 방향과 다르면 `ur5e_rg2.xml`의 `attachment` body `quat`을 z축 기준으로 돌리면 됩니다.

## 다시 만들기

```bash
python tools/step_explode.py Quick_Changer_for_IO_v2.STEP /tmp/qc     # needs cadquery-ocp, trimesh
python tools/step_explode.py RG2_v2.STEP /tmp/rg2
python tools/build_rg2.py /tmp/qc /tmp/rg2                             # needs fast-simplification, scipy
```
