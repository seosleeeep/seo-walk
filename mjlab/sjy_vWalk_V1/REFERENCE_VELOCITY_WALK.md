# K1 20 DOF Velocity Walk

JY_train `scratch_symmetry`의 활성 MDP를 MJLab으로 이식한 보행 환경입니다.
다리 12관절에 팔 8관절을 추가하고 목 2관절은 고정합니다. `velocity_walk`는
`vx`, `vy`, yaw-rate 명령을 추종하는 보행 task이며, 관절은 위치 목표로 제어합니다.

## 실행

기존 `inha_mjlab` 환경을 활성화하고 실행합니다. Task 등록은 패키지의 기존
`mjlab.tasks` entry point를 사용하므로 새 entry point 설치는 필요하지 않습니다.

```bash
conda activate inha_mjlab
list_envs
train Mjlab-Velocity-Walk-Flat-INHA-K1 --gpu-ids 0
```

짧은 GPU 학습 검사에서는 환경 수와 iteration만 줄일 수 있습니다.

```bash
train Mjlab-Velocity-Walk-Flat-INHA-K1 \
  --gpu-ids 0 \
  --env.scene.num-envs 32 \
  --agent.max-iterations 1

play Mjlab-Velocity-Walk-Flat-INHA-K1 --agent zero
play Mjlab-Velocity-Walk-Flat-INHA-K1 \
  --agent trained --checkpoint-file /absolute/path/to/model.pt --device cuda:0
```

기본 로그는 `logs/rsl_rl/inha_k1_velocity_walk/`에 저장됩니다. 새 정책의 입력과
출력이 달라졌으므로 기존 12 DOF JY checkpoint를 그대로 불러오는 것은 지원하지
않습니다. 이 task에서 저장한 checkpoint는 일반 MJLab resume 경로를 사용하며
command curriculum의 `common_step_counter`도 저장·복원합니다.

### Play 중 터미널 속도 명령

Play는 `(0, 0, 0)` 정지 명령으로 시작합니다. 실행한 터미널에
`vx vy yaw_rate` 세 숫자를 입력하고 Enter를 누르면 다음 정책 step에 반영됩니다.
단위는 `m/s`, `m/s`, `rad/s`이며 입력 예시는 다음과 같습니다.

```text
1.5 0 0
0 0.3 0
0 0 0.5
0 0 0
```

마지막 명령은 새 입력 전까지 유지하며 자동 command resample이나 episode reset으로
덮어쓰지 않습니다. 크기가 `0.05` 미만인 명령은 Booster Deploy처럼 정지로 처리해
주파수와 phase를 0으로 만듭니다. 이동 시 주파수는 기존 `1.5 Hz`입니다.
입력 대기는 시뮬레이션을 멈추지 않습니다. 속도 명령을 추종하는 보행을 보려면
학습된 checkpoint로 실행해야 하며, `--agent zero`는 시뮬레이션 동작 검사에 사용합니다.

## 로봇·모터 설정

보행용 설정의 기준 파일은 `robots/k1/velocity_walk.py`입니다.
기존 K1 MJCF의 mesh·관성·몸체를 재사용하고, 보행용 MjSpec에서 목 joint만 제거해
머리 질량과 형상을 유지합니다. Free base를 제외한 가동 관절은 20개입니다.

관절 순서는 기존 `K1_JOINT_ORDER`에서 `Head_Yaw`, `Head_Pitch`를 제외합니다.
왼팔 4, 오른팔 4, 왼다리 6, 오른다리 6 순서이며 관측과 action도 이를 따릅니다.

| 그룹 | kp | kd | 정지 시 torque cap | command delay |
|---|---:|---:|---:|---|
| Hip Pitch | 80 | 2 | 40 Nm | 전 관절 공통 0..18 step |
| Hip Roll | 80 | 2 | 50 Nm | 전 관절 공통 0..18 step |
| Hip Yaw | 80 | 2 | 36 Nm | 전 관절 공통 0..18 step |
| Knee Pitch | 80 | 2 | 56 Nm | 전 관절 공통 0..18 step |
| Ankle Pitch/Roll | 30 | 2 | 36 Nm | 전 관절 공통 0..18 step |
| 팔 8관절 | 10 | 1 | 14 Nm | 전 관절 공통 0..18 step |

다리 gain은 JY 원본 설정이고 torque cap은 사용자 요청에 따라
recovery/get-up과 동일하게 맞췄습니다. 해당 값은 `velocity_walk.py`의
`WALK_*_EFFORT_LIMIT` 상수에 정의합니다. JY 원본에서 고정했던 팔은 합의한 초기안에 따라
현재 INHA 팔 gain·HOME을 사용합니다. 모터 하드웨어 값은 기존
`motor_parameters.py`를 참조하며, 보행용 controller override는 이 파일에서만
정의합니다. Joint gain은 reset 시 원본과 동일하게 0.9..1.1배로 랜덤화합니다.

명령 지연은 사용자 요청에 따라 팔·다리를 구분하지 않고 20관절 전체에 같은 값을
적용합니다. `WALK_MIN_DELAY=0`, `WALK_MAX_DELAY=18`로 기존 leg 범위를 사용하며,
로봇마다 reset에서 한 번 샘플링한 뒤 episode 동안 유지합니다. 현재 physics step
2 ms 기준으로 0..36 ms입니다. 관절 상태 피드백에는 지연을 추가하지 않습니다.

500 Hz explicit PD로 계산한 torque에는 `min(모터 물리 곡선, controller cap)`을
적용합니다. 물리 곡선은 knee-point speed까지 하드웨어 최대 토크를 유지하고
no-load speed에서 0까지 선형 감소합니다. 최종 상한 곡선은 recovery/get-up과
동일하며, 정책 action clipping과 별개입니다.

## 정책 입출력

```text
q_target = q_default_per_env + action * 1.0
```

20차원 raw action과 목표 관절각을 clip하지 않습니다. Action offset은 매 정책
step에서 읽으므로 startup의 기본 관절각 랜덤화와 상대 관측 기준이 일치합니다.
기본 관절각은 INHA HOME, root 초기 높이는 JY의 0.57 m입니다.

팔 HOME은 양쪽 `Elbow_Pitch=0`, `Left_Elbow_Yaw=-0.4`,
`Right_Elbow_Yaw=+0.4 rad`입니다. 모델의 `Elbow_Pitch`는 하드웨어 그림의
4/8번 Shoulder Yaw, `Elbow_Yaw`는 5/9번 팔꿈치 굽힘 관절입니다.
이전 Elbow Pitch 기준 HOME으로 학습한 checkpoint는 관측 기준과 action offset이
다르므로 수정된 HOME 기준으로 재학습해야 합니다.

| 관측 | Actor slice | 차원 |
|---|---|---:|
| Command `[vx, vy, yaw_rate, gait_frequency]` | 0:4 | 4 |
| Gait cos/sin | 4:6 | 2 |
| Projected gravity | 6:9 | 3 |
| Base angular velocity | 9:12 | 3 |
| Joint position relative to default | 12:32 | 20 |
| Joint velocity relative to default | 32:52 | 20 |
| Previous action | 52:72 | 20 |

Critic은 위 관측을 noise 없이 사용하고 COM·질량 4, COM 선속도 3, 지면 기준 높이 1,
몸체 좌표계의 외력 3·외부 토크 3을 추가합니다. 최종 차원은 actor 72, critic 86,
action 20입니다. Layout 상수는 `config/k1/policy_contract.py`에 있습니다.
기존 command의 foot yaw·body pitch/roll·feet offset 여섯 채널과 관련 샘플링
설정을 제거했습니다. 발 간격 보상의 목표 offset은 기존과 같은 0으로 고정하며,
좌우 발 기준 간격 `0.18 m`는 유지합니다. 입력 차원이 달라 기존 78/92차원
checkpoint를 그대로 불러올 수 없습니다.

## MDP·학습 기준

- 물리 step 0.002 s, decimation 10, 정책 50 Hz, episode 30 s, 환경 4096개.
- JY 보행 보상을 기준으로 하며 base height는 고정 목표, 팔 자세는 Booster 기준을 사용합니다.
- Joint torque·power·속도·가속도·한계·action rate 비용은 20관절 합산입니다.
  12/20 보정은 적용하지 않습니다.
- `arm_posture`는 팔 8관절의 환경별 기본 자세 이탈 비용으로,
  `mean(((q_arm - q_default_arm) / std)^2)`, weight `-0.1`입니다.
  Booster처럼 명령값 `s = norm([vx, vy]) + abs(yaw_rate)`로 구간을 선택합니다.
  정지(`s < 0.05`) std는 모든 팔 관절 `0.05`, 보행(`0.05 <= s < 1.0`)은
  `0.15`, 달리기(`s >= 1.0`)도 모든 팔 관절 `0.15` rad입니다.
  사용자 설정으로 고속에서도 보행보다 자세 패널티를 완화하지 않습니다.
  std는 관절 제한이 아닌 비용 정규화 기준입니다.
  목은 고정이므로 대상에서 제외합니다. 전 관절 `standing_pose_l1`은 추가하지 않습니다.
- Command 4차원, 정지 비율 10%, gait frequency 1.5 Hz, 원본 속도 샘플링 범위 유지.
- Base height 목표는 정지·보행 모두 지면 기준 `0.52 m`로 고정합니다.
  비용은 `(root_z - terrain_height - 0.52)^2`, weight는 `-20.0`입니다.
  속도 명령·실제 속도에 따른 목표 높이 감소, deadzone과 보간은 제거했습니다.
- 활성 curriculum 경계가 모두 500이므로 0..499 iteration은 단일 축,
  500..999는 1/2/3축 비율 0.3/0.4/0.3, 이후는 0.1/0.3/0.6입니다.
  비율은 moving command 기준이며 원본의 건너뛰는 중간 분기도 그대로 둡니다.
- Train은 기존 plane·접촉 랜덤화를 유지합니다. Play는 Booster Deploy 기준의
  plane·고정 접촉 재질을 사용하며 kick/push를 끕니다. 속도 명령은 터미널로 입력합니다.
- PPO 은닉층 `(512,256,128)`, ELU, observation normalization, rollout 24,
  epoch 5, minibatch 4, 초기 표준편차 0.15, learning rate 2.5e-6 등은 JY 기준입니다.
- Action clipping과 symmetry loss/augmentation은 구성하지 않습니다.
  PPO ratio clipping `0.2`는 기존 PPO 알고리즘 설정으로 유지합니다.

## 프레임워크 이식 시 처리한 차이

Isaac Lab의 legacy 속도 필드는 COM 기준이고 위치·자세는 link 기준입니다.
MJLab에서 이를 명시적으로 구분합니다. Root acceleration은 MuJoCo frame sensor의
중력 보상을 제거해 원본의 관성 가속도 비용으로 계산합니다.

Contact history는 JY의 `[env,history,body,3]`와 MJLab의
`[env,body,history,3]` 차이를 반영했습니다. 원본처럼 self-contact도 포함합니다.
높이 스캐너의 20 m offset은 yaw frame에 두어 몸통 tilt가 ray 시작점을 기울이지
않도록 했습니다.

학습 접촉 설정은 `booster_mjlab`의 `FULL_COLLISION`과 `terrain_collisions()`를
따릅니다. 로봇·지면 모두 `priority=0`, `condim=3`이며 자기충돌을 유지합니다.
지면 마찰은 0이므로 접촉 마찰에는 로봇의 값이 반영됩니다. 로봇의 기본 sliding
friction은 발 1.0, 나머지 0.45이지만, startup에서 모든 collision geom에 JY의
dynamic friction 범위 `0.1~2.0`을 64개 bucket으로 샘플링해 덮어씁니다.

로봇의 `solref=(0.01, 1.0)`, `solmix=0.001`, 지면의 `solmix=1.0`도 Booster와
같습니다. 지면의 startup 랜덤화는 Booster velocity task의 `solref` 범위
`{0: (0.006, 0.03), 1: (0.95, 1.05)}`와 `solimp` 범위
`{0: (0.88, 0.92), 1: (0.94, 0.99), 2: (0.003, 0.01)}`를 사용합니다.
같은 환경의 지면 geom들은 같은 표본을 공유합니다. 이 랜덤화는 Train에 적용합니다.
PhysX의 별도 static friction 표본과 restitution을 MuJoCo damping으로 바꾸던
근사식은 제거했습니다.

Play는 `booster_deploy`가 참조하는 공식
[`K1_22dof.xml`](https://github.com/BoosterRobotics/booster_assets/blob/main/robots/K1/K1_22dof.xml)의
지면·접촉 재질을 사용합니다. 바닥 마찰 `(0.4, 0.005, 0.0001)`, 로봇 collision
마찰 `(1.0, 0.005, 0.0001)`, 양쪽 `solref=(0.001, 1.0)`,
`solimp=(0.90, 0.95, 0.001, 0.5, 2.0)`, `solmix=1.0`, `priority=0`입니다.
바닥 `condim=1`과 로봇 `condim=3`의 조합으로 실제 발 접촉은 `condim=3`,
sliding friction `max(0.4, 1.0)=1.0`이 됩니다. Play에서는 관측 노이즈와
마찰·접촉·질량·COM·기본 관절각·gain 랜덤화를 끄고 인위적인 명령 지연을 0으로
설정합니다. Reset은 기본 자세와 0 속도로 실행합니다. 학습 설정은 유지합니다.
`0 0 0`은 정지 속도 명령이며, 학습된 정책은 균형을 유지하기 위한 관절 목표를
계속 계산합니다. 정책 출력을 강제로 0으로 고정하지 않습니다.

원본 외력 종료 코드의 advanced-index `.zero_()`는 원본 버퍼를 수정하지 못할 수
있어 indexed assignment로 고쳤습니다. 외력은 지정한 1초 뒤 해제하고 reset된
환경에서도 지웁니다. 몸체 로컬 외력은 매 물리 step에서 world로 회전해 적용합니다.

## 검사

```bash
pytest -q tests/test_velocity_walk.py
ruff check src/inha_mjlab/robots/k1/velocity_walk.py src/inha_mjlab/tasks/velocity_walk tests/test_velocity_walk.py
```

CPU에서 20관절 모델 compile, action/관측 계약, command/reset/외력, 29개 보상,
실제 MJLab 환경 step 및 PPO 1회 update와 checkpoint 저장·복원을 확인했습니다.
대규모 CUDA 학습과 보행 성능 평가는 별도로 실행해야 합니다.