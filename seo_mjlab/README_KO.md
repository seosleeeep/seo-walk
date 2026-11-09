# Seo K1 vv0: mjlab 독립 학습 패키지

2026-11-07 작성. 사용자 vv0를 공식 mjlab 1.6.0 API로 이식한 첫 버전이다.
LSH_mjlab/lsh_walk 및 Recovery 프로젝트의 task·보상·정책 코드는 사용하지 않았다.
입력 폴더의 공식 API 소스는 인터페이스 확인에만 사용했고, 라이브러리 대체본으로 배포하지 않는다.

## 팔 ㄴㅇㄱ 자세에 대한 변경

팔 8관절의 실제 관절 공간 모터 토크(qfrc_actuator)에만 다음 항을 추가했다.

```
r_arm_torque = -0.0002 * sum(tau_arm ** 2)
r_arm_deviation = -0.05 * mean((q_arm - q_arm_reference) ** 2)
```

기존 전 관절 torque 항도 -0.0002로 유지한다. 따라서 두 torque 항만 합치면 팔은
-0.0004 × 제곱 합, 다리는 -0.0002 × 제곱 합의 계수를 갖는다.
추가한 팔 항이 다리 관절 토크를 다시 계산하지 않도록 관절 이름으로 8개를 선택한다.
reward manager는 각 항에 정책 주기 dt=0.02초를 곱한다.

팔 기준 자세(rad)는 왼팔 [0, -1.35, 0, -0.5], 오른팔 [0, 1.35, 0, 0.5]다.
순서는 shoulder pitch, shoulder roll, elbow pitch, elbow yaw다.
deviation은 팔을 내린 기준에 가까워지도록 하고, torque 항은 불필요한 토크를 줄인다.
토크가 작다고 반드시 팔이 아래를 향하는 것은 아니므로 자세 해결을 보장하지 않는다.
새 reward로 학습해야 효과를 확인할 수 있다. 기존 체크포인트에는 소급 적용되지 않는다.

## 파일과 수정 위치

| 파일 | 역할 |
|---|---|
| src/seo_walk_mjlab/env_cfg.py | 환경, 관측, 이벤트, 종료 조건 |
| src/seo_walk_mjlab/reward_config.json | 보상 31개의 계수와 파라미터 |
| src/seo_walk_mjlab/mdp.py | 팔 보상, 관측, 외력 상태 |
| src/seo_walk_mjlab/legacy_rewards.py | 사용자 기존 보상 수식 |
| src/seo_walk_mjlab/bridge.py | MuJoCo 토크·접촉력과 기존 수식 연결 |
| src/seo_walk_mjlab/robot.py | 머리 고정, 모터, 명목 자세, 관절 순서 |
| src/seo_walk_mjlab/commands.py | 기존 단계별 속도 명령 샘플러 |
| src/seo_walk_mjlab/symmetry.py | 좌우 반사와 RSL 5.5 TensorDict 콜백 |
| src/seo_walk_mjlab/ppo_cfg.py | PPO 5 epoch, 4 minibatch 등 |
| scripts/preflight.py | GPU 사용 전 API·실제 K1 모델 검사 |
| scripts/smoke.py | GPU 16환경, 관측·보상·reset 검사 |
| scripts/train.py | 학습 |
| scripts/play.py | MuJoCo에서 직접 재생·영상 저장 |

task 이름은 `seo_walk_vv0_mjlab`이다. 제공 스크립트가 이 task를 내부 등록한다.
전역 train 명령에 다른 사람의 task 이름을 입력하지 않는다.
별도 패키지 설치 없이 scripts가 자신의 src를 연결하므로 공용 conda 환경을 재설치하지 않는다.
공식 mjlab의 외부 task 자동 등록은 이 프로세스에서만 필터해 다른 개인 task의 자동 import를 막는다.
설치된 라이브러리 파일과 다른 사람의 프로젝트는 수정하지 않는다.

## 원격 서버 실행 순서

ZIP을 서버 `/home/user/Downloads/seo_mjlab_vv0_bundle.zip`으로 보낸 뒤 실행한다.
같은 이름의 기존 개인 폴더가 있다면 먼저 백업한다. 이번 패키지는 `/home/user/seo`를 덮어쓰지 않는다.

### 1. 압축 해제와 환경

```bash
cd /home/user
unzip /home/user/Downloads/seo_mjlab_vv0_bundle.zip -d /home/user
source /home/user/miniconda3/bin/activate
conda activate inha_mjlab
cd /home/user/seo_mjlab
export SEO_K1_MESH_DIR="/home/user/booster_assets/robots/K1/meshes"
python scripts/preflight.py
```

`PASS`가 표시돼야 다음 단계로 간다. mesh가 없으면 오류에 표시된 파일 경로를 먼저 확인한다.
공용 환경의 mjlab/torch/rsl 패키지를 임의로 upgrade 또는 downgrade하지 않는다.

### 2. GPU 0번 짧은 검증

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -u scripts/smoke.py
```

출력 마지막에 `PASS: 20 actions / actor 78 / critic 92 ...`가 있어야 한다.
이는 넘어지지 않는 정책을 검증하는 명령이 아니다. 환경 reset, 보상 수치, 관측 및 관절 계약을 검사한다.
오류가 나면 장시간 학습을 시작하지 말고 traceback을 확인한다.

이어서 PPO 업데이트 자체도 2회만 검사한다.
아래는 공식 mjlab CLI의 GPU 선택 기능으로 물리 GPU 0번을 선택한다.

```bash
python -u scripts/train.py --gpu-ids 0 --env.scene.num-envs 16 --agent.max-iterations 2 --agent.run-name smoke_2iter
```

### 3. 본 학습

위 검사가 성공한 뒤 사용 가능한 GPU를 확인하고 실행한다.
새 엔진과 RSL 5.5 네트워크 형식으로 새로 학습하며 Isaac 체크포인트에서 resume하지 않는다.

```bash
python -u scripts/train.py --gpu-ids 0 --env.scene.num-envs 1024 --agent.max-iterations 10000 --agent.run-name vv0_arm_torque
```

기본 설정 파일의 상한은 기존과 같은 30000 iteration이다. 위 명령은 첫 실험을 10000으로 지정한 예다.
iteration당 PPO epoch는 5다. rollout은 환경당 24step, minibatch는 4다.
실제 총 시간은 첫 수십 iteration의 ETA로 판단한다. Isaac에서의 1.4초/iteration을 그대로 적용하지 않는다.
100 iteration마다 체크포인트를 저장한다.

로그는 실행 폴더 아래 `logs/rsl_rl/seo_walk_vv0_mjlab/날짜_실험명/`에 생긴다.
기본 로깅은 로컬 TensorBoard다. W&B 로그인이나 업로드는 자동으로 하지 않는다.

### 4. 체크포인트 찾기

```bash
find /home/user/seo_mjlab/logs/rsl_rl/seo_walk_vv0_mjlab -type f -name 'model_*.pt' | sort -V | tail -15
```

다음 명령의 CHECKPOINT에는 위 출력에서 원하는 model 파일의 전체 경로를 복사한다.
`여기에_전체_경로`는 실제 경로로 바꿔야 한다.

```bash
CHECKPOINT="여기에_전체_경로/model_9999.pt"
ls -lh "$CHECKPOINT"
```

### 5. 로봇 1개로 정지·보행 확인

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py --checkpoint-file "$CHECKPOINT" --device cuda:0 --num-envs 1 --vx 0 --vy 0 --yaw 0
```

창을 닫은 뒤 0.2m/s부터 올린다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py --checkpoint-file "$CHECKPOINT" --device cuda:0 --num-envs 1 --vx 0.2
```

`--vx 1.0`, `--vx 2.0`, `--vx 3.0`으로 바꿔 각각 확인한다.
3.0m/s는 명령 범위의 상한이며 실제 속도 달성을 보장하지 않는다.
이번 방식은 학습과 재생 모두 MuJoCo다. Isaac play를 켜서 export한 후 booster_deploy로 옮기는 과정이 필요 없다.
현재 booster_deploy의 옛 TorchScript 모델 경로에 mjlab 학습 체크포인트를 덮어쓰지 않는다.

### 6. 영상과 학습 곡선

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play.py --checkpoint-file "$CHECKPOINT" --device cuda:0 --num-envs 1 --vx 0.2 --video --video-length 500
```

재생 창에서 500step 이상 진행한 뒤 닫는다. 정책 주기 기준 약 10초 분량이다.
영상은 체크포인트가 있는 run 폴더의 `videos/play/`에 생긴다.
다른 속도를 녹화하기 전에 생성된 mp4를 `vx_0p2.mp4` 등으로 복사해서 덮어쓰기를 방지한다.
W&B 영상 업로드는 이 패키지에서 자동 처리하지 않는다.

```bash
tensorboard --logdir /home/user/seo_mjlab/logs/rsl_rl/seo_walk_vv0_mjlab --host 127.0.0.1 --port 6006
```

원격 데스크톱의 브라우저로 `http://127.0.0.1:6006`을 연다.
팔 torque/deviation, 속도 추종, heading drift, episode length를 함께 비교한다.

## 이식 시 달라지는 점과 검증 범위

- 행동 20, actor 78, critic 92를 유지하며 모든 action과 joint 관측을 명시적 이름 순서로 연결했다.
- 머리 2관절은 MJCF에서 joint를 제거해 고정하되 머리의 질량과 형상은 유지한다.
- yaw 속도 추종 +1.5, heading drift -2, 전진 최대 +3m/s다. 전신 각운동량 비용은 없다.
- 기존 보상 수식은 유지했다. 접촉력 history 축을 MuJoCo B,N,H,3에서 기존 B,H,N,3으로 변환한다.
- 몸체 가속도 비용은 정책 주기 0.02초 속도 차분을 사용한다. Isaac 물리 단계 가속도와 수치가 동일하지 않다.
- 모터는 Ideal PD + 토크 제한 + 학습 시 0~18 physics-step 지연이다. Isaac의 Booster 토크-속도 곡선을 동일하게 재현한 것은 아니다.
- 토크 제한은 제공된 MuJoCo 모델에 맞춰 팔14Nm, 다리 hip pitch30/roll35/yaw20/knee40/ankle20Nm를 쓴다. 기존 Isaac 모터 peak 값과 차이가 있다.
- 마찰 무작위화는 MuJoCo 접선 마찰 0.1~2다. Isaac의 정지/동마찰·반발계수 64버킷은 동일 대응하지 않아 그대로 이식하지 않았다.
- 몸통 질량0.8~1.2배, COM±0.05m, 초기 자세·속도 무작위화, 관절 기준각±0.03rad, reset 시 gain0.9~1.1배, 주기적 외란을 연결했다.
- 평가 모드에서는 무작위화와 외란을 끄고 0속도에서 gait frequency를 0으로 둔다.
- RSL 2.3.1에서 5.5로 옮기며 actor/critic 내부 정규화와 TensorDict 대칭 콜백을 사용한다. 기존 체크포인트의 직접 재개는 지원하지 않는다.
- 로컬에서 문법, 제공 API 생성자 인자, 원본 보상 AST 일치, 관절 집합, 대칭 수학, 팔 비용 수학을 확인했다.
- **서버의 실제 GPU 시뮬레이션 및 PPO 학습은 아직 실행하지 않았다.** preflight → smoke → 2 iteration → 본 학습 순서로 검증해야 한다.

공식 RSL API 참고: https://github.com/leggedrobotics/rsl_rl/blob/v5.5.0/rsl_rl/extensions/symmetry.py
