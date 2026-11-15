# sjy_vWalk V1 — 직진 속도 추종 실험

V0 ZIP 기반 독립 작업본. Task/experiment: sjy_vWalk_V1, 패키지: sjy_vwalk_v1.
제공된 K1 Velocity Walk README를 참고했지만 동일 환경의 전체 이식본은 아니다.
원본 V0는 보존했다. V1은 새 학습으로 시작한다.

## 변경

| 항목 | V0 | V1 |
|---|---|---|
| 학습 명령 | 절반 waypoint | 전 환경 속도 명령; waypoint는 선택적 play에만 사용 |
| 직진 표본 | 기존 다축 curriculum | 50%를 전진 0.2~3m/s, vy=yaw=0으로 지정 |
| 정지 표본 | 표본 수 10% 정수 절삭 | 각 표본 확률 10%, 한 환경 resample에도 적용 |
| 명령 유지 | 0.02~4초 | 2~4초 |
| 기본각 랜덤화 | 좌우 독립 ±0.03 | 좌우 미러 부호를 지키는 ±0.03rad |
| 팔 kp/kd | 20/0.5 | 10/1 (참고 문서) |
| Elbow_Yaw HOME | -0.5/+0.5 | -0.4/+0.4rad (참고 문서) |
| 높이 비용 | 속도 연동, -30 | 평면 높이 0.52m 고정, -20 (참고 문서) |
| play 초기 명령 | 전진 0.2 | 정지 0 |

정지 추출을 마지막에 적용하므로 전체의 약 45%는 전진 전용, 10%는 정지,
나머지 45%는 기존 curriculum이다. curriculum 자체에도 직진 표본이 있다.
전체 vx 범위 -1.5~3m/s, vy ±1.5m/s, yaw ±1.6rad/s 유지. 실제 도달 속도 보장은 아니다.

## 유지한 항목 / 참고 문서와 다른 부분

- Actor78/critic92/action20과 기존 관절 순서 유지. 참고 환경의 72/86·SDK 순서와 다르다.
- 팔 포함 20관절 대칭 데이터 증강·mirror loss 0.005 유지. 참고 문서는 이를 사용하지 않는다.
- 다리 반주기 대칭 비용 -0.5, heading_drift -2 유지. 전신 각운동량 yaw 페널티는 없다.
  yaw-rate 추종과 roll/pitch 각속도 비용은 별도로 남아 있다.
- 팔 자세 비용 -0.1 유지. std는 정지 0.05, 보행 기준 0.15,
  고속에서 각 팔 [0.5,0.2,0.35,0.35]rad로 연속 보간한다.
  참고 문서의 고속 std=0.15 고정과 다르다. 별도 팔 토크 항은 없지만 전체 torque/power에는 포함된다.
- 팔 action scale Pitch 계열 0.3, Roll/Elbow_Yaw 0.05, 다리1.0 유지. 팔 관절은 가동 상태다.
- Ideal PD와 V0 일정 torque cap 유지: 다리 [30,35,20,40,20,20]Nm, 팔14Nm.
  참고 문서의 속도 의존 모터 곡선과 다리 cap [40,50,36,56,36,36]은 미구현이다.
- V0 접촉·외란·마찰/질량/COM/gain 랜덤화 유지. 참고 문서의 접촉재질, 공통 지연,
  COM 속도 등까지 동일하게 이식·검증한 것은 아니다.
- 1024환경, 물리0.002초, decimation10, 정책50Hz, episode30초, gait1.5Hz,
  rollout24, PPO epoch5, 기본30000 iteration 유지.
- 별도 curriculum counter 저장은 추가하지 않았다. resume 시 counter 복원은 별도 확인이 필요하다.
- play 터미널 입력은 미구현. --vx/--vy/--yaw 옵션으로 지정한다.
- V0 checkpoint는 차원이 같아도 HOME/gain이 달라 V1 결과로 사용하지 않는다.
  배포에도 V1의 관절 순서/HOME/gain/action scale을 함께 적용해야 한다.

## 서버 실행

ZIP 최상위 폴더는 sjy_vWalk_V1. 서버에 동명 폴더가 있으면 먼저 백업한다.

```bash
source ~/miniconda3/bin/activate
conda activate inha_mjlab
unzip ~/Downloads/sjy_vWalk_V1.zip -d ~/
cd ~/sjy_vWalk_V1
export SEO_K1_MESH_DIR=/home/user/booster_assets/robots/K1/meshes
python scripts/preflight.py
CUDA_VISIBLE_DEVICES=1 python scripts/smoke.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --gpu-ids 0 --env.scene.num-envs 1024 --agent.max-iterations 30000 --agent.run-name v1_velocity_straight
```

물리 GPU1만 노출했으므로 내부 GPU 인덱스는 0이다. mesh는 ZIP에 포함하지 않았다.

## 검증

```bash
cd ~/sjy_vWalk_V1
MODEL=$(find "$PWD/logs/rsl_rl/sjy_vWalk_V1" -type f -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
printf '%s\n' "$MODEL"
CUDA_VISIBLE_DEVICES=1 python scripts/play.py --agent trained --checkpoint-file "$MODEL" --device cuda:0 --vx 0
CUDA_VISIBLE_DEVICES=1 python scripts/play.py --agent trained --checkpoint-file "$MODEL" --device cuda:0 --vx 1
for speed in 1 1.5 2 2.5; do
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_velocity.py --checkpoint-file "$MODEL" --vx "$speed" --seconds 10
done
```

checkpoint 옆 evaluations/<시간>/velocity.csv 및 summary.json 저장.
외부 heading/waypoint 보정 없이 고정 속도 명령을 평가한다.
초기2초 제외 평균 전진속도·RMSE, 최대 횡이탈·방향오차, 마지막 전진거리를 비교한다.
속도는 시작 방향에 투영한 root-link 속도이며 COM 속도와 다르다.
episode_ended는 성공 판정이 아니다. 낙상/시간제한 등 종료 원인은 별도로 확인한다.
기존 evaluate_waypoint.py는 보정기를 포함한 별도 평가용이다.

## 로컬 검사 한계

Python 구문, 모델20관절 이름, 팔 HOME 한계·좌우 대칭 정적 검사.
이 컴퓨터에 torch/mjlab/mujoco가 없어 모델 compile, GPU step/PPO, 보행은 실행하지 않았다.
서버 preflight/smoke 후 학습한다. 직진 개선 여부는 실험 로그로 판단해야 한다.
