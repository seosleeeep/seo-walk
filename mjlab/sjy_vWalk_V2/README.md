# sjy_vWalk V2 — 성능 기준 전진 속도 curriculum

V1 대비 실험 변수는 **전진 양수 속도 명령의 상한을 결정하는 방식**이다.
팔/보상/관측/관절 순서/HOME/PD/토크 제한/action scale/대칭/PPO는 버전 식별자 외에 V1과 동일하다.
기존 V1의 후진·횡이동·회전 범위와 명령 비율도 유지한다. 별도 완전한 저속 전용 학습은 아니다.
Task/experiment/package: sjy_vWalk_V2 / sjy_vWalk_V2 / sjy_vwalk_v2.

## 변경한 이유

V1 진단에서3만 모델의2m/s 명령에 몸체전진속도가약0.230m/s였으며 관측된 상한근접 토크 표본이 없었다.
속도 추종 원시보상은 큰 속도오차에서 거의0이었다. 이를 원인으로 확정하지 않고,
낮은 속도부터 범위를 확대하면 학습이 달라지는지를 비교한다.
이전 정책에서 추가 학습하지 않고 새 학습으로 시작한다. 기본30000 iteration,1024환경,단일 GPU.

## 승급 규칙 — 실험용 기준이며 검증된 최적값 아님

상한: 1.0 → 1.5 → 2.0 → 2.5 → 3.0m/s.
전진 전용 표본뿐 아니라 다축 조합의 양수 vx에도 현재 상한을 적용한다.
후진 최소-1.5,vy±1.5,yaw-rate±1.6은 V1과 같다.

평가 대상은 현재 상한의80% 이상인 순수 직진 명령 구간이다.
각2~4초 명령 구간의 첫1초를 제외한 표본에서 다음을 동시에 확인한다:

- 몸체vx 오차 <= 0.15 + 0.15 × 명령vx (m/s)
- 몸체vy 절댓값 <= 0.15m/s
- 몸체yaw 각속도 절댓값 <= 0.2rad/s
- 해당 명령 시작 방향 대비 heading 오차 <=10도
- 해당 명령 시작 직선 대비 횡이탈 <=0.15m

구간 길이1.5초 이상·평가표본20개 이상이고, 평가표본80% 이상이 위 기준을 통과해야 성공 구간이다.
리셋으로 끊긴 대상 구간은 실패로 센다. 단계별 최소1000 iteration 유지한다.
매100 iteration의 독립 창에서 완료 구간200개 이상과 성공률80% 이상을 두 번 연속 만족하면 승급한다.
승급 시 이전 상한의 진행 중 구간은 평가에서 제외하고 다음 명령 재추출부터 새 상한을 적용한다.

이 기준은2~4초 학습 구간의 승급용이다.10초 별도 평가를 대체하지 않는다.
외란·마찰 등 학습 랜덤화가 켜진 상태이므로 엄격해서1m/s에서 멈출 수 있다.
시간만 지나면 승급하는 구조가 아니며,3만 iteration 내3m/s 도달을 보장하지 않는다.
고속으로 올라가지 못하면 실패 원인을 로그로 확인한다. 자동으로 기준을 느슨하게 바꾸지 않는다.

## 로그·재개

터미널에100 iteration 간격으로 [speed_curriculum] JSON:
iteration, completed_segments, success_rate, cap_before/after, promoted.
Command metrics에도 speed_cap_mps/gate_success_rate/gate_stage를 제공한다(episode 로그 갱신 시 기록).
현재 단계·카운터·승급 이력을 위한 상태는 checkpoint infos.v2_speed_gate에 저장하고,
기존 runner의 common_step_counter 복원과 함께 사용한다. 진행 중 물리 궤적은 이어지지 않으므로
resume에서 구간별 평가 버퍼는 비운다. 완료 구간 집계는 복원한다.
V1 checkpoint로 V2 resume를 시도하면 오류를 낸다. V2는 새 학습부터 비교한다.
정책 play에서는 curriculum을 끄므로 --vx3도 명령할 수 있지만, 도달한 학습 단계 밖일 수 있다.

## 서버 새 학습 — 직접 실행

Downloads에 sjy_vWalk_V2.zip을 보낸 뒤 실행. 프로젝트는 새 시간 폴더에 압축 해제한다.

```bash
source ~/miniconda3/bin/activate
conda activate inha_mjlab
UPLOAD_DIR="$HOME/sjy_vWalk_V2_upload_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$UPLOAD_DIR"
unzip ~/Downloads/sjy_vWalk_V2.zip -d "$UPLOAD_DIR"
cd "$UPLOAD_DIR/sjy_vWalk_V2"
export SEO_K1_MESH_DIR=/home/user/booster_assets/robots/K1/meshes
python scripts/test_speed_curriculum.py
python scripts/preflight.py
CUDA_VISIBLE_DEVICES=1 python scripts/smoke.py
```

모든 검사 통과 후 짧은 실제 PPO/저장 검사:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --gpu-ids 0 --env.scene.num-envs 32 --agent.max-iterations 1 --agent.run-name v2_smoke
```

성공 후1024환경 새 학습(위1회 검사에서 resume하지 않음):

```bash
mkdir -p run_records
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --gpu-ids 0 --env.scene.num-envs 1024 --agent.max-iterations 30000 --agent.run-name v2_speed_curriculum 2>&1 | tee "run_records/train_$(date +%Y%m%d_%H%M%S).log"
```

GPU1만 노출하므로 내부 --gpu-ids0. 학습을 시작한 터미널/프로세스를 임의 종료하지 않는다.
중간1000·3000·5000·10000 부근 checkpoint에서 정지/1m/s를 기존 evaluate_velocity.py와 diagnose_velocity.py로 평가한다.
현재 상한보다 높은 속도 평가는 별도 외삽 시험으로 구분한다.

## 코드 위치

- src/sjy_vwalk_v2/speed_curriculum.py: 승급 조건과 저장 상태(순수 Python).
- src/sjy_vwalk_v2/curriculum_command.py: 명령 상한, 구간 측정, 로그.
- src/sjy_vwalk_v2/runner.py: checkpoint curriculum 상태 저장/복원.
- scripts/test_speed_curriculum.py: 시간만으로 승급하지 않음·독립 창·최소 체류·최대 상한·상태 복원 검사.

## 검증 범위

Windows에서4개 단위 테스트와 Python 구문 검사 통과.
V1 대비 보상·모터·팔·관측·action·대칭·PPO 파일이 식별자 외에는 같음을 비교했다.
제공된 MJLab API의 command 호출순서·runner/registry 인터페이스를 확인했다.
로컬에 torch/mjlab/mujoco가 없어 실제 GPU 환경과 PPO 저장/재개를 실행하지 않았다.
서버 smoke 및1회 PPO 검사에서 오류가 나면 학습을 시작하지 않고 오류를 확인한다.
