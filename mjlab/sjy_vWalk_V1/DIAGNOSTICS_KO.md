# V1 진단 패치 — 재학습 없음

기존 V1의 scripts 폴더에 diagnose_velocity.py만 추가한다. 모델·보상·환경 설정을 수정하지 않는다.
진단은 동일 play 환경에서 정지/전진 명령으로 수행한다. 영상은 자동 저장하지 않는다.

## 설치 및 첫 검사

ZIP을 서버 ~/Downloads에 전송한 뒤:

```bash
source ~/miniconda3/bin/activate
conda activate inha_mjlab
cd /home/user/sjy_vWalk_V1_upload_20261116_024158/sjy_vWalk_V1
unzip -n ~/Downloads/sjy_vWalk_V1_diagnostics.zip -d .
export SEO_K1_MESH_DIR=/home/user/booster_assets/robots/K1/meshes
RUN_DIR="$PWD/logs/rsl_rl/sjy_vWalk_V1/2026-11-16_02-42-37_v1_velocity_straight"
CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_velocity.py --checkpoint-file "$RUN_DIR/model_20000.pt" --vx 1 --seconds 3 --warmup 1 --output-dir "$RUN_DIR/diagnostic_smoke"
```

Saved diagnostics가 출력되고 오류 없이 끝나면 아래 전체 진단을 수행한다.
3초 검사는 동작 확인용이며 10초 평가와 직접 비교하지 않는다.
ZIP에 같은 이름 파일이 이미 있으면 -n은 덮어쓰지 않는다. 이전 진단 파일을 임의 덮어쓰지 말고 버전을 확인한다.

## 전체 진단: 2모델 × 4명령

같은 터미널에서 실행한다. 매 실행 고유 하위 폴더를 생성하므로 기존 로그를 덮어쓰지 않는다.

```bash
(
  set -e
  for checkpoint in 20000 29999; do
    for speed in 0 1 1.5 2; do
      CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_velocity.py --checkpoint-file "$RUN_DIR/model_${checkpoint}.pt" --vx "$speed" --seconds 10 --seed 42
    done
  done
)
```

seed 변경만으로 다양한 초기조건 평가가 보장되지 않는다. 현재 play는 랜덤화를 끄므로 초기조건이 같을 수 있다.
이번에는 원인 진단용8회만 수행한다. 여러 초기조건 평가를 위한 perturbation은 별도 설계해야 한다.

## 저장 내용

checkpoint 옆 diagnostics/<모델_속도_seed_시간>/:
- state.csv: 몸체 기준 vx/vy/vz, 몸체/월드 z축 각속도, 시작 방향 기준 속도·거리·횡이탈, heading, root 높이.
- joints.csv: 관절 이름, 실제각/각속도/기본각 차이/목표각, actuator torque, V1 cap, cap 대비 비율.
- summary.json: 2초 이후 통계, 관절별 토크·각도 통계, 좌우 평균 자세/범위 차이, 종료 항목, 실제 코드·모델 SHA256.

near_cap_fraction은 50Hz 표본 중 abs(torque)/V1 고정 cap >=0.95 비율이다.
500Hz 전체 토크 포화시간 또는 포화가 원인이라는 확정 판정이 아니다. 순간 포화는 놓칠 수 있다.
cap은 현재 V1 robot.py의 effort_limit에서 가져온다. 다른 actuator 모델에는 그대로 해석하지 않는다.
world_wz/body_wz는 축 각속도이며 몸통이 기울면 heading의 시간 미분과 정확히 같지 않을 수 있다.
좌우 mean_pose_mirror_bias는 Pitch는 같은 부호, Roll/Yaw는 반대 부호로 비교한 평균각 차이다.
보행 위상 정렬을 하지 않으므로 완전한 보행 대칭 점수가 아니다.
목표각은 해당 물리 step 뒤 읽은 target 값이며 이 스크립트가 목표를 수정하지 않는다.
종료 step은 auto-reset된 자세가 섞이지 않도록 표본에서 제외하고 종료 항목만 저장한다.
오류가 나면 가능한 부분 로그와 error 상태를 저장하며, 완료 결과로 사용하지 않는다.

## 가져오기

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
run = Path('/home/user/sjy_vWalk_V1_upload_20261116_024158/sjy_vWalk_V1/logs/rsl_rl/sjy_vWalk_V1/2026-11-16_02-42-37_v1_velocity_straight')
files = sorted((run / 'diagnostics').rglob('*'))
files = [p for p in files if p.is_file()]
if not files:
    raise SystemExit('진단 결과가 없습니다.')
out = Path.home() / 'Downloads' / 'v1_diagnostics_results.zip'
with ZipFile(out, 'w', ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(run))
print(out)
PY
```

## 검증 상태

Windows에서 구문 및 순수 계산 테스트(초기구간 제외, RMSE, 빈 로그, 좌우 부호, 토크 임계 비율)를 통과했다.
제공된 MJLab API 소스에서 데이터 필드와 종료 manager 인터페이스를 확인했다.
이 Windows 환경에는 torch/mjlab/mujoco가 없어 실제 모델 로딩·GPU 시뮬레이션은 미실행이다.
서버에서 위3초 검사를 먼저 수행한다.
