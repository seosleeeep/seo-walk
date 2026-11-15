# Booster RL Tasks

## Overview

This repository provides a set of reinforcement learning tasks for Booster robots using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).
Currently it includes the fabulous [BeyondMimic motion tracking](https://github.com/HybridRobotics/whole_body_tracking) framework adapted to Booster K1 robots.
This repository follows the standard Isaac Lab project structure, and is tested with IsaacLab 2.2 and Isaac Sim 5.0.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
    git clone https://github.com/BoosterRobotics/booster_train.git
    ```

- Download and install booster_assets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and motion data.
   - Install booster_assets python helper following the instructions in the repository.

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/booster_train
    ```

- Prepare BeyondMimic motion data:
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/csv_to_npz.py --headless --input_file=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.csv --input_fps=<FPS> --output_name=<PATH_TO_BOOSTER_ASSETS>/motions/K1/<MOTION>.npz
    ```

## Usage

- Listing the available tasks:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

- Running a task:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless --device cuda:N
    ```

- Play a trained policy and export it for deployment:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rsl_rl/play.py --task=<TASK_NAME> --checkpoint=<CHECKPOINT_PATH>
    ```

    This script also exports the trained policy to a TorchScript/ONNX file for deployment on real robots in `logs/rsl_rl/<EXPERIMENT>/<RUN>/exported/`.

## Deploy

After a model has been trained and exported, you can deploy the trained policy in MuJoCo or on real Booster robots using the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository. For more details, please refer to the instructions in the [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) repository.


## Acknowledgements

- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): the motion tracking training in BeyondMimic, which is a versatile humanoid control framework that provides highly dynamic motion tracking.

- -----
# Seo Walk V2 서버 적용부터 MuJoCo 실행까지

작성 기준: 2026-11-05. 기존 서버 `/home/user/seo`에서 V0/V1을 학습했고,
`/home/user/booster_deploy`에서 `walk_seo.py`로 시뮬레이션했던 환경 기준이다.
이 문서의 명령은 원격 Ubuntu 터미널에서 직접 실행한다. Windows PowerShell용이 아니다.
이번 작업에서 서버 적용이나 학습을 대신 실행하지 않았다.

## 1 보낼 파일

**`seo_walk_v2_2mps_bundle.zip` 하나만 보낸다.**

- `seo_walk_v2_2mps_bundle.zip`: V2 학습 패치 + MuJoCo V2 설정 + 적용 안내.
- `seo_walk_v2_patch.zip`: 학습 패치만 포함. bundle을 보내면 별도로 필요 없다.
- Windows 탐색기가 확장자를 숨기면 이름 뒤 `.zip`이 안 보여도 압축 파일이다.
- bundle은 전체 프로젝트가 아니다. 기존 `/home/user/seo`와 서버 로봇 asset이 필요하다.

서버 **Downloads**에 ZIP을 받은 다음 시작한다. 단계별로 실행하고 오류가 나면 다음 단계로 넘어가지 않는다.
터미널의 `(env_isaaclab) user@...$` 프롬프트는 복사하지 않는다.
여러 줄 명령 끝의 `\` 뒤에는 공백을 넣지 않는다.

## 2 파일 위치 확인

```bash
ls -lh /home/user/Downloads/seo_walk_v2_2mps_bundle.zip
```

파일명이나 위치가 다르면 다음으로 찾는다.

```bash
find /home/user/Downloads -maxdepth 2 -type f -name '*v2*2mps*.zip'
```

예를 들어 결과가 `/home/user/Downloads/seo_walk_v2_2mps_bundle (1).zip`이면 다음 단계의
`ZIP_FILE`에 그 전체 경로를 큰따옴표 안에 넣는다.

## 3 압축 해제와 기존 코드 백업

아래는 다운로드 파일 이름이 원래 이름일 때 그대로 실행할 수 있다.

```bash
ZIP_FILE="/home/user/Downloads/seo_walk_v2_2mps_bundle.zip"
STAGE_DIR="/home/user/seo_v2_upload_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STAGE_DIR"
unzip "$ZIP_FILE" -d "$STAGE_DIR"
find "$STAGE_DIR" -maxdepth 2 -type d
```

`STAGE_DIR` 안에 `seo`, `deploy_seo`가 나타나야 한다.
ZIP 루트에 seo가 있으므로 이를 기존 seo 안에 그대로 넣어 `seo/seo/source`로 만들면 안 된다.

기존 학습 task와 배포 등록 파일을 백업한다.

```bash
TASK_DIR="/home/user/seo/source/JY_train/JY_train/tasks/manager_based/scratch_symmetry/robots/k1/seo_walk"
BACKUP_DIR="/home/user/seo_v2_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a "$TASK_DIR" "$BACKUP_DIR/seo_walk"
cp -a /home/user/booster_deploy/tasks/scratch/__init__.py "$BACKUP_DIR/deploy_init.py"
echo "$BACKUP_DIR"
```

`cp`에서 경로 오류가 나면 중단하고 실제 프로젝트 위치부터 확인한다.
이 백업은 task 코드이며 대용량 학습 로그를 복제하지 않는다.

## 4 패치 파일 적용

앞 단계와 **같은 터미널**에서 실행한다. 코드 일부를 교체하며 기존 학습 로그와 asset은 유지된다.

```bash
cp -a "$STAGE_DIR/seo/source/." /home/user/seo/source/
cp "$STAGE_DIR/deploy_seo/walk_seo_v2.py" /home/user/booster_deploy/tasks/scratch/walk_seo_v2.py

grep -n 'seo_walk_v2' "$TASK_DIR/__init__.py"
grep -n -E 'lin_vel_x|max_forward_vel|velocity_scales' "$TASK_DIR/velocity_v2.py"
ls -l /home/user/booster_deploy/tasks/scratch/walk_seo.py
```

확인 기준:

- `seo_walk_v2`, `seo_walk_v2-Play` 등록이 나온다.
- `lin_vel_x = (-1.5, 2.0)`이 나온다.
- 기존 배포 파일 `walk_seo.py`가 존재한다. 새 `walk_seo_v2.py`가 이를 상속한다.
- **기존 서버 walk_seo.py를 로컬의 과거 파일로 덮어쓰지 않는다.**

## 5 Isaac 학습 환경 설정

**새 터미널을 열 때마다** 다음 블록을 다시 실행한다.

```bash
source /home/user/miniconda3/bin/activate
conda activate env_isaaclab
cd /home/user/seo
export PYTHONPATH="/home/user/seo/source/JY_train:/home/user/booster_assets/src${PYTHONPATH:+:$PYTHONPATH}"
export SEO_K1_MESH_DIR="/home/user/booster_assets/robots/K1/meshes"

echo "$CONDA_DEFAULT_ENV"
python -c "import importlib.util; print(importlib.util.find_spec('JY_train').origin)"
python -c "from booster_assets import BOOSTER_ASSETS_DIR; print(BOOSTER_ASSETS_DIR)"
```

`env_isaaclab`와 `/home/user/seo/source/JY_train/JY_train/__init__.py`가 나와야 한다.
예전에 나온 `/home/user/JY_train/...`이면 다른 프로젝트를 읽는 것이므로 진행하지 않는다.

```bash
TASK_DIR="/home/user/seo/source/JY_train/JY_train/tasks/manager_based/scratch_symmetry/robots/k1/seo_walk"
python -m py_compile "$TASK_DIR/__init__.py" "$TASK_DIR/velocity_v1.py" "$TASK_DIR/velocity_v2.py" "$TASK_DIR/arm_rewards.py"
nvidia-smi
```

구문 검사는 오류 없이 프롬프트로 돌아오면 통과다. 이것만으로 Isaac 통합 검증이 끝나는 것은 아니다.
GPU 1번에 다른 학습이 실행 중이면 사용 일정을 확인한다. 다른 사람 프로세스를 종료하지 않는다.
아래 명령은 물리 GPU 1번을 사용하며 `--device cuda:0`을 그대로 둔다.

## 6 V1 최종 모델을 V2 시작점으로 준비

기존 대화에서 확인한 실제 V1 모델 경로다. 모델이 그대로 있다면 수정 없이 실행한다.

```bash
V1_CHECKPOINT="/home/user/seo/logs/rsl_rl/seo_walk_v1/2026-11-04_20-06-08_seo_20dof_v1_continue/model_10997.pt"
ls -lh "$V1_CHECKPOINT"
```

파일이 확인되면 복사한다. 여기서는 **학습용 model_10997.pt**를 사용하며 exported 파일을 쓰지 않는다.

```bash
mkdir -p /home/user/seo/logs/rsl_rl/seo_walk_v2/init_from_v1
cp "$V1_CHECKPOINT" /home/user/seo/logs/rsl_rl/seo_walk_v2/init_from_v1/model_10997.pt
```

경로가 없으면 다음 결과에서 실제 V1 모델을 골라 `V1_CHECKPOINT="전체 경로"`로 바꾼다.

```bash
find /home/user/seo/logs/rsl_rl/seo_walk_v1 -type f -name 'model_*.pt' | sort -V | tail -20
```

## 7 먼저 작은 실행으로 오류 확인

```bash
cd /home/user/seo
CUDA_VISIBLE_DEVICES=1 python -u scripts/rsl_rl/train.py \
  --headless --device cuda:0 \
  --task seo_walk_v2 --num_envs 16 \
  --resume --load_run init_from_v1 --checkpoint model_10997.pt \
  --max_iterations 2 --run_name seo_v2_smoke
```

이 단계는 16개 환경에서 2 iteration만 실행하는 통합 검사다. 처음 초기화에는 시간이 걸린다.
정상적으로 완료되어 프롬프트가 돌아오고 Traceback/NaN 오류가 없어야 한다.
로그에 `arm_shoulder_pose`, `arm_elbow_pose`, `arm_torque`, `arm_roll_symmetry`가 등록되는지도 확인한다.
episode 집계가 아직 없으면 개별 Episode_Reward가 모두 출력되지 않을 수도 있다.

## 8 본 학습 시작

작은 실행이 끝난 뒤 실행한다. smoke 모델 대신 **같은 V1 최종 모델에서** 본 학습을 시작한다.

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/rsl_rl/train.py \
  --headless --device cuda:0 \
  --task seo_walk_v2 --num_envs 1024 \
  --resume --load_run init_from_v1 --checkpoint model_10997.pt \
  --max_iterations 10000 --run_name seo_v2_2mps
```

- V1에서 **추가 10,000 iteration**이다. 10,000 epoch라는 뜻이 아니다.
- 각 iteration의 PPO learning epochs는 5다.
- 체크포인트는 설정상 100 iteration 간격으로 저장된다.
- 완료 예상은 초기화 후 나오는 ETA를 참고한다. 정확한 시간을 미리 보장하지 않는다.
- 위 명령을 여러 터미널에서 중복 실행하지 않는다.
- 학습을 유지하려면 프로세스가 실행 중인 터미널과 서버를 종료하지 않는다.
- 중단 시 해당 학습 터미널에서 Ctrl+C를 한 번 누른다. 재개는 아래 주의사항 참고.

별도 터미널에서 학습 여부만 확인하려면:

```bash
pgrep -af 'python.*scripts/rsl_rl/train.py'
nvidia-smi
```

## 9 학습이 끝나면 실행 폴더와 모델 선택

새 터미널이라면 5번 환경 설정부터 다시 한다.
아래는 `seo_v2_2mps`라는 이름으로 시작한 실행 중 가장 최근 폴더를 선택한다.

```bash
RUN_DIR=$(find /home/user/seo/logs/rsl_rl/seo_walk_v2 -mindepth 1 -maxdepth 1 -type d -name '*_seo_v2_2mps' | sort | tail -n 1)
echo "$RUN_DIR"
find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -15
```

`RUN_DIR`가 빈 값이면 계속하지 않는다. 학습 run_name이나 로그 경로가 다른지 확인한다.
최신 번호 모델을 고른다. 최신 번호가 최상의 성능이라는 뜻은 아니다.

```bash
CHECKPOINT=$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)
echo "$CHECKPOINT"
ls -lh "$CHECKPOINT"
```

중간 모델을 확인하고 싶으면 이 변수만 직접 바꾼다. 예를 들어 실제 목록에 model_12000.pt가 있으면:

```bash
CHECKPOINT="$RUN_DIR/model_12000.pt"
```

목록에 없는 번호는 쓰지 않는다. `$RUN_DIR`를 쓰기 때문에 날짜와 시간을 매번 타이핑할 필요가 없다.

## 10 Isaac에서 확인하거나 창 없이 export

### 방법 A 로봇 1개를 Isaac 화면에서 확인

```bash
cd /home/user/seo
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play.py \
  --device cuda:0 --task seo_walk_v2-Play \
  --num_envs 1 --checkpoint "$CHECKPOINT"
```

V2 Play 기본 명령은 전진 0.2m/s다. 명령 상한이 2.0이라고 자동으로 2.0으로 걷지는 않는다.
export 파일이 생성되고 보행을 확인한 뒤 **이 play를 실행한 터미널**에서 Ctrl+C로 종료한다.
다른 사람의 학습 터미널에는 입력하지 않는다.

2.0m/s를 Isaac에서 직접 확인하려면, 로컬 스크립트가 사용하는 Hydra override로 다음처럼 실행할 수 있다.
서버 코드가 다르거나 override 오류가 나오면 기본 명령으로 먼저 확인한다.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play.py \
  --device cuda:0 --task seo_walk_v2-Play \
  --num_envs 1 --checkpoint "$CHECKPOINT" \
  'env.commands.motion.play_velocity_command=[2.0,0.0,0.0]'
```

### 방법 B Isaac 창 없이 export만

서버 play.py에 자동 종료 기능이 있는지 먼저 확인한다. bundle은 play.py를 교체하지 않는다.

```bash
grep -n 'Exiting after model export' /home/user/seo/scripts/rsl_rl/play.py
```

해당 문구가 나오면 아래 명령은 export 후 자동으로 끝난다.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play.py \
  --headless --device cuda:0 --task seo_walk_v2-Play \
  --num_envs 1 --checkpoint "$CHECKPOINT"
```

문구가 없으면 방법 A를 사용한다. Isaac 앱은 export를 위해 로봇과 환경을 초기화하지만,
headless 모드는 창을 표시하지 않는다. 별도 수동 빌드가 필요한 과정이 아니다.

## 11 export 확인과 체크포인트별 보존

**play 실행이 끝난 같은 터미널**에서 실행한다.

```bash
ls -lh "$RUN_DIR/exported/"
EXPORT_PT="$RUN_DIR/exported/seo_walk_v2_$(basename "$RUN_DIR").pt"
ls -lh "$EXPORT_PT"
```

학습용 model 파일과 달리 이 `.pt`는 배포 정책이다. 옆에 `.onnx`도 생성된다.
위 경로에 없으면 `ls`에 출력된 실제 `.pt` 경로를 `EXPORT_PT="전체 경로"`에 넣는다.
play가 실패했다면 예전 export를 새 결과로 착각하지 말고 수정 시각부터 확인한다.

같은 run의 다른 체크포인트를 export하면 기본 export 이름은 같아 덮어쓸 수 있다.
현재 결과를 체크포인트별 폴더에 보존한다.

```bash
CKPT_NAME=$(basename "$CHECKPOINT" .pt)
ARCHIVE_DIR="$RUN_DIR/exports_by_checkpoint/$CKPT_NAME"
mkdir -p "$ARCHIVE_DIR"
cp "$EXPORT_PT" "$ARCHIVE_DIR/policy.pt"
printf '%s\n' "$CHECKPOINT" > "$ARCHIVE_DIR/source_checkpoint.txt"
sha256sum "$ARCHIVE_DIR/policy.pt" > "$ARCHIVE_DIR/policy.sha256"
export SEO_V2_CHECKPOINT="$ARCHIVE_DIR/policy.pt"
echo "$SEO_V2_CHECKPOINT"
```

이 예시는 파일을 models 폴더에 또 복사하지 않고 **실제 export의 절대 경로**로 연결한다.
V2 배포 설정이 이 환경 변수를 읽도록 작성되어 있다.

## 12 MuJoCo V2 task 등록

4번에서 `walk_seo_v2.py`를 복사했다. 기존 `walk_seo.py`와 `walk_20dof.py`도 있어야 한다.

```bash
ls /home/user/booster_deploy/tasks/scratch/walk_seo.py
ls /home/user/booster_deploy/tasks/scratch/walk_20dof.py
grep -n 'seo_walk_v2_20dof' /home/user/booster_deploy/tasks/scratch/__init__.py
```

마지막 명령에서 등록이 이미 나오면 다시 추가하지 않는다.
없으면 서버 VS Code에서 파일을 연다.

```bash
code /home/user/booster_deploy/tasks/scratch/__init__.py
```

기존 코드를 지우지 말고 맨 아래에 두 줄을 추가하고 **Ctrl+S**로 저장한다.
아래 두 줄은 터미널이 아니라 VS Code 편집기에 넣는 Python 코드다.

```python
from .walk_seo_v2 import SeoWalkV2ControllerCfg
register_task("seo_walk_v2_20dof", SeoWalkV2ControllerCfg())
```

기존 파일 위쪽에 `from booster_deploy.utils.registry import register_task`가 있던 구조를 전제로 한다.

## 13 MuJoCo 환경 활성화와 실행

**11번의 변수가 남아 있는 같은 터미널**에서 실행한다.

```bash
conda activate inha_mjlab
cd /home/user/booster_deploy
export PYTHONPATH="/home/user/booster_deploy:/home/user/booster_assets/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "from booster_assets import BOOSTER_ASSETS_DIR; print(BOOSTER_ASSETS_DIR)"
python -m py_compile tasks/scratch/walk_seo_v2.py tasks/scratch/__init__.py
echo "$SEO_V2_CHECKPOINT"
ls -lh "$SEO_V2_CHECKPOINT"

python scripts/deploy.py --task seo_walk_v2_20dof --mujoco --device cpu
```

CPU는 배포 정책 추론 장치다. 학습은 앞에서 GPU 1번을 사용했다.
`--mujoco`를 반드시 포함한다. 실물 로봇 실행은 이 매뉴얼 범위가 아니다.

새 터미널을 열었다면 `source /home/user/miniconda3/bin/activate`부터 한 뒤,
11번에서 출력한 실제 경로를 직접 지정한다. 아래 대문자 날짜 표시는 예시이며 그대로 입력하지 않는다.

```bash
export SEO_V2_CHECKPOINT="/home/user/seo/logs/rsl_rl/seo_walk_v2/실제실행폴더/exports_by_checkpoint/실제모델이름/policy.pt"
```

직접 입력이 어려우면 5번, 9번의 변수 설정을 다시 하고 기존 archive 경로를 다음으로 지정한다.

```bash
CKPT_NAME=$(basename "$CHECKPOINT" .pt)
export SEO_V2_CHECKPOINT="$RUN_DIR/exports_by_checkpoint/$CKPT_NAME/policy.pt"
```

그 파일이 11번에서 실제로 생성됐는지 `ls -lh`로 확인한다.

## 14 걷기 명령 입력

MuJoCo 화면이 아니라 **deploy를 실행한 터미널**을 클릭한다.
`Set command (x, y, yaw) or 'kick':`가 표시되면 아래 한 줄씩 입력하고 Enter를 누른다.

| 입력 | 의미 |
|---|---|
| `0.2 0 0` | 전진 0.2m/s 명령 |
| `0.5 0 0` | 전진 0.5m/s |
| `1.0 0 0` | 전진 1.0m/s |
| `1.5 0 0` | 전진 1.5m/s |
| `2.0 0 0` | 전진 2.0m/s |
| `0 0 0` | 정지 명령 |

처음에는 0.2에서 팔 자세·직진성·낙상 여부를 확인하고 올린다.
숫자 세 개는 전진속도(m/s), 좌우속도(m/s), yaw 각속도(rad/s)다.
2.0은 목표 명령이며 실제 속도가 2.0에 도달했다는 뜻은 아니다. 3이나 10을 넣어도 상한은 2.0이다.
종료는 해당 터미널에서 Ctrl+C. 프로세스가 끝나야 새 모델 경로로 다시 실행할 수 있다.

## 15 다음 체크포인트를 보고 싶을 때

1. 기존 MuJoCo를 종료한다.
2. 5번 Isaac 환경을 다시 설정한다.
3. 9번에서 같은 RUN_DIR과 다른 CHECKPOINT를 선택한다.
4. 10번 play/export, 11번 체크포인트별 보존을 실행한다.
5. 13번 MuJoCo 환경 및 실행을 반복한다.

task 등록은 한 번이면 된다. 체크포인트마다 다시 추가하지 않는다.
모델 파일은 프로세스 시작 시 로드되므로 실행 중 경로만 바꿔서는 새 정책으로 교체되지 않는다.

## 16 V2 중단 후 재개 시 주의

8번 명령을 다시 실행하면 V1에서 새 V2를 시작한다. 중단된 V2를 이어가는 명령이 아니다.
V2를 이어갈 때는 `--load_run`에 실제 V2 폴더 이름, `--checkpoint`에 해당 모델 파일명을 넣는다.
또한 팔 허용 폭 커리큘럼의 `curriculum_start_step`은 완료한 **V2 policy step**으로 맞춰야 한다.
checkpoint 번호에는 V0/V1 이력이 섞여 있으므로 번호 자체를 넣지 않는다.
V2에서 2,000 iteration 이상 진행해 최종 폭에 도달했다면 두 pose 항목이 사용하는
`curriculum_start_step`을 48000으로 설정해 최종 폭을 유지할 수 있다.
중간 단계에서는 실제 완료한 V2 rollout 수 × 24를 기준으로 offset을 계산하고 기록한다.

## 17 흔한 문제

| 증상 | 우선 확인 |
|---|---|
| python을 찾지 못함 | conda 활성화부터 다시 실행 |
| JY_train 경로가 다른 프로젝트 | 5번 PYTHONPATH 설정 및 find_spec 재확인 |
| booster_assets 없음 | 해당 환경에서 booster_assets/src를 PYTHONPATH에 포함 |
| seo_walk_v2 task 없음 | 4번 파일 복사 위치 및 task __init__.py 확인 |
| seo_walk_v2_20dof task 없음 | 12번 배포 등록·저장 확인 |
| checkpoint 없음 | 실제 파일 목록과 경로 변수 echo/ls 확인 |
| 로봇이 가만히 있음 | deploy 터미널에 `0.2 0 0` 입력; pause/prepare와 오류 로그 확인 |
| V1과 같은 팔 자세 | SEO_V2_CHECKPOINT가 V2 새 export인지 확인 |
| headless export가 안 끝남 | 서버 play.py 자동 종료 문구 확인; 방법 A로 진행 |
| 실행 중 Traceback/NaN | 다음 단계로 넘어가지 말고 오류 마지막 부분 보존 |

로그가 너무 길면 명령 뒤에 `2>&1 | tee 파일경로.log`를 붙여 저장할 수 있다.
이 경우 파이프의 성공만 보고 학습 성공으로 판단하지 말고 로그와 생성 모델을 확인한다.
원격 접속이 끊겨도 견디는 학습 실행은 tmux 등 별도 세션 설정이 필요하며, 일반 터미널 유지와는 다르다.

