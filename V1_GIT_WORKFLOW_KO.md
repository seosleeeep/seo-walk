# V1 Git 관리와 서버 실행

브랜치: `sjy-vwalk-v1`. 학습 코드: `mjlab/sjy_vWalk_V1`.
기존 main의 Isaac Lab 코드와 과거 실험 기록을 보존한다.
V1의 변경과 미검증 사항은 [V1 설명](mjlab/sjy_vWalk_V1/README_V1_STATUS.md)을 참고한다.
추가된 train_recorded.py는 학습 설정을 바꾸지 않고 commit·명령·패키지 버전·소스 해시를 기록한다.

## Windows: 최초 작업본 만들기

GitHub 로그인 후 PowerShell에서 실행한다. 동일 폴더가 이미 있다면 새 clone 대신 해당 폴더에서 상태를 확인한다.

```powershell
cd C:\Users\seoja\OneDrive\Desktop\seoslept
git clone --branch sjy-vwalk-v1 https://github.com/seosleeeep/seo-walk.git seo-walk-v1
cd seo-walk-v1
```

이후 수정은 `seo-walk-v1/mjlab/sjy_vWalk_V1`에서 한다. 기존 Desktop/sjy_vWalk/V1은 별도 사본이며 자동 동기화되지 않는다.

```powershell
git pull --ff-only origin sjy-vwalk-v1
# 코드 수정 후 diff를 확인하고 명시한 경로만 commit
git diff
git add mjlab/sjy_vWalk_V1
git commit -m "fix: describe the walking change"
git push origin sjy-vwalk-v1
```

## 서버: 최초 다운로드 및 고정 코드로 학습

비공개 저장소이므로 먼저 서버 GitHub 인증이 필요하다. 비밀번호 대신 GitHub CLI 로그인 또는 SSH/PAT를 사용한다. 토큰을 코드/문서에 저장하지 않는다.

```bash
git clone --branch sjy-vwalk-v1 https://github.com/seosleeeep/seo-walk.git ~/seo-walk-git
cd ~/seo-walk-git
git fetch origin sjy-vwalk-v1
COMMIT=$(git rev-parse origin/sjy-vwalk-v1)
RUN_ROOT="$HOME/seo-walk-runs/v1_$(date +%Y%m%d_%H%M%S)_${COMMIT:0:8}"
mkdir -p "$HOME/seo-walk-runs"
git worktree add --detach "$RUN_ROOT" "$COMMIT"
cd "$RUN_ROOT/mjlab/sjy_vWalk_V1"
source ~/miniconda3/bin/activate
conda activate inha_mjlab
export SEO_K1_MESH_DIR=/home/user/booster_assets/robots/K1/meshes
python scripts/preflight.py
CUDA_VISIBLE_DEVICES=1 python scripts/smoke.py
CUDA_VISIBLE_DEVICES=1 python scripts/train_recorded.py --gpu-ids 0 --env.scene.num-envs 1024 --agent.max-iterations 30000 --agent.run-name v1_velocity_straight
```

GPU1만 노출했으므로 내부 번호는 0. 기존 외부 mesh 경로를 사용한다.
학습 폴더에 pull하지 않는다. Windows에서 새 버전을 push하면 서버 관리 저장소에서 fetch 후 새 worktree를 만든다.
서버 학습/모델 검증은 아직 실행하지 않았다. 검사 실패 시 학습을 시작하지 않고 오류를 확인한다.

## 결과 기록

학습 worktree의 logs와 run_records는 Git에서 제외된다. 학습 폴더는 모델과 함께 보존한다.
평가는 동일 worktree에서 README의 evaluate_velocity.py 명령으로 실행한다.
서버 관리 저장소 `~/seo-walk-git`에서 결과를 commit한다.

```bash
cd ~/seo-walk-git
git switch sjy-vwalk-v1
git pull --ff-only origin sjy-vwalk-v1
mkdir -p experiments/sjy_vWalk_V1/run_001
cp experiments/sjy_vWalk_V1/TEMPLATE.md experiments/sjy_vWalk_V1/run_001/REPORT.md
```

REPORT.md를 실제 값으로 채우고 해당 실행의 manifest.json 및 평가 summary.json만 run_001에 복사한다.
checkpoint 경로와 `sha256sum /정확한/경로/model_번호.pt` 결과를 함께 기록한다.
모델·영상·전체 CSV는 서버에 보관한다. 다음 실험은 run_002처럼 고유한 폴더를 사용한다.

```bash
git add experiments/sjy_vWalk_V1/run_001
git diff --cached
git commit -m "results: record v1 run_001 evaluation"
git push origin sjy-vwalk-v1
```

Windows에서도 다음 수정 전에 pull한다. 서버에서는 코드 수정 대신 결과 기록만 하는 흐름을 권장한다.
