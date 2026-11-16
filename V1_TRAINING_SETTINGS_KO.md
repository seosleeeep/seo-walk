# V1 학습 환경·입력 속도·학습 주기

작성일: 2026-11-17. 기준: 업로드한 sjy_vWalk_V1 코드와 사용자에게 안내한 실행 명령.
서버에서 변경한 설정이 있다면 실제 실행 로그의 params와 실행 명령이 최종 기준이다.
설정 설명이며 보행 성능 검증 보고서가 아니다.

- [실험 운영 가이드](SELF_SERVICE_RL_GUIDE_KO.md)
- [Git 관리·서버 실행](V1_GIT_WORKFLOW_KO.md)
- [V1 변경 사항](mjlab/sjy_vWalk_V1/README_V1_STATUS.md)
- [실험 기록 양식](experiments/sjy_vWalk_V1/TEMPLATE.md)

## 환경

| 항목 | 설정 |
|---|---|
| 시뮬레이션 | MuJoCo 기반 MJLab, 평지 |
| 로봇 | K1 다리12 + 팔8관절 제어, 목2관절 고정 |
| 병렬 환경 | 1,024 |
| 학습량 | 30,000 iteration |
| GPU | 안내 명령 기준 물리 GPU1; CUDA_VISIBLE_DEVICES=1, 내부 --gpu-ids 0 |
| 에피소드 | 최대30초, 종료 조건에 따라 조기 reset |
| 물리 step | 0.002초 = 500Hz |
| 정책 step | decimation10, 0.02초 = 50Hz |
| 보행 기준 위상 | 이동1.5Hz, 정지 명령에서0 |
| 관측·출력 | Actor78 / Critic92 / Action20 |

학습 환경에는 마찰·질량·COM·기본각·PD gain 랜덤화, 외력·외부 토크, 속도 kick이 있다.
Play는 해당 인위적 외란과 랜덤화를 끄는 설정이다.
기본각 랜덤화는 좌우 대칭 부호를 지키도록 변경했다.
학습은 waypoint 보정 없이 속도 명령을 추종한다.

## 입력 속도

환경마다 명령을 2~4초 간격으로 재추출한다. reset에서도 명령이 바뀔 수 있다.

| 명령 | 샘플링 범위 |
|---|---|
| 전후 vx | -1.5~3.0m/s |
| 좌우 vy | -1.5~1.5m/s |
| yaw-rate | -1.6~1.6rad/s |

명령 표본 기준 대략:
- 45%: 전진0.2~3m/s, vy=0, yaw-rate=0.
- 10%: 정지.
- 45%: 기존 단계별 단일 축·조합 명령 분포.

전진 전용 확률50%를 적용한 뒤 정지 확률10%를 적용하므로 45/10/45가 된다.
기존 분포에도 직진 표본이 포함되어 총 직진 비중은 더 높다.
이는 표본 추출 확률이며 실제 학습 시간 비율의 정확한 보장은 아니다.

1→2→3m/s 순서로 속도를 늘리는 curriculum은 아니다.
처음부터 최대3m/s 명령을 샘플링한다.
기존 분포의 curriculum은 주로 활성 명령 축의 조합 비율을 바꾼다.
3m/s 명령을 학습했다는 사실은 실제3m/s 달성이나 안정성을 보장하지 않는다.

## 서로 다른 '속도'와 '주기'

| 개념 | 의미 |
|---|---|
| 명령 속도 m/s | 정책에 요구하는 이동 속도 |
| 실제 속도 m/s | 시뮬레이션에서 측정한 이동 속도 |
| 보행 위상1.5Hz | 보상·관측이 참조하는 보행 위상 진행 속도; 실제 보행이 반드시 일치한다는 뜻은 아님 |
| 정책50Hz | 관측을 받아 관절 목표를 계산하는 빈도 |
| 물리500Hz | 시뮬레이션 적분 주기 |
| 학습률 | 정책 파라미터 업데이트 크기를 조절하는 값 |
| iteration time | 실제 컴퓨터가 iteration 하나를 처리한 시간 |

## PPO 학습

- 초기 learning rate: 2.5e-6, adaptive schedule.
- rollout: 환경당24 정책 step.
- iteration당 1,024 × 24 = 24,576개 환경 전이 수집.
- 수집 데이터를 minibatch4개로 나눠 5 epoch 학습.
- 최대30,000 iteration, checkpoint 저장 간격100 iteration.
- Actor/Critic 은닉층512·256·128, ELU, 관측 정규화.
- 팔 포함20관절 대칭 데이터 증강과 mirror loss0.005 유지.

epoch5와 iteration30,000은 다른 단위다.
정책50Hz라는 이유로 학습이 현실 시간과 같은 속도로 진행되는 것은 아니다.
실제 계산 시간은 서버 로그의 Iteration time / Time elapsed로 확인한다.
이번 실행의 실제 소요 시간은 아직 이 문서에서 확인하지 않았다.

## 이번 실행에 대해 확인된 기록

사용자가 제공한 파일 목록:
- 실행 폴더: 2026-11-16_02-42-37_v1_velocity_straight
- 최종 확인 파일: model_29999.pt
- 서버 프로젝트: /home/user/sjy_vWalk_V1_upload_20261116_024158/sjy_vWalk_V1
- 로그 하위: logs/rsl_rl/sjy_vWalk_V1/

최종 checkpoint의 존재는 확인됐으나, 평가 결과는 아직 분석 전이다.
ZIP 기반 실행이므로 GitHub에 후속 추가한 train_recorded.py가 실행됐다고 간주하지 않는다.
실제 설정은 해당 실행의 params로 대조한다.

## 평가 해석

play --vx 1은 전진1m/s 명령이다. 실제 속도1m/s를 의미하지 않는다.
evaluate_velocity.py는 waypoint 보정 없이 고정 명령을 평가하고 CSV·JSON을 저장한다.
초기2초 제외 평균 전진속도·속도RMSE, 최대 횡이탈·heading 오차와 전진거리를 확인한다.
속도는 시작 방향에 투영한 root-link 속도이며 COM 속도와 구분한다.
자동 영상 녹화는 없다.

## 직접 문서를 올리는 법

GitHub 웹에서 sjy-vwalk-v1 브랜치를 선택한다.
새 문서는 Add file → Create new file로 이름.md와 내용을 입력해 commit한다.
기존 문서는 파일의 편집 버튼으로 수정하고 Commit changes로 저장한다.
변경 대상 브랜치가 sjy-vwalk-v1인지 확인한다. main에 병합할 필요는 없다.

로컬 Git clone을 완료했다면 PowerShell에서 저장소 루트에서 실행한다:

```powershell
git switch sjy-vwalk-v1
git pull --ff-only origin sjy-vwalk-v1
# 문서를 수정하고 저장한 뒤
git diff -- V1_TRAINING_SETTINGS_KO.md
git add V1_TRAINING_SETTINGS_KO.md
git diff --cached
git commit -m "docs: update V1 training settings"
git push origin sjy-vwalk-v1
```

git add는 올릴 파일 선택, commit은 로컬 이력 저장, push는 GitHub 업로드이다.
기존 ZIP 압축 해제 폴더는 자동으로 Git 작업본이 되지 않는다.
웹에서 수정한 뒤에는 로컬 수정 전에 pull한다.
실행 중인 서버 학습 worktree는 수정하거나 pull하지 않는다.
