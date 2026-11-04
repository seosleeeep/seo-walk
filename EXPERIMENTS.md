# seo-walk 개인 실험 기록

## 버전 기준

- `seo-walk-v0`: 처음 서버로 전송한 `seo_walk.zip`의 134개 파일을 그대로 보존한 기준 커밋. 서버 최신본을 다시 받은 것은 아니다.
- `seo-walk-v1`: Windows에서 수정한 전진 추종·보폭·몸통 높이 제약 완화 설정. 서버 적용 및 학습 검증 전.
- 원본 코드 출처: https://github.com/Inha-united-soccer/INHA-WAlk . 기존 LICENSE와 README를 유지한다.
- 개인 원격 저장소: https://github.com/seosleeeep/seo-walk . 팀 저장소에는 push하지 않는다.

## v0 서버 학습 기록 (사용자가 제공한 결과)

| 실행 폴더 | 마지막 checkpoint | 확인된 상태 |
|---|---|---|
| 2026-11-03_21-38-07_seo_20dof_first | model_499.pt | 500회 학습, export 후 MuJoCo 실행 |
| 2026-11-03_22-37-29_seo_20dof_continue | model_998.pt | 파일 목록 확인 |
| 2026-11-03_22-56-09_seo_20dof_continue | model_998.pt | 파일 목록 확인 |
| 2026-11-03_23-09-18_seo_20dof_continue | model_998.pt | export 후 MuJoCo 실행 |

서버 로그 상위 경로: `/home/user/seo/logs/rsl_rl/seo_walk/`.
continue 명령은 최초 model_499.pt에서 500회를 추가하는 방식이었다. 세 실행을 연속 누적 학습으로 해석하지 않는다.
체크포인트와 export 파일 자체는 로컬에 없고 Git에 포함하지 않는다.

관찰: 로봇이 대체로 서 있지만 낮은 전진 명령에서 제자리걸음과 회전이 발생했고, 큰 명령에서도 전진이 작았다. 정량 속도·낙상률은 측정하지 않았다. 배포 명령 3/10은 설정상 전진 1.5m/s로 제한된다.

학습 actuator와 MuJoCo의 토크 제한·지연 모델 차이가 발견되었다. 실제 런타임 관절 순서 일치 여부는 미검증이다. 보행 실패 원인을 학습 부족이나 보상만으로 확정하지 않는다.

## v1

Task: `seo_walk_v1`, `seo_walk_v1-Play`.
상세 변경: [V1_CHANGES_KO.md](source/JY_train/JY_train/tasks/manager_based/scratch_symmetry/robots/k1/seo_walk/V1_CHANGES_KO.md).
문법과 격리된 설정 검사 및 보상 계산 비교만 완료. Isaac 런타임 검증, 학습, 보행 성능 검증은 미실시.
서버에서 수정한 booster_deploy 전체 코드는 이 저장소에 포함되지 않는다.

## 이후 기록 양식

```text
실험 이름:
Git commit (git rev-parse HEAD):
작업 폴더 변경 여부 (git status --short):
Task:
초기 checkpoint / 처음부터 학습:
변경한 보상·설정:
시드 / 환경 수 / 추가 학습 횟수:
서버 로그 경로 / 최종 checkpoint:
Export 경로와 원본 checkpoint:
평가 명령 (vx, vy, yaw):
실제 속도 / yaw 오차 / 낙상 / 몸통 높이 범위:
영상 경로:
결과와 다음 변경:
```

모델·영상은 별도 보관하고 이 문서에는 경로만 기록한다. Export 파일명에 checkpoint 번호가 없으므로 재생한 원본 checkpoint를 반드시 남긴다.
