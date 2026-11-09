# Seo vv0 mjlab 독립 이식 진행 상태

2026-11-07. 독립 학습 패키지 코드를 작성했다. 실제 서버 GPU 검증과 학습은 아직 실행하지 않았다.
오프라인 문법·API 생성자·원본 보상 수식·관절 및 미러링·팔 비용 수학 검사를 통과했다.

## 소스 기준

- 사용자 프로젝트 seo_walk_vv0의 관측, 명령, 보상, 학습 설정.
- 공식 mjlab API와 MuJoCo K1 모델.
- LSH_mjlab/lsh_walk와 INHA-Recovery-Motion의 task/정책/보상 소스는 사용하지 않는다.
- 서버 공용 가상환경과 다른 프로젝트 파일은 수정하지 않는다.

## 목표 계약

- 20개 팔다리 관절 제어. 머리 2관절은 고정 또는 학습 action 밖에서 유지.
- 원본 vv0와 동등한 action 20 / actor 78 / critic 92 계약을 구현하고 실제 로드 시 검증.
- 명목 자세 기준 위치 제어, action scale 1.0, 정책 주기 0.02초.
- 팔 deviation -0.05와 팔 전용 torque 제곱 합 -0.0002.
- 기존 전체 관절 torque -0.0002를 유지하므로 팔의 해당 두 비용 합은 -0.0004.
- heading drift -2, yaw 속도 추종 +1.5, 전신 각운동량 비용 없음.
- 전진 명령 상한 +3.0m/s, 후진 하한 -1.5m/s.
- reward manager 시간 적분과 원시 비용의 단위를 확인해 계수의 의미를 보존.
- 임의의 공식 G1 보상을 K1 vv0 보상 대신 사용하지 않는다.

## 서버 확인값

- mjlab 1.6.0
- MuJoCo / MuJoCo-Warp 3.11.0
- rsl-rl-lib 5.5.0
- Python 3.13

공식 upstream v1.6.0 pyproject는 rsl-rl-lib 5.4.2를 지정하지만 서버는 5.5.0이다.
공용 환경을 임의로 downgrade하지 않는다. 실제 설치 API와 실행 검사를 기준으로 호환성을 확인한다.

## 자료 수신 후 구현 상태

공통 K1 MJCF 및 실제 공식 API snapshot을 수신해 로봇·관측·명령·보상·센서·등록·학습/재생 스크립트를 구현했다.
mesh는 서버의 booster_assets 경로를 참조한다.
실행 프로세스에서만 다른 개인 task entry point의 자동 import를 차단하며 설치 파일은 수정하지 않는다.
다음 순서는 preflight → GPU 16환경 smoke → PPO 2 iteration → 본 학습이다.
모터·마찰·가속도 수치의 이식 차이는 README_KO.md에 명시했다.
기존 Isaac checkpoint의 직접 재개는 지원하지 않는다.
