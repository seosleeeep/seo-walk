# sjy_vWalk_V0

vv3_waypoint_1112.zip 기반 팔 자세 보상 수정본. Waypoint task를 유지합니다.
패키지 seo_walk_vv3와 TASK_ID는 호환성을 위해 원본 그대로입니다.
기존 코드와 동시에 설치하지 말고 이 폴더의 scripts/train.py를 직접 실행하세요.

## 변경
- arm_swing_center 제거, arm_deviation을 arm_posture로 대체.
- 별도 arm_torque 제거. 기존 전체 관절 torques/power 등은 유지.
- straight_pose_mirror는 다리 12개 관절만 평균. 원래 20관절 평균과 정규화가 달라집니다.
- arm_posture: mean(((q_arm - default_joint_pos_arm) / std)^2), weight -0.1.
- default_joint_pos는 관측/action offset과 같은 환경별 기준. 팔 HOME 값 변경 없음.
- activity = norm(command[:2]) + abs(command[2]). yaw를 포함한 휴리스틱 지표입니다.
- activity <= 0.1: std=0.05 rad (팔 보상에서만 정지 취급).
- 0.1~1.0: std 0.05 -> 0.15 선형 보간.
- 1.0~2.0: std 0.15 -> [0.5,0.2,0.35,0.35] 선형 보간, 좌우 동일.
- 2.0 이상: 최종 std 유지.
- 관절 순서: 각 측 Shoulder Pitch/Roll, Elbow Pitch/Yaw.
- std는 비용 정규화 기준이지 관절 제한이나 clipping이 아닙니다.
- 로그 experiment_name/run_name: sjy_vWalk_V0.

## 유지 및 주의
Waypoint, 0.3~3.0m/s 목표 속도 샘플링, gait 정지 판정, actor/critic 78/92,
action 20, 모터 gain/torque/action scale, PPO 설정을 원본대로 유지했습니다.
원본 PPO symmetry augmentation과 mirror loss도 여전히 활성 상태입니다.
새 Velocity Walk README 전체를 이식한 버전이 아닙니다. HOME/토크 곡선/관측 72/86 등은 가져오지 않았습니다.
원본 기본 자세 랜덤화 +/-0.03rad는 유지되어 개별 환경의 기준은 엄밀한 좌우 대칭이 아닐 수 있습니다.
대칭 비용은 대칭 동작을 보장하지 않습니다.
기존 checkpoint와 입출력 구조는 같지만, 보상이 달라졌으므로 우선 새 학습을 권장합니다.

## 실행
ZIP을 홈에 풀면 ~/sjy_vWalk_V0/ 아래에 프로젝트가 생깁니다.
로그 권한 문제를 피하려고 홈에서 스크립트를 절대경로로 실행합니다.

```bash
source ~/miniconda3/bin/activate
conda activate inha_mjlab
cd ~
export SEO_K1_MESH_DIR=/home/user/booster_assets/robots/K1/meshes
python ~/sjy_vWalk_V0/scripts/test_arm_posture.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u ~/sjy_vWalk_V0/scripts/train.py --gpu-ids 0 --env.scene.num-envs 32 --agent.max-iterations 1 --agent.run-name sjy_vWalk_V0_smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u ~/sjy_vWalk_V0/scripts/train.py --gpu-ids 0 --env.scene.num-envs 1024 --agent.max-iterations 20000 --agent.run-name sjy_vWalk_V0_20k
```
위 smoke 실행이 성공하는지 확인한 뒤 본 학습 명령을 실행하세요.

## 검증
Python 전체 문법 검사와 NumPy tensor adapter를 통한 실제 보상 함수의 수치 검사를 완료했습니다.
0 기준 비용, 0.1 경계, 연속 보간, 명령 증가에 따른 비용 감소, 반사 대칭을 검사했습니다.
PyTorch 단위검사는 scripts/test_arm_posture.py로 제공합니다.
제작 단계에서는 PyTorch/mjlab 실행을 검증하지 못했습니다. GPU smoke 및 학습 성능은 미검증입니다.
모든 팔 관절이 기준에서 0.1rad 이탈할 때 가중 비용(환경 dt 곱하기 전):
activity 0.0/0.1: -0.4, 1.0: -0.044444, 2.0: -0.011332.
