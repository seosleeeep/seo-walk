# VV2 직전 K1 보행 기준 실험

- 서버 실행명: `2026-11-08_14-44-35_vv0_arm_limit_3000`
- 기준 체크포인트: `model_15996.pt`. 서버 절대 경로는 [checkpoint_path.txt](last_run/checkpoint_path.txt)에 기록했다. 가중치 파일은 서버에 있다.
- 소스 출처: 2026-11-09 22:51:55에 서버에서 묶은 `seo_mjlab_for_vv2_20261109_225155.zip`. 학습 당시 설정은 [env.yaml](last_run/params/env.yaml), [agent.yaml](last_run/params/agent.yaml) 참조. 소스는 실행 후 백업 시점의 사본이다.
- 제어: 다리 12축과 팔 8축. 백업 코드에서 양쪽 어깨·팔꿈치 pitch의 action scale은 0.3, 어깨 roll과 팔꿈치 yaw는 0.05다. 각도 제한값이 아니라 정책 출력을 목표각 변화량으로 바꾸는 계수다.
- 사용자 관찰: 이전에 보이던 한쪽 팔이 위로 들리는 ㄴㅇㄱ 자세가 이 버전에서는 보이지 않았다. 팔의 좌우 차이와 경직감은 남았다. 관절각과 좌우 편차의 정량 평가는 기록되지 않았다.
