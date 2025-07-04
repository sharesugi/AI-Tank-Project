# ✨ 사격 시연 영상

[![Video Label](http://img.youtube.com/vi/jnq8N-Le7EA/maxresdefault.jpg)](https://youtu.be/jnq8N-Le7EA)

# 🧠 YOLO 모델 학습

 ### 📌 1. 시뮬레이션 환경 및 데이터 생성
 각 객체들은 맵에 랜덤한 위치로 생성되고, 탱크를 랜덤으로 이동시키며 `save snapshot` 기능을 통해 이미지 1000장 생성

 ### 📌 2. 라벨링 및 데이터셋 구성
  - 클래스 : car1, car2, human, tank
  - 데이터 분할
    - Train Set : 300장
    - Validation Set : 100장
    - Test Set : 50장
 
 ### 📌 3. YOLO 모델 별 성능 평가 및 선정
  - 총 12개 모델 평가
    - YOLOv8 : yolov8n, yolov8s, yolov8m
    - YOLOv9 : yolov9t, yolov9s, yolov9m
    - YOLOv10 : yolov10n, yolov10s, yolov10m
    - YOLOv11 : yolov11n, yolov11s, yolov11m

  - 평가 결과
    |모델명|mAP@50-95|mAP@50|mAP@75|
    |---|---|---|---|
    |yolov9m_final|0.600|0.950|0.687|
    |yolov8s_final|0.599|0.945|0.685|
    |yolov10m_final|0.598|0.944|0.684|
    |yolo11m_final|0.597|0.943|0.686|
    |yolov9t_final|0.597|0.946|0.682|
    |yolo11n_final|0.596|0.945|0.685|
    |yolov10s_final|0.595|0.948|0.685|
    |yolov9s_final|0.594|0.943|0.649|
    |yolov10n_final|0.592|0.939|0.655|
    |yolo11s_final|0.590|0.941|0.653|
    |yolov8n_final|0.587|0.938|0.643|
    |yolov8m_final|0.586|0.940|0.610|

    ![image](https://github.com/user-attachments/assets/612387bf-49c9-4283-b101-ba42585965b7)

 ### 📌 4. 결론
 실시간 감지를 위해 비교적 가볍고 성능이 좋은 YOLOv8s 모델 채택

# 🧠 포탑 각도 예측 모델 학습

 ### 📌 1. 시뮬레이션 환경 및 데이터 생성
 탱크 위치와 포탑의 각도를 무작위로 변경시키며 사격 시 `탱크의 위치, 포탑의 각도와 떨어지는 포탄의 위치 데이터` 10000개 생성

 ### 📌 2. Feature Engineering
  - 입력 특징 조합
    - **Position Only**: `x/y/z_pos`, `x/y/z_target`
    - **Position + Distance**: `position` + `distance`
    - **Distance Only**: `distance` (직선 거리)
    - **Distance + dy**: `distance` + `dy (y_pos - y_target)`
  
  - 입력 조합별 성능 (DNN 모델 기준)
    |Input Type|MAE_y|RMSE_y|R²_y|
    |---|---|---|---|
    |Position Only|1.033246|1.340617|0.984256|
    |Position + Distance|0.28633|0.525747|0.995579|
    |Distance Only|4.978907|9.497897|0.209751|
    |Distance + dy|0.2659 ✅|0.5227 ✅|0.9976 ✅|

    ![image](https://github.com/user-attachments/assets/a8fe2250-441e-4930-8e6f-997f64b32adf)
    ![image](https://github.com/user-attachments/assets/5d480de2-eab7-4041-9ed8-fcac07f77ccb)

 ### 📌 3. 모델별 성능 비교 (입력: Distance + dy)
  |Model|MAE_y|RMSE_y|R²_y|
  |---|---|---|---|
  |DNN|0.2659|0.5227|0.9976|
  |XGBoost|0.3589|0.6620|0.9961|
  |LightGBM|0.3586|0.6704|0.9960|
  |DNN + XGBoost|0.2531 ✅|0.5051 ✅|0.9977 ✅|
  |DNN + LightGBM|0.2540|0.5082|0.9977 ✅|
  |XGBoost + LightGBM|0.3346|0.6382|0.9964|
  |DNN + XGBoost + LightGBM|0.2660|0.5294|0.9975|

  ![image](https://github.com/user-attachments/assets/feb168f1-c679-4160-829f-a839b31f035d)
  ![image](https://github.com/user-attachments/assets/60e746d0-240b-41f9-8fd6-7729ceb2691f)

 ### 📌 4. Optuna를 통한 하이퍼파라미터 튜닝
 최적 조합 결과
  - DNN: 256 units, lr=0.00018
  - XGBoost: n=199, depth=6, lr=0.06525
    
  |Model|MAE_y|RMSE_y|R²_y|
  |---|---|---|---|
  |DNN + XGBoost|0.2531|0.5051|0.9977|
  |DNN + XGBoost (optuna)|0.1563 ✅|0.3552 ✅|0.9988 ✅|

 ### 📌 5. 알고리즘 작동 방식

 ![사격 ppt](https://github.com/user-attachments/assets/6abcdb3e-78ea-44ea-8f1f-2ad73fa6863a)

 
