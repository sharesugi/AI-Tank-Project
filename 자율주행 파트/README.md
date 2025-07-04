# ✨ 주행 시연 영상

[![Video Label](http://img.youtube.com/vi/bz-GHivsUG8/maxresdefault.jpg)](https://youtu.be/bz-GHivsUG8)

# 🧠 A* 알고리즘 적용

### 📌 1. 휴리스틱 함수 선정

시작 지점, 장애물 위치 고정 후, 각 휴리스틱 함수마다 100회씩 수행 후 평가 (평균)

|함수|시간 (s)|거리 (m)|충돌 횟수|
|---|---|---|---|
|Diagonal|29.611 ✅|276.744 ✅|0.11 ✅|
|Manhatten|33.814|328.587|1.667|
|Euclidean|94.082|340.101|2.334|

### 📌 2. Buffer 추가

최단 경로를 구하는 A* 알고리즘 특성 상, 장애물과 근접하게 경로 생성 → ***충돌 발생***<br>
따라서 장애물 주변에 Buffer를 추가하여 기존 장애물보다 더 크게 인식하도록 만듦 → ***충돌 해결***

![image](https://github.com/user-attachments/assets/2148c16a-5f41-4123-8a76-c79fd7e8311a)

# 🧠 알고리즘 작동 방식

![주행 pptmp4](https://github.com/user-attachments/assets/d5f985e6-dbe2-45a4-bef2-3da574696c39)
