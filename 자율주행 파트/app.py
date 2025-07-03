# raw points 빼버림
from flask import Flask, request, jsonify
from queue import PriorityQueue
import os
import torch
from ultralytics import YOLO
import math
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
import time  # 추가0605
import numpy as np
from queue import PriorityQueue

# Flask 앱 초기화 및 YOLO 모델 로드
app = Flask(__name__)
model = YOLO('yolov8n.pt')


# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 20
start_z = 20
start = (start_x, start_z)
# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
destination_x = 260 # 기존에는 destination과 적 전차 위치를 똑같이 줬으나, LiDAR로 물체를 감지할 경우 적 전차도 감지해서 장애물이라 생각하고 목표에 끝까지 도달을 안함. 그래서 이제부터 따로 줌.
destination_z = 260
destination = (destination_x, destination_z)
print(f"🕜️ 초기 destination 설정: {destination}")

INITIAL_YAW = 0.0  # 초기 YAW 값 - 맨 처음 전차의 방향이 0도이기 때문에 0.0 줌. 이를  
current_yaw = INITIAL_YAW  # 현재 차체 방향 추정치 -> playerBodyX로 바꾸면 좋겠으나 실패... playerBodyX의 정보를 받아 오는데 딜레이가 걸린다면 지금처럼 current_yaw값 쓰는게 좋다고 함(by GPT)
previous_position = None  # 이전 위치 (yaw 계산용)
target_reached = False  # 목표 도달 유무 플래그
current_angle = 0.0  # 실제 플레이어의 차체 각도 저장용 (degree) -> playerBodyX 받아오는 방법 사용해 볼 것임.
collision_count = 0  # 충돌 횟수 카운터 추가
total_distance = 0

# 시각화 관련 부분
current_position = None
last_position = None
position_history = []
original_obstacles = []  # 원본 장애물 좌표 저장용 (버퍼 없이)
collision_points = [] # 전역변수에 collision point 추가(충돌 그림에 필요)

# 충돌 없을 때 파일 저장
with open('collision_points.json', 'w') as f:
    json.dump({
        "collision_count": 0,
        "collision_points": []
    }, f, indent=2)

# 시간 세는 부분
start_time = None
end_time = None

# A* 알고리즘 관련 클래스 및 함수 정의
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b): # Diagonal (Octile) 방식으로 heuristic 변경
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def get_neighbors(pos):
    neighbors = []
    for dx, dz in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        x, z = pos[0] + dx, pos[1] + dz
        if 0 <= x < GRID_SIZE and 0 <= z < GRID_SIZE:
            # 대각선 이동일 경우 추가 확인
            if dx != 0 and dz != 0:
                if maze[pos[1]][x] == 1 or maze[z][pos[0]] == 1:
                    continue  # 대각선 경로에 인접한 직선 중 하나라도 막혀있으면 skip # 즉 모서리를 못 뚫고 지나가게 수정
            if maze[z][x] == 0: 
                neighbors.append((x, z))
    return neighbors

def a_star(start, goal):
    open_set = PriorityQueue()
    counter = 0  # 시간 순서 유지용
    
    open_set.put((0, -counter, Node(start)))  # cost=0, 최신 순서 우선
    closed = set()

    while not open_set.empty():
        _, _, current = open_set.get()

        if current.position == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        closed.add(current.position)

        for nbr in get_neighbors(current.position):
            if nbr in closed:
                continue
            node = Node(nbr, current)

            dx = abs(nbr[0] - current.position[0])
            dz = abs(nbr[1] - current.position[1])
            step_cost = math.sqrt(2) if dx != 0 and dz != 0 else 1

            node.g = current.g + step_cost
            node.h = heuristic(nbr, goal)
            node.f = node.g + node.h

            counter += 1
            open_set.put((node.f, -counter, node))  # 최신 노드 우선

    return [start]

# 현재 위치와 다음 위치 간 각도 계산 함수
def calculate_angle(current, next_pos): # A*알고리즘을 통해서 어디로 갈지 전체 경로를 정했기 때문에 다음 위치로만 가면 됨.
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]
    return (math.degrees(math.atan2(dz, dx)) + 360) % 360

# 장애물 맵 유효 위치 확인
def is_valid_pos(pos, size=GRID_SIZE): # 장애물이 300x300 안에 있는지 확인
    x, z = pos
    return 0 <= x < size and 0 <= z < size

# Flask API 라우팅 시작
@app.route('/init', methods=['GET'])
def init():
    global current_yaw, previous_position, target_reached
    current_yaw = INITIAL_YAW
    previous_position = None
    target_reached = False

    config = {
        "startMode": "start",
        "blStartX": start_x, "blStartY": 10, "blStartZ": start_z,
        "rdStartX": 150, "rdStartY": 10, "rdStartZ": 270,
        "trackingMode": True, "detectMode": False, "logMode": True,
        "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000
    }
    print("🛠️ /init config:", config)
    return jsonify(config)

def calculate_actual_path():
    global total_distance
    
    if len(position_history) > 1:
        for i in range(len(position_history) -1):
            x1, z1 = position_history[i] # 이전 좌표
            x2, z2 = position_history[i+1] # 현재 좌표
            step_distance = math.sqrt((x2 - x1)**2 + (z2 - z1)**2) # 가장 최근 두 지점의 좌표 추출
            total_distance += step_distance                        # 지금 이동한 거리(step_distance)를 누적 거리(total_distance)에 더함
    return total_distance

    
# 여기 리스트에 cmd 2개를 넣는다
combined_command_cache = []

@app.route('/get_action', methods=['POST'])
def get_action():
    global target_reached, previous_position, current_yaw, current_position, last_position
    global start_time, end_time
    data = request.get_json(force=True)
    pos = data.get('position', {})
    pos_x = float(pos.get('x', 0))
    pos_z = float(pos.get('z', 0))

    # tracking_mode가 True일 때만 시간 측정 시작
    if start_time is None: # 추가0605
        start_time = time.time()  
        print("🟢 trackingMode 활성화: 시간 기록 시작")  
        
    if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0:
        target_reached = True  
        end_time = time.time()  # 추가0605
        elapsed = end_time - start_time  
        print(f"⏱️ 도착까지 걸린 시간: {elapsed:.3f}초")
        print(f"이동거리: {calculate_actual_path():.3f}")
        print("✨ 목표 도달: 전차 정지 플래그 설정")
        
    if target_reached:
        stop_cmd = {k: {'command': 'STOP', 'weight': 1.0} for k in ['moveWS', 'moveAD']}
        return jsonify(stop_cmd)

    if previous_position is not None:
        dx = pos_x - previous_position[0]
        dz = pos_z - previous_position[1]
        if math.hypot(dx, dz) > 0.01:
            current_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
    previous_position = (pos_x, pos_z)

    current_grid = (int(pos_x), int(pos_z))
    path = a_star(current_grid, destination)

    #######################################################################
    # 2 좌표 이동한 후. astar(현좌표, 최종목적지) 함수 실행해서 path 새로 뽑기 반복

    if combined_command_cache:
    # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
        cmd = combined_command_cache.pop(0)
        return jsonify(cmd)
    
    if len(path) > 2:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
        next_grid = path[1:3]  # 1~2번째 좌표 참조
    elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
        next_grid = [path[1]]      # 한개씩 참조  
    else:
        next_grid = [current_grid]   # 0개면 멈춰라! 도착한거니까!

    for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1

        # next_grid[1]의 회전 각도는 current 가 아니라 next_grid[0]에서 계산해야 맞음 
        base_pos = current_grid if i == 0 else next_grid[i - 1]  
    
        if not is_valid_pos(next_grid[i]):  # 가야하는 곳이 맵 외에 있으면 움직이는거 멈춤
            stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
            stop_cmd['fire'] = False
            return jsonify(stop_cmd)

        target_angle = calculate_angle(base_pos, next_grid[i])  # 현재 좌표에서 두번째 좌표로
        diff = (target_angle - current_yaw + 360) % 360   # 현 각도랑 틀어야할 각도 차이 알아내고
        if diff > 180:  # 이거는 정규화 비슷
            diff -= 360

        # 이건 그냥 유클리드 거리. sqrt는 제곱근! 현위치랑 목적좌표까지의 거리 
        distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)

        if distance < 20 :
            w_weight = 0.1
            acceleration = 'S'
        else :
            w_weight = 0.2
            acceleration = 'W'

        abs_diff = abs(diff)
        if 0 < abs_diff < 30 :  
            w_degree = 0.3
        elif 30 <= abs_diff < 60 :    
            w_degree = 0.6
            stop = True
        elif 60 <= abs_diff < 90 : 
            w_degree = 0.75
        else :
            w_degree = 1.0
    
        forward = {'command': acceleration, 'weight': w_weight}
        turn = {'command': 'A' if diff > 0 else 'D', 'weight': w_degree}

        cmd = {
            'moveWS': forward,
            'moveAD': turn
        }

        combined_command_cache.append(cmd)   # 두 좌표에 대한 명령값 2개가 여기 리스트에 저장됨

    # 처음 1회 A* 경로 계산_ 기홍님이 새로 추가
    if len(position_history) == 0:
        path = a_star((int(pos_x), int(pos_z)), destination)  # 현 위치에서 최종 목적지까지 다시 계산
        df = pd.DataFrame(path, columns=["x", "z"])
        df.to_csv("a_star_path.csv", index=False)

    
    if current_grid:
        last_position = current_grid
    position_history.append(current_grid)
    
    df = pd.DataFrame(position_history, columns=["x", "z"])
    df.to_csv("tank_path0.csv", index=False)


    # print문 살짝 수정-희연
    print(f"📍 현재 pos=({pos_x:.1f},{pos_z:.1f}) yaw={current_yaw:.1f} 두번째 좌표로 가는 앵글 ={target_angle:.1f} 차이 ={diff:.1f}")
    print(f"🚀 cmd 2개 {combined_command_cache}")
    return jsonify(combined_command_cache.pop(0))


@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination
    data = request.get_json()
    if not data or 'destination' not in data:
        return jsonify({'status': 'ERROR', 'message': 'Missing destination'}), 400
    try:
        x, y, z = map(float, data['destination'].split(','))
        destination = (int(x), int(z))
        print(f"🎯 destination set to: {destination}")
        return jsonify({'status': 'OK', 'destination': {'x': x, 'y': y, 'z': z}})
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 400

@app.route('/start', methods=['GET'])
def start():
    print('start')
    return jsonify({'control': ''})

@app.route('/collision', methods=['POST'])
def collision():
    global collision_points, collision_count
    d = request.get_json(force=True)
    p = d.get('position', {})
    x = p.get('x')
    z = p.get('z')

    if x is not None and z is not None:
        collision_points.append((x, z))
        collision_count += 1  # 충돌 횟수 증가

        # 저장 파일 구조: 충돌 좌표 목록과 총 횟수 포함
        save_data = {
            "collision_count": collision_count,
            "collision_points": collision_points
        }

        with open('collision_points.json', 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"💥 Collision #{collision_count} at ({x}, {z})")

    return jsonify({'status': 'success', 'collision_count': collision_count})

@app.route('/info', methods=['POST'])
def info():
    global maze, original_obstacles

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    lidar_data = data.get("lidarPoints", [])
    buffer = 3
    # raw_points = []

    for point in lidar_data:
        if point.get("isDetected") and abs(point.get("verticalAngle", 999)) < 2:
            pos = point.get("position", {})
            x = int(pos.get("x", -1))
            z = int(pos.get("z", -1))

            if 0 <= x < GRID_SIZE and 0 <= z < GRID_SIZE:
                # raw_points.append((x, z))

                # ✅ 버퍼 영역 마킹
                for dx in range(-buffer, buffer + 1):
                    for dz in range(-buffer, buffer + 1):
                        nx, nz = x + dx, z + dz
                        if 0 <= nx < GRID_SIZE and 0 <= nz < GRID_SIZE:
                            maze[nz][nx] = 1

                # ✅ 원본 장애물 저장
                x_min_raw = x_max_raw = x
                z_min_raw = z_max_raw = z
                
                original_obstacles.append({
                    "x_min": x_min_raw,
                    "x_max": x_max_raw,
                    "z_min": z_min_raw,
                    "z_max": z_max_raw
                })

    # 저장
    try:
        with open("original_obstacles.json", "w") as f:
            json.dump(original_obstacles, f, indent=2)
        print("✅ 장애물 저장 완료")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

    return jsonify({"status": "success"})

# 서버 실행
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5009)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 감지됨 (Ctrl+C)")
    finally:
        print(f"📊 총 충돌 횟수: {collision_count}회")