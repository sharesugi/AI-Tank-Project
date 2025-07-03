import math
from ultralytics import YOLO
from flask import Flask, request, jsonify
from queue import PriorityQueue

import fire, drive

app = Flask(__name__)
model_yolo = YOLO('./Models/best_8s.pt')

# 화면 해상도 (스크린샷 찍었을 때 이미지 크기)
IMAGE_WIDTH = 1921
IMAGE_HEIGHT = 1080

# 카메라 각도
FOV_HORIZONTAL = 47.81061 
FOV_VERTICAL = 28       

# 적 전차를 찾는 상태
DRIVE_MODE = True

# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 55
start_z = 55
start = (start_x, start_z)

# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
dest_list = [(245, 55), (245, 245), (55, 245), (55, 80)]
dest_idx = 0

INITIAL_YAW = 0.0  # 초기 YAW 값 - 맨 처음 전차의 방향이 0도이기 때문에 0.0 줌. 이를  
current_yaw = INITIAL_YAW  # 현재 차체 방향 추정치 -> playerBodyX로 바꾸면 좋겠으나 실패... playerBodyX의 정보를 받아 오는데 딜레이가 걸린다면 지금처럼 current_yaw값 쓰는게 좋다고 함(by GPT)
previous_position = None  # 이전 위치 (yaw 계산용)
target_reached = False  # 목표 도달 유무 플래그

GRID_SIZE = 300  # 맵 크기
astar_how_many_implement = 0

enemy_pos_locked = False
locked_enemy_pos = None
lock_frame_counter = 0
LOCK_TIMEOUT = 100  # 예: 100프레임 = 인터벌 * 100초

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
    global astar_how_many_implement

    astar_how_many_implement+=1 # 0616 희연님 코드 추가
    open_set = PriorityQueue()
    open_set.put((0, Node(start)))
    closed = set()
    while not open_set.empty():
        _, current = open_set.get()
        if current.position == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        closed.add(current.position)
        for nbr in get_neighbors(current.position):
            if nbr in closed: continue
            node = Node(nbr, current)

            # 이 부분 추가함.
            dx = abs(nbr[0] - current.position[0])
            dz = abs(nbr[1] - current.position[1])
            step_cost = math.sqrt(2) if dx != 0 and dz != 0 else 1

            node.g = current.g + step_cost
            node.h = heuristic(nbr, goal)
            node.f = node.g + node.h
            open_set.put((node.f, node))
    return [start]

lidar_data = [] # /info 에서 가져오는 라이다 데이터 저장

@app.route('/detect', methods=['POST'])
def detect():
    global lidar_data, enemy_pos, DRIVE_MODE, yolo_results
    global enemy_pos_locked, locked_enemy_pos, lock_frame_counter

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model_yolo(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    is_tank = False
    target_classes = {3: "tank"}
    filtered_results = []
    current_bboxes = []

    for i, box in enumerate(detections):
        if box[4] >= 0.70:
            class_id = int(box[5])
            if class_id == 3:
                is_tank = True
                current_bboxes.append({
                    'id': i,
                    'x1': float(box[0]), 'y1': float(box[1]),
                    'x2': float(box[2]), 'y2': float(box[3])
                })

            if class_id in target_classes:
                filtered_results.append({
                    'className': target_classes[class_id],
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4]),
                    'color': '#00FF00',
                    'filled': False,
                    'updateBoxWhileMoving': True
                })

    # YOLO + LiDAR 매칭
    yolo_results = fire.match_yolo_to_lidar(
        bboxes=current_bboxes,
        lidar_points=lidar_data,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fov_h=FOV_HORIZONTAL,
        fov_v=FOV_VERTICAL
    )

    print(f'🗺️ yolo_results : {yolo_results}')

    for i, r in enumerate(yolo_results):
        print(f"탐지된 전차 {i+1}:")
        print(f"  바운딩 박스: {r['bbox']}")
        print(f"  LiDAR 좌표: {r['matched_lidar_pos']}")
        print(f"  거리: {r['distance']:.2f}m\n")

    ## 🔒 타겟 락온 및 추적 로직
    lock_frame_counter += 1

    if not enemy_pos_locked:
        # 락 안 걸려 있을 때 → 가장 가까운 전차 락온
        min_distance = 100
        closest_result = None
        for r in yolo_results:
            if r['distance'] < min_distance:
                closest_result = r
                min_distance = r['distance']

        if closest_result:
            matched = closest_result['matched_lidar_pos']
            enemy_pos = {
                'x': matched.get('x', 0),
                'y': matched.get('y', 0),
                'z': matched.get('z', 0)
            }
            locked_enemy_pos = matched
            enemy_pos_locked = True
            lock_frame_counter = 0
            print("🔒 타겟 락온 완료!")

    else:
        # 이미 락이 걸려 있는 상태 → 좌표 기반으로 가장 비슷한 전차 추적
        updated = False
        for r in yolo_results:
            matched = r['matched_lidar_pos']
            if fire.distance_3d(matched, locked_enemy_pos) < 2.0:  # 2m 이내면 동일 타겟으로 간주
                enemy_pos = {
                    'x': matched.get('x', 0),
                    'y': matched.get('y', 0),
                    'z': matched.get('z', 0)
                }
                locked_enemy_pos = matched
                updated = True
                break

        if not updated:
            print("⚠️ 락된 전차 위치 유실 → 락 해제")
            enemy_pos_locked = False
            locked_enemy_pos = None
            lock_frame_counter = 0

        if lock_frame_counter > LOCK_TIMEOUT:
            print("⏲️ 락 유지 시간 초과 → 해제")
            enemy_pos_locked = False
            locked_enemy_pos = None
            lock_frame_counter = 0

    if is_tank and yolo_results and enemy_pos_locked:
        DRIVE_MODE = False

    return jsonify(filtered_results)

# 아래 세 변수 모두 사격 불가능한 각도 판별할 때 사용하는 변수
angle_hist = []
save_time = 0
len_angle_hist = -1

# 여기 리스트에 cmd 2개를 넣는다
combined_command_cache = []
@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, angle_hist, save_time, len_angle_hist, DRIVE_MODE, yolo_results
    global target_reached, previous_position, current_yaw, dest_list, dest_idx
    global body_x

    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    # 현재 내 위치
    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    # 현재 터렛 각도 (x: yaw, y: pitch)
    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    destination = dest_list[dest_idx]

    print(f'🗺️ DRIVE_MODE : {DRIVE_MODE}')
    print(f'🗺️ Destination coord : {destination}')

    if DRIVE_MODE: # 적 전차를 탐색하는 상태일 때 
        if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0: # 거리 5 미만이면 도착으로 간주
            target_reached = True 
            
        if target_reached:
            if dest_idx < len(dest_list):
                target_reached = False
                dest_idx += 1

            if dest_idx >= len(dest_list):
                stop_cmd = {k: {'command': 'STOP', 'weight': 1.0} for k in ['moveWS', 'moveAD']}
                return jsonify(stop_cmd)

        if previous_position is not None:
            dx = pos_x - previous_position[0]
            dz = pos_z - previous_position[1]
            if math.hypot(dx, dz) > 0.01:
                current_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
        previous_position = (pos_x, pos_z)

        current_grid = (int(pos_x), int(pos_z))

        if combined_command_cache:
        # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
            cmd = combined_command_cache.pop(0)
            print(f"🚀 cmd 1개 {cmd}")
            return jsonify(cmd)
        elif not combined_command_cache:  # 명령어 두 개 다 실행해서 비어있으면
            path = a_star(current_grid, destination)  # 이 때만 astar 실행

        # print(f"✅ A* 경로가 {filepath} 에 누적 저장되었습니다.")
        if len(path) > 3:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
            next_grid = path[1:4]  # 1~2번째 좌표 참조
        elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
            next_grid = [path[1]]      # 한개씩 참조  
        else: 
            next_grid = [current_grid]   # 0개면 멈춰라! 도착한거니까!

        for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1
            base_pos = current_grid if i == 0 else next_grid[i - 1]  
        
            if not drive.is_valid_pos(next_grid[i]):  # 가야하는 곳이 맵 외에 있으면 움직이는거 멈춤
                stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
                stop_cmd['fire'] = False
                return jsonify(stop_cmd)

            target_angle = drive.calculate_angle(base_pos, next_grid[i])  # 현재 좌표에서 두번째 좌표로
            diff = (target_angle - current_yaw + 360) % 360   # 현 각도랑 틀어야할 각도 차이 알아내고
            if diff > 180:  # 이거는 정규화 비슷
                diff -= 360

            # 이건 그냥 유클리드 거리. sqrt는 제곱근! 현위치랑 목적좌표까지의 거리
            # 목적지 근처에 가면 천천히 뒷 키 잡으려고, 목적지까지 남은 거리를 계산.
            distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)
    
            w_weight = 0.3
            acceleration = 'W'

            abs_diff = abs(diff)
            if 0 < abs_diff < 30 :  
                w_degree = 0.3
            elif 30 <= abs_diff < 60 :    
                w_degree = 0.6
            elif 60 <= abs_diff < 90 : 
                w_degree = 0.75
            else :
                w_degree = 1.0

            # 현재 터렛 각도와 목표 각도 차이 계산
            yaw_diff = body_x - turret_x

            # 각도 차이 보정 (-180 ~ 180)
            if yaw_diff > 180:
                yaw_diff -= 360
            elif yaw_diff < -180:
                yaw_diff += 360

            # 최소 가중치 0.01 설정, 최대 1.0 제한
            def calc_yaw_weight(diff):
                w = min(max(abs(diff) / 30, 0.01), 0.3)  # 30도 내외로 가중치 조절 예시
                return w
            
            # 위 두 함수에서 최소 가중치를 낮게 할수록 조준 속도는 낮아지지만 정밀 조준 가능
            yaw_weight = calc_yaw_weight(yaw_diff)

            # 좌우 회전 명령 결정
            if yaw_diff > 0.1:  # 목표가 오른쪽
                turretQE_cmd = "E"
            elif yaw_diff < -0.1:  # 목표가 왼쪽
                turretQE_cmd = "Q"
            else:
                turretQE_cmd = ""

            forward = {'command': acceleration, 'weight': w_weight}
            turn = {'command': 'A' if diff > 0 else 'D', 'weight': w_degree}
            turret = {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0}

            cmd = {
                'moveWS': forward,
                'moveAD': turn,
                "turretQE" : turret
            }

            combined_command_cache.append(cmd)   # 두 좌표에 대한 명령값 2개가 여기 리스트에 저장됨

        # print문 살짝 수정-희연
        print(f"📍 현재 pos=({pos_x:.1f},{pos_z:.1f})") # yaw={current_yaw:.1f} 두번째 좌표로 가는 앵글 ={target_angle:.1f} 차이 ={diff:.1f}")
        print(f"🚀 cmd 3개 {combined_command_cache}")
        cmd = combined_command_cache.pop(0)
        return jsonify(cmd)

    else: # 적 전차를 찾았다면 (화면에 적 전차에 대한 바운딩 박스가 그려져 있다면)
        # 아래 273~284번 줄은 조준 가능한 각도인지 판단하고, 조준불가능한 각도라면 reset하는 코드
        save_time += 1
        if save_time > 10:
            save_time = 0
            angle_hist.append([round(turret_x, 2), round(turret_y, 2)])
            len_angle_hist += 1

        patience = 1 # 3 x n초
        if len_angle_hist > 3:
            if angle_hist[len_angle_hist][:] == angle_hist[len_angle_hist - patience][:]:
                angle_hist = []
                len_angle_hist = -1
                last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}
        
        # 적 위치
        enemy_x = enemy_pos.get("x", 0)
        enemy_y = enemy_pos.get("y", 0)
        enemy_z = enemy_pos.get("z", 0)

        player_pos = {"x": pos_x, "y": pos_y, "z": pos_z}
        enemy_pos = {"x": enemy_x, "y": enemy_y, "z": enemy_z}

        # 수평 각도 계산
        target_yaw = fire.get_yaw_angle(player_pos, enemy_pos)

        # 모델 입력을 위한 거리 계산
        distance = math.sqrt(
            (pos_x - enemy_x)**2 +
            (pos_y - enemy_y)**2 +
            (pos_z - enemy_z)**2
        )

        # 모델 입력을 위한 dy 계산
        dy = pos_y - enemy_y

        # 5번 맵 테스트용으로 내 전차랑 적 전차가 맵밖으로 떨어지면 reset
        if pos_y < 5 or enemy_y < 5:
            last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

        # y축 (pitch) 각도 예측 후 앙상블
        target_pitch_dnn = fire.find_angle_for_distance_dy_dnn(distance, dy)
        target_pitch_xgb = fire.find_angle_for_distance_dy_xgb(distance, dy)
        target_pitch = (target_pitch_dnn + target_pitch_xgb) / 2 # 사용할 y 각도

        # 현재 터렛 각도와 목표 각도 차이 계산
        yaw_diff = target_yaw - turret_x
        pitch_diff = target_pitch - turret_y

        # 각도 차이 보정 (-180 ~ 180)
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360

        # 최소 가중치 0.01 설정, 최대 1.0 제한
        def calc_yaw_weight(diff):
            w = min(max(abs(diff) / 30, 0.01), 1.0)  # 30도 내외로 가중치 조절 예시
            return w
        
        # 최소 가중치 0.1 설정, 최대 1.0 제한
        def calc_pitch_weight(diff):
            w = min(max(abs(diff) / 10, 0.1), 3.0)  # 30도 내외로 가중치 조절 예시
            return w

        # 위 두 함수에서 최소 가중치를 낮게 할수록 조준 속도는 낮아지지만 정밀 조준 가능
        yaw_weight = calc_yaw_weight(yaw_diff)
        pitch_weight = calc_pitch_weight(pitch_diff)

        # 좌우 회전 명령 결정
        if yaw_diff > 0.1:  # 목표가 오른쪽
            turretQE_cmd = "E"
        elif yaw_diff < -0.1:  # 목표가 왼쪽
            turretQE_cmd = "Q"
        else:
            turretQE_cmd = ""

        # 상하 포탑 명령 (R: up, F: down)
        if pitch_diff > 0.1:  # 포탑을 위로 올림
            turretRF_cmd = "R"
        elif pitch_diff < -0.1:
            turretRF_cmd = "F"
        else:
            turretRF_cmd = ""

        # 조준 완료 판단 (yaw, pitch 오차가 1도 이내일 때)
        aim_ready = bool(abs(yaw_diff) <= 0.1 and abs(pitch_diff) <= 0.1)
        print(f'🏹target_yaw : {target_yaw}, 🏹target_pitch : {target_pitch}')

        # 이동은 일단 멈춤, 위에서 계산한 각도 오차에 따른 가중치로 조준
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0},
            "turretRF": {"command": turretRF_cmd, "weight": pitch_weight if turretRF_cmd else 0.0},
            "fire": aim_ready
        }
    return jsonify(command)
    

# 전역 상태 저장 (시뮬레이터 reset 시킬 때 사용)
last_bullet_info = {}

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global last_bullet_info, enemy_pos_locked, locked_enemy_pos, lock_frame_counter
    last_bullet_info = request.get_json()
    print("💥 탄 정보 갱신됨:", last_bullet_info)

    if last_bullet_info.get('hit', False):
        print("🎯 적중 확인 → 락 해제")
        enemy_pos_locked = False
        locked_enemy_pos = None
        lock_frame_counter = 0

    return jsonify({"yolo_results": "ok"})

enemy_pos = {} # 적 전차의 위치
true_hit_ratio = [] # 평가를 위해서 사용했던 변수
s_time = 0 # 시뮬레이터 시간
body_x = 0

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, true_hit_ratio, s_time, lidar_data, DRIVE_MODE, enemy_pos
    global maze, body_x

    data = request.get_json()
    lidar_data = data.get('lidarPoints', [])
    s_time = data.get("s_time", 0)
    body_x = data.get('playerBodyX', 0)
    control = ""

    # 발사된 탄이 어딘가에 떨어졌을 때
    if last_bullet_info:
            DRIVE_MODE = True
            control = ""
            last_bullet_info = {}
            enemy_pos = {}

    return jsonify({
        "status": "success",
        "message": "Data received",
        "control": control,
    })

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    global DRIVE_MODE, last_bullet_info, enemy_pos
    global current_yaw, previous_position, target_reached
    current_yaw = INITIAL_YAW
    previous_position = None
    target_reached = False

    DRIVE_MODE = True
    last_bullet_info = {}
    enemy_pos = {}

    print("🛠️ /init 라우트 진입 확인!")

    config = {
        "startMode": "start",
        "blStartX": start_x, 
        "blStartY": 10, 
        "blStartZ": start_z,
        "rdStartX": 120, 
        "rdStartY": 0, 
        "rdStartZ": 50,
        "trackingMode": True,
        "detactMode": True,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Init config:", config)
    
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=False, use_reloader=False)