# main.py — Orbit Wars Hybrid Transformer + Hellburner guardrails
# Generated for checkpoint: winner_each_replay_transformer_hb_best.pt
# Put this file in the same folder as:
#   - winner_each_replay_transformer_hb_best.pt
#   - feature_stats.npz
#
# Core idea:
#   Transformer ranks candidate targets.
#   Hellburner-style rules compute angle/ships and reject unsafe actions.

import os, json, math, glob, argparse
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONSTANTS
# =============================================================================
BOARD        = 100.0
CENTER       = 50.0
TOTAL_STEPS  = 500.0
MAX_ETA      = 25.0
EPS          = 1e-6
K_SHIPS      = 40.0
SHIP_SPEED   = 6.0
LAMBDA_ECEP  = 0.3
LAMBDA_POT   = 0.15
MAX_LABEL_ANGLE_ERR = 0.05

# Định nghĩa tên các đặc trưng (features) được sử dụng trong mô hình
FEATURE_NAMES = [
    'game_progress', # tiến trình của trò chơi, có thể là phần trăm hoàn thành hoặc số cấp độ đã đạt được
    'players_alive', # số lượng người chơi còn sống
    'my_planet_share', # tỷ lệ phần trăm hành tinh mà người chơi sở hữu so với tổng số hành tinh
    'my_gdp_share', # tỷ lệ phần trăm GDP (Gross Domestic Product) của người chơi so với tổng GDP của tất cả người chơi
    'my_ship_share', # tỷ lệ phần trăm số lượng tàu của người chơi so với tổng số tàu của tất cả người chơi
    'momentum', # động lượng, có thể là sự thay đổi về số lượng tàu hoặc hành tinh trong một khoảng thời gian nhất định
    'my_fleet_ratio', # tỷ lệ giữa số lượng tàu của người chơi và số lượng tàu của đối thủ
    'leader_gap', # khoảng cách giữa người chơi và người dẫn đầu, có thể được tính bằng cách so sánh số lượng tàu, hành tinh hoặc GDP của người chơi với người dẫn đầu
    'am_i_targeted', # một đặc trưng nhị phân cho biết liệu người chơi có đang bị đối thủ nhắm đến hay không, có thể dựa trên số lượng tàu của đối thủ hướng về phía người chơi hoặc các hành động tấn công gần đây
    'fastest_grower_gap', # khoảng cách giữa người chơi và người phát triển nhanh nhất, có thể được tính bằng cách so sánh tốc độ tăng trưởng của người chơi với người phát triển nhanh nhất trong trò chơi
    'aggression_trend', # xu hướng tấn công, có thể được tính bằng cách phân tích các hành động tấn công gần đây của người chơi và đối thủ để xác định xem người chơi có đang trở nên hung hăng hơn hay không
    'enemy_rhythm', # nhịp độ của đối thủ, có thể được tính bằng cách phân tích tần suất và cường độ của các hành động tấn công của đối thủ để xác định xem họ có đang chơi một cách nhanh chóng và hung hăng hay không
    'src_ships', # số lượng tàu của người chơi tại thời điểm hiện tại
    'src_production', # sản lượng của người chơi tại thời điểm hiện tại
    'src_safety', # mức độ an toàn của người chơi, có thể được tính bằng cách phân tích số lượng tàu của đối thủ hướng về phía người chơi hoặc các hành động tấn công gần đây để xác định xem người chơi có đang ở trong tình trạng nguy hiểm hay không
    'target_diplomacy', # một đặc trưng nhị phân cho biết liệu người chơi có đang trong một liên minh ngoại giao với đối thủ hay không, có thể dựa trên các hành động ngoại giao gần đây hoặc trạng thái liên minh hiện tại của người chơi
    'target_production', # sản lượng của đối thủ tại thời điểm hiện tại
    'is_comet', # một đặc trưng nhị phân cho biết liệu người chơi có đang bị tấn công bởi một hành tinh sao chổi hay không, có thể dựa trên số lượng tàu của đối thủ hướng về phía người chơi hoặc các hành động tấn công gần đây liên quan đến hành tinh sao chổi
    'owner_power', # sức mạnh của người chơi, có thể được tính bằng cách kết hợp các đặc trưng khác như số lượng tàu, hành tinh và GDP để tạo ra một chỉ số tổng hợp về sức mạnh của người chơi
    'is_weakest_target', # một đặc trưng nhị phân cho biết liệu người chơi có đang là mục tiêu yếu nhất của đối thủ hay không, có thể dựa trên việc so sánh sức mạnh của người chơi với các đối thủ khác để xác định xem người chơi có đang bị nhắm đến như một mục tiêu dễ dàng hay không
    'projected_owner', # một đặc trưng dự đoán về chủ sở hữu của hành tinh mục tiêu trong tương lai, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem ai sẽ kiểm soát hành tinh mục tiêu trong tương lai
    'projected_resistance', # một đặc trưng dự đoán về khả năng kháng cự của hành tinh mục tiêu trong tương lai, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem hành tinh mục tiêu sẽ có khả năng kháng cự như thế nào trong tương lai
    'threat_after', # một đặc trưng dự đoán về mức độ đe dọa mà người chơi sẽ phải đối mặt sau khi thực hiện một hành động cụ thể, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi sẽ phải đối mặt với những mối đe dọa nào sau khi thực hiện một hành động cụ thể
    'support_after', # một đặc trưng dự đoán về mức độ hỗ trợ mà người chơi sẽ nhận được sau khi thực hiện một hành động cụ thể, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi sẽ nhận được những hỗ trợ nào sau khi thực hiện một hành động cụ thể
    'pre_volatility', # một đặc trưng dự đoán về mức độ biến động của trò chơi trước khi thực hiện một hành động cụ thể, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem trò chơi sẽ trở nên biến động như thế nào trước khi thực hiện một hành động cụ thể
    'threat_potential', # một đặc trưng dự đoán về tiềm năng đe dọa mà người chơi có thể đối mặt trong tương lai, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi có thể phải đối mặt với những mối đe dọa nào trong tương lai
    'support_potential', # một đặc trưng dự đoán về tiềm năng hỗ trợ mà người chơi có thể nhận được trong tương lai, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi có thể nhận được những hỗ trợ nào trong tương lai
    'eta', # thời gian ước tính để hoàn thành một hành động cụ thể, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi sẽ mất bao lâu để hoàn thành một hành động cụ thể
    'commitment_ratio', # tỷ lệ cam kết của người chơi, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi có đang cam kết với một chiến lược cụ thể hay không
    'required_ratio', # tỷ lệ yêu cầu, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi sẽ cần phải đáp ứng những yêu cầu nào để thực hiện một hành động cụ thể
    'margin', # một đặc trưng dự đoán về khoảng cách giữa người chơi và đối thủ sau khi thực hiện một hành động cụ thể, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi sẽ có khoảng cách như thế nào so với đối thủ sau khi thực hiện một hành động cụ thể
    'can_win', # một đặc trưng nhị phân cho biết liệu người chơi có thể giành chiến thắng ngay sau khi thực hiện một hành động cụ thể hay không, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của người chơi và đối thủ để dự đoán xem người chơi có thể giành chiến thắng ngay sau khi thực hiện một hành động cụ thể hay không
    'garrison_strength', # sức mạnh của lực lượng đồn trú, có thể được tính bằng cách phân tích số lượng tàu và hành tinh mà người chơi sở hữu để tạo ra một chỉ số tổng hợp về sức mạnh của lực lượng đồn trú của người chơi
    'defense_sustainability', # khả năng duy trì phòng thủ, có thể được tính bằng cách phân tích số lượng tàu và hành tinh mà người chơi sở hữu để tạo ra một chỉ số tổng hợp về khả năng duy trì phòng thủ của người chơi
    'target_roi', # lợi tức đầu tư của mục tiêu, có thể được tính bằng cách phân tích số lượng tàu và hành tinh mà người chơi sở hữu để tạo ra một chỉ số tổng hợp về lợi tức đầu tư của mục tiêu
    'front_gradient', # một đặc trưng đo lường độ dốc của chiến tuyến, có thể được tính bằng cách phân tích vị trí và hướng di chuyển của các tàu
    'avg_enemy_distance', # khoảng cách trung bình đến các đối thủ, có thể được tính bằng cách phân tích vị trí của các tàu đối thủ
    'angular_spread', # sự phân bố góc của các tàu đối thủ, có thể được tính bằng cách phân tích góc giữa các tàu đối thủ
    'orbital_trend', # xu hướng quỹ đạo của các tàu, có thể được tính bằng cách phân tích chuyển động của các tàu theo thời gian
    'convergence_threat', # mức độ đe dọa từ sự hội tụ của các tàu đối thủ, có thể được tính bằng cách phân tích vị trí và hướng di chuyển của các tàu đối thủ
    'approach_rate', # tốc độ tiếp cận của các tàu đối thủ, có thể được tính bằng cách phân tích chuyển động của các tàu theo thời gian
    'enemy_commitment', # mức độ cam kết của các tàu đối thủ, có thể được tính bằng cách phân tích các hành động gần đây và xu hướng của đối thủ
    'weakest_colony', # hành tinh yếu nhất của người chơi, có thể được xác định bằng cách phân tích vị trí và sức mạnh của các hành tinh
    'local_superiority', # sự ưu thế địa phương, có thể được tính bằng cách phân tích vị trí và sức mạnh của các tàu trong khu vực cụ thể
    'economic_impact', # tác động kinh tế, có thể được tính bằng cách phân tích số lượng tàu và hành tinh mà người chơi sở hữu để tạo ra một chỉ số tổng hợp về tác động kinh tế của người chơi
    'enemy_exposed' # một đặc trưng nhị phân cho biết liệu đối thủ có đang bị phơi bày hay không, có thể được tính bằng cách phân tích vị trí và hướng di chuyển của các tàu đối thủ để xác định xem họ có đang ở trong tình trạng dễ bị tấn công hay không
]


# =============================================================================
# MATH UTILS
# =============================================================================
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))

def ship_sigmoid(x):
    x = max(0.0, float(x))
    return x / (x + K_SHIPS)

def dist_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def angle_diff(a, b):
    """Khoảng cách góc nhỏ nhất (absolute), kết quả trong [0, pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)
# =============================================================================
# BƯỚC 1 – ORBITAL MECHANICS
# =============================================================================
def is_orbiting_planet(p):
    """
    Planet quay quanh tâm nếu khoảng cách tâm + radius < 50.
    Planet format: [id, owner, x, y, radius, ships, production]
    """
    _, _, x, y, radius, *_ = p
    return dist_xy(x, y, CENTER, CENTER) + radius < 50.0


def predict_rotated_xy(p, delta_step, angular_velocity):
    """
    Tính vị trí (x, y) của planet sau delta_step turns.
    Planet tĩnh (không orbit) trả về vị trí hiện tại.
    """
    _, _, x, y, radius, *_ = p
    if not is_orbiting_planet(p):
        return x, y
    dx = x - CENTER
    dy = y - CENTER
    r  = math.hypot(dx, dy)
    theta = math.atan2(dy, dx) + angular_velocity * delta_step
    return CENTER + r * math.cos(theta), CENTER + r * math.sin(theta)

# =============================================================================
# BƯỚC 2 – TÌM TARGET TỪ ANGLE
# =============================================================================
def estimate_eta(src, tgt, ships):
    d = dist_xy(src[2], src[3], tgt[2], tgt[3])
    return max(1, int(math.ceil(d / SHIP_SPEED))), d


def infer_target_by_angle_lead(src, planets, action_angle, ships, angular_velocity,
                                max_error=0.25):
    """
    Suy ra target_id từ action angle.

    Cải tiến so với pipeline cũ:
    - Test 3 vị trí ứng viên cho mỗi planet (vị trí hiện tại, sau eta turns,
      sau eta//2 turns) để bắt được các trường hợp bắn lead vào planet đang quay.
    - Threshold chặt hơn (0.25 rad vs 1.5 rad cũ), giảm false positive.

    Trả về (target_id, best_err) hoặc (None, err) nếu không tìm thấy.
    """
    best_id  = None
    best_err = 1e9
    sx, sy   = src[2], src[3]

    for tgt in planets:
        if int(tgt[0]) == int(src[0]):
            continue

        eta, _ = estimate_eta(src, tgt, ships)

        # 3 vị trí candidate: hiện tại, sau eta turns, sau eta//2 turns
        candidate_positions = [
            (tgt[2], tgt[3]),
            predict_rotated_xy(tgt, eta,              angular_velocity),
            predict_rotated_xy(tgt, max(1, eta // 2), angular_velocity),
        ]

        for tx, ty in candidate_positions:
            expected = math.atan2(ty - sy, tx - sx)
            err      = angle_diff(action_angle, expected)
            if err < best_err:
                best_err = err
                best_id  = int(tgt[0])

    if best_err > max_error:
        return None, best_err
    return best_id, best_err


# =============================================================================
# BƯỚC 3 – CANDIDATE GENERATION
# =============================================================================
def build_candidates(src, planets, K=24):
    """
    Candidate generator stratified (từ GPT, tốt hơn nearest-only).

    Ưu tiên:
    - Enemy:   K//3 gần nhất (muốn tấn công)
    - Neutral: K//2 có tỷ lệ ships/prod thấp nhất (dễ chiếm, sinh lời cao)
    - Friendly: K//4 gần nhất (có thể reinforce)
    - Fill bằng nearest chưa được chọn

    Quan trọng: đảm bảo true_target luôn nằm trong tập candidates.
    """
    src_id = int(src[0])
    player = int(src[1])
    sx, sy = src[2], src[3]

    rows = []
    for p in planets:
        if int(p[0]) == src_id:
            continue
        d     = dist_xy(sx, sy, p[2], p[3])
        owner = int(p[1])
        rows.append((p, d, owner))

    enemy   = [(p, d) for p, d, o in rows if o not in (-1, player)]
    neutral = [(p, d) for p, d, o in rows if o == -1]
    friendly= [(p, d) for p, d, o in rows if o == player]

    enemy.sort(key=lambda x: x[1])
    # Neutral ưu tiên theo ships/prod (thấp = dễ chiếm, sinh lời cao)
    neutral.sort(key=lambda x: (x[0][5] / (x[0][6] + EPS), x[1]))
    friendly.sort(key=lambda x: x[1])

    selected    = []
    selected_ids= set()

    for bucket in [enemy[:K//3], neutral[:K//2], friendly[:K//4]]:
        for p, d in bucket:
            pid = int(p[0])
            if pid not in selected_ids:
                selected.append((p, d))
                selected_ids.add(pid)

    # Fill bằng nearest chưa có
    all_nearest = sorted([(p, d) for p, d, _ in rows], key=lambda x: x[1])
    for p, d in all_nearest:
        if len(selected) >= K:
            break
        pid = int(p[0])
        if pid not in selected_ids:
            selected.append((p, d))
            selected_ids.add(pid)

    return selected[:K]

# =============================================================================
# BƯỚC 3.5 – HELLBURNER-STYLE HEURISTIC FEATURES / GUARDRAILS
# =============================================================================
# Các hàm này được rút gọn từ ý tưởng trong 82_hellburner_upgraded.py:
# - fleet_speed theo số quân
# - intercept planet quay
# - first_planet_hit để tránh bắn nhầm planet
# - point_to_segment_distance để tránh đường bay xuyên Sun
# - ROI / source safety / required ships
#
# Lưu ý:
# - Đây không phải nhúng nguyên agent Hellburner.
# - Transformer vẫn học target ranking.
# - Hellburner heuristics chỉ bổ sung feature + mask candidate nguy hiểm.

SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
SHIP_SPEED_MAX = 6.0
GAME_LENGTH = 500.0

HELLBURNER_FEATURE_NAMES = [
    "hb_intercept_eta_norm",        # ETA khi bắn lead vào target quay
    "hb_intercept_angle_sin",       # sin(angle intercept)
    "hb_intercept_angle_cos",       # cos(angle intercept)
    "hb_crosses_sun",               # đường từ src đến điểm intercept cắt Sun
    "hb_first_hit_is_target",       # ray bắn đầu tiên có chạm đúng target không
    "hb_required_ships_norm",       # số quân heuristic cần gửi
    "hb_required_ratio",            # required / src_ships
    "hb_surplus_after_send_norm",   # src còn dư bao nhiêu sau khi gửi required
    "hb_source_safe_after_send",    # source vẫn giữ được garrison tối thiểu
    "hb_roi_score",                 # production * payoff_turns / cost
    "hb_payoff_turns_norm",         # số turn còn sinh lời sau khi đến nơi
    "hb_defense_urgency",           # target của mình đang bị đe dọa
    "hb_target_enemy_pressure",     # áp lực enemy quanh target
    "hb_target_ally_support",       # support của mình quanh target
    "hb_local_pressure_ratio",      # support / pressure
    "hb_target_is_orbiting",        # target có quay quanh Sun không
    "hb_source_is_frontline",       # source gần enemy / frontier
    "hb_counter_risk",              # enemy gần target có thể counter không
]


def hb_fleet_speed(ships):
    """Công thức tốc độ fleet giống Hellburner."""
    ships = max(1.0, float(ships))
    return min(
        SHIP_SPEED_MAX,
        1.0 + (SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5,
    )


def hb_point_to_segment_distance(px, py, ax, ay, bx, by):
    """Khoảng cách từ điểm P đến đoạn AB."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def hb_segment_crosses_sun(ax, ay, bx, by, margin=0.0):
    return hb_point_to_segment_distance(CENTER, CENTER, ax, ay, bx, by) <= (SUN_RADIUS + margin)


def hb_is_orbiting(p):
    return dist_xy(float(p[2]), float(p[3]), CENTER, CENTER) + float(p[4]) < ROTATION_RADIUS_LIMIT


def hb_intercept_planet(src, tgt, ships, angular_velocity, scene_step, tol=1e-6, max_iters=30):
    """
    Dạng rút gọn từ Hellburner.intercept_planet().
    Trả về: angle, intercept_x, intercept_y, travel_turns.
    """
    sx, sy = float(src[2]), float(src[3])
    speed = hb_fleet_speed(ships)

    if not hb_is_orbiting(tgt):
        tx, ty = float(tgt[2]), float(tgt[3])
        travel = dist_xy(sx, sy, tx, ty) / speed
    else:
        # Không có object orbital_info như Hellburner, nên lấy góc hiện tại làm mốc.
        r = dist_xy(float(tgt[2]), float(tgt[3]), CENTER, CENTER)
        cur_angle = math.atan2(float(tgt[3]) - CENTER, float(tgt[2]) - CENTER)

        # Ước lượng lặp ETA -> vị trí target tương lai.
        travel = dist_xy(sx, sy, float(tgt[2]), float(tgt[3])) / speed
        for _ in range(max_iters):
            a = cur_angle + angular_velocity * max(0.0, travel - 0.5)
            tx = CENTER + r * math.cos(a)
            ty = CENTER + r * math.sin(a)
            new_travel = dist_xy(sx, sy, tx, ty) / speed
            # damping giống tinh thần Hellburner để tránh dao động.
            new_travel = 0.5 * (travel + new_travel)
            if abs(new_travel - travel) < tol:
                travel = new_travel
                break
            travel = new_travel
        else:
            return 0.0, float(tgt[2]), float(tgt[3]), math.inf

        a = cur_angle + angular_velocity * max(0.0, travel - 0.5)
        tx = CENTER + r * math.cos(a)
        ty = CENTER + r * math.sin(a)

    angle = math.atan2(ty - sy, tx - sx)
    return angle, tx, ty, travel


def hb_first_planet_hit(src, angle, ships, planets, angular_velocity, scene_step):
    """
    Dạng rút gọn từ Hellburner.first_planet_hit().
    Tìm planet đầu tiên mà ray/fleet sẽ chạm theo angle.
    Nếu đường đến hit đầu tiên cắt Sun -> trả về None.
    """
    sx, sy = float(src[2]), float(src[3])
    source_id = int(src[0])

    best = None
    best_t = float("inf")

    for p in planets:
        if int(p[0]) == source_id:
            continue

        needed_angle, px, py, travel = hb_intercept_planet(
            src, p, ships, angular_velocity, scene_step
        )

        if not math.isfinite(travel):
            continue

        d = dist_xy(sx, sy, px, py)
        radius = float(p[4])
        half_cone = math.pi if d < radius else math.asin(min(1.0, radius / max(d, EPS)))
        delta = angle_diff(angle, needed_angle)

        if delta <= half_cone and travel < best_t:
            best_t = travel
            best = p

    if best is None:
        return None

    ex = sx + best_t * hb_fleet_speed(ships) * math.cos(angle)
    ey = sy + best_t * hb_fleet_speed(ships) * math.sin(angle)

    if hb_segment_crosses_sun(sx, sy, ex, ey, margin=0.75):
        return None

    return best


def hb_required_ships(src, tgt, travel):
    """
    Heuristic ship sizing:
    - friendly: reinforce nhỏ, không phải attack;
    - neutral: tgt.ships + 1 + buffer;
    - enemy: tgt.ships + production trong ETA + buffer.
    """
    player = int(src[1])
    tgt_owner = int(tgt[1])
    tgt_ships = float(tgt[5])
    tgt_prod = float(tgt[6])

    eta_turns = max(1, int(math.ceil(travel))) if math.isfinite(travel) else 99

    if tgt_owner == player:
        return max(0.0, min(float(src[5]) - 8.0, 12.0))

    if tgt_owner == -1:
        return tgt_ships + 1.0 + max(1.0, 0.15 * tgt_ships)

    # Enemy planet sẽ sinh thêm quân trước khi fleet tới.
    return tgt_ships + tgt_prod * eta_turns + 2.0 + max(1.0, 0.10 * tgt_ships)


def hb_local_pressure(tgt, planets, player):
    """
    Ước lượng pressure/support quanh target dựa trên ships và khoảng cách.
    """
    tx, ty = float(tgt[2]), float(tgt[3])
    enemy_pressure = 0.0
    ally_support = 0.0

    for p in planets:
        if int(p[0]) == int(tgt[0]):
            continue
        d = max(1.0, dist_xy(float(p[2]), float(p[3]), tx, ty))
        influence = ship_sigmoid(float(p[5])) * math.exp(-d / 35.0)

        if int(p[1]) == player:
            ally_support += influence
        elif int(p[1]) != -1:
            enemy_pressure += influence

    return enemy_pressure, ally_support


def hb_source_is_frontline(src, planets, player):
    sx, sy = float(src[2]), float(src[3])
    nearest_enemy = min(
        (
            dist_xy(sx, sy, float(p[2]), float(p[3]))
            for p in planets
            if int(p[1]) not in (-1, player)
        ),
        default=BOARD,
    )
    return 1.0 if nearest_enemy <= 38.0 else 0.0


def hb_candidate_metrics(src, tgt, g):
    """
    Tính toàn bộ Hellburner-style metrics cho một candidate.
    """
    planets = g.get("planets", [])
    player = int(src[1])
    scene_step = int(g.get("step_idx", 0))
    angular_velocity = float(g.get("angular_velocity", 0.0))

    src_ships = max(1.0, float(src[5]))

    # Tính theo required sơ bộ rồi intercept lại bằng required.
    coarse_angle, coarse_ix, coarse_iy, coarse_travel = hb_intercept_planet(
        src, tgt, max(1.0, src_ships), angular_velocity, scene_step
    )
    if not math.isfinite(coarse_travel):
        return {
            "valid": False,
            "features": [0.0] * len(HELLBURNER_FEATURE_NAMES),
        }

    required = hb_required_ships(src, tgt, coarse_travel)
    required = max(1.0, min(required, src_ships))

    angle, ix, iy, travel = hb_intercept_planet(
        src, tgt, required, angular_velocity, scene_step
    )

    if not math.isfinite(travel):
        return {
            "valid": False,
            "features": [0.0] * len(HELLBURNER_FEATURE_NAMES),
        }

    crosses_sun = 1.0 if hb_segment_crosses_sun(float(src[2]), float(src[3]), ix, iy, margin=0.75) else 0.0

    first_hit = hb_first_planet_hit(
        src, angle, required, planets, angular_velocity, scene_step
    )
    first_hit_is_target = 1.0 if (first_hit is not None and int(first_hit[0]) == int(tgt[0])) else 0.0

    # Source safety giống tinh thần Hellburner: không rút sạch source, nhất là frontline.
    garrison_keep = 8.0
    if scene_step >= 440:
        garrison_keep = 1.0

    source_frontline = hb_source_is_frontline(src, planets, player)
    if source_frontline > 0.5:
        garrison_keep += 4.0

    surplus_after = src_ships - required
    source_safe = 1.0 if surplus_after >= garrison_keep else 0.0

    turns_remaining = max(0.0, GAME_LENGTH - scene_step)
    payoff_turns = max(1.0, turns_remaining - math.ceil(travel))

    if int(tgt[1]) == player:
        roi = 0.0
    else:
        roi = float(tgt[6]) * payoff_turns / (required + 1.0)

    enemy_pressure, ally_support = hb_local_pressure(tgt, planets, player)
    pressure_ratio = ally_support / (enemy_pressure + 0.10)

    # Defense urgency: target của mình nhưng pressure cao và ships thấp.
    if int(tgt[1]) == player:
        defense_urgency = clamp((enemy_pressure - ship_sigmoid(float(tgt[5]))) * 2.0, 0.0, 1.0)
    else:
        defense_urgency = 0.0

    # Counter risk: target có enemy mạnh gần đó, dễ bị retake.
    counter_risk = clamp(enemy_pressure / (ally_support + 0.25), 0.0, 2.0) / 2.0

    features = [
        clamp(travel / MAX_ETA),                                      # hb_intercept_eta_norm
        math.sin(angle),                                              # hb_intercept_angle_sin
        math.cos(angle),                                              # hb_intercept_angle_cos
        crosses_sun,                                                  # hb_crosses_sun
        first_hit_is_target,                                          # hb_first_hit_is_target
        min(required, 1000.0) / 1000.0,                               # hb_required_ships_norm
        clamp(required / (src_ships + EPS), 0.0, 2.0) / 2.0,          # hb_required_ratio
        clamp(surplus_after / 1000.0, -1.0, 1.0),                    # hb_surplus_after_send_norm
        source_safe,                                                  # hb_source_safe_after_send
        clamp(roi / 50.0),                                            # hb_roi_score
        clamp(payoff_turns / GAME_LENGTH),                            # hb_payoff_turns_norm
        defense_urgency,                                              # hb_defense_urgency
        clamp(enemy_pressure),                                        # hb_target_enemy_pressure
        clamp(ally_support),                                          # hb_target_ally_support
        clamp(pressure_ratio / 3.0),                                  # hb_local_pressure_ratio
        1.0 if hb_is_orbiting(tgt) else 0.0,                          # hb_target_is_orbiting
        source_frontline,                                             # hb_source_is_frontline
        counter_risk,                                                 # hb_counter_risk
    ]

    # Candidate được xem là hợp lệ nếu:
    # - đường bay không xuyên Sun;
    # - first planet hit là đúng target;
    # - source còn đủ quân sau khi gửi required.
    valid = (crosses_sun < 0.5) and (first_hit_is_target > 0.5) and (source_safe > 0.5)

    return {
        "valid": bool(valid),
        "angle": angle,
        "travel": travel,
        "required": required,
        "features": [0.0 if not math.isfinite(float(x)) else float(x) for x in features],
    }


def hb_candidate_valid(src, tgt, g):
    return hb_candidate_metrics(src, tgt, g)["valid"]

# =============================================================================
# BƯỚC 4 – REPLAY HISTORY (gọi một lần per step)
# =============================================================================
class ReplayHistory:
    """
    Theo dõi temporal features xuyên suốt một game.
    Quan trọng: phải gọi update() một lần cho mỗi step, trước vòng lặp candidate.
    """
    def __init__(self):
        self.ship_share     = {i: deque(maxlen=5) for i in range(4)}
        self.targeted       = deque(maxlen=5)
        self.incoming       = deque(maxlen=10)
        self.prev_enemy_dists = None

    def update(self, player, shares, targeted_ratio, hostile_incoming, enemy_dists):
        old_my   = self.ship_share.get(player, deque())
        momentum = shares.get(player, 0.0) - old_my[0] if len(old_my) >= 5 else 0.0

        enemy_momentums = []
        for o in range(4):
            if o == player:
                continue
            old = self.ship_share.get(o, deque())
            if len(old) >= 5:
                enemy_momentums.append(shares.get(o, 0.0) - old[0])

        fastest_grower_gap = max(enemy_momentums) - momentum if enemy_momentums else 0.0
        aggression_trend   = (targeted_ratio - self.targeted[0]
                              if len(self.targeted) >= 5 else 0.0)

        enemy_rhythm = 0.0
        if len(self.incoming) >= 10:
            vals = np.array(self.incoming, dtype=np.float32)
            enemy_rhythm = clamp(float(vals.var() / (vals.mean() ** 2 + EPS)))

        approach_rate = 0.0
        if self.prev_enemy_dists:
            vals = [max(0.0, self.prev_enemy_dists[o] - cur)
                    for o, cur in enemy_dists.items()
                    if o in self.prev_enemy_dists]
            if vals:
                approach_rate = clamp(sum(vals) / (len(vals) * 12.0 + EPS))

        for o in range(4):
            self.ship_share[o].append(shares.get(o, 0.0))
        self.targeted.append(targeted_ratio)
        self.incoming.append(hostile_incoming)
        self.prev_enemy_dists = dict(enemy_dists)

        return {
            "momentum":            clamp(momentum,           -1.0, 1.0),
            "fastest_grower_gap":  clamp(fastest_grower_gap, -1.0, 1.0),
            "aggression_trend":    clamp(aggression_trend,   -1.0, 1.0),
            "enemy_rhythm":        enemy_rhythm,
            "approach_rate":       approach_rate,
        }
    
# =============================================================================
# BƯỚC 5 – GLOBAL STATE PER STEP
# =============================================================================
def center_of_mass(planets, fleets, owner):
    """
    - Dùng cả planets lẫn fleets.
    Planet format: [id, owner, x, y, radius, ships, production]
    Fleet  format: [id, owner, x, y, angle, from_planet_id, ships]
    """
    sx = sy = sw = 0.0
    for p in planets:
        if int(p[1]) == owner:
            w   = max(1.0, float(p[5]))
            sx += p[2] * w;  sy += p[3] * w;  sw += w
    for f in fleets:
        if int(f[1]) == owner:
            w   = max(1.0, float(f[6]))
            sx += f[2] * w;  sy += f[3] * w;  sw += w
    return (sx / sw, sy / sw) if sw > 0 else (CENTER, CENTER)


def angular_spread(my_com, enemy_coms):
    if len(enemy_coms) <= 1:
        return 0.0
    angles = sorted(math.atan2(y - my_com[1], x - my_com[0]) for x, y in enemy_coms)
    gaps   = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
    gaps.append(angles[0] + 2*math.pi - angles[-1])
    return clamp(1.0 - max(gaps) / (2*math.pi))


def compute_step_globals(step_idx, total_steps, player, planets, fleets, history):
    """
    Tính tất cả global state cho một step.
    Kết quả được pass vào extract_candidate_features cho mỗi candidate.
    """
    totals = defaultdict(lambda: {
        "planets": 0, "planet_ships": 0.0, "fleet_ships": 0.0,
        "prod": 0.0, "transit": 0.0, "total_ships": 0.0,
    })
    alive = set()

    for p in planets:
        owner = int(p[1])
        if owner == -1:
            continue
        totals[owner]["planets"]      += 1
        totals[owner]["planet_ships"] += float(p[5])
        totals[owner]["prod"]         += float(p[6])
        alive.add(owner)

    for f in fleets:
        owner = int(f[1])
        totals[owner]["fleet_ships"] += float(f[6])
        totals[owner]["transit"]     += float(f[6])
        alive.add(owner)

    for o in totals:
        totals[o]["total_ships"] = totals[o]["planet_ships"] + totals[o]["fleet_ships"]

    total_ships  = sum(v["total_ships"] for v in totals.values()) + EPS
    total_prod   = sum(v["prod"]        for v in totals.values()) + EPS
    my_total     = totals[player]["total_ships"]
    my_prod      = totals[player]["prod"]

    enemy_owners = [o for o in totals if o != player and totals[o]["total_ships"] > 0]
    max_enemy_ships = max((totals[o]["total_ships"] for o in enemy_owners), default=0.0)
    hostile_transit = sum(float(f[6]) for f in fleets if int(f[1]) != player)

    shares      = {o: totals[o]["total_ships"] / total_ships for o in range(4)}
    my_com      = center_of_mass(planets, fleets, player)
    enemy_coms  = {o: center_of_mass(planets, fleets, o) for o in enemy_owners}
    enemy_dists = {o: dist_xy(my_com[0], my_com[1], c[0], c[1])
                   for o, c in enemy_coms.items()}

    # Targeted ratio
    my_planet_ids  = {int(p[0]) for p in planets if int(p[1]) == player}
    hostile_to_me  = sum(float(f[6]) for f in fleets
                         if int(f[1]) != player
                         and int(f[5]) in my_planet_ids)
    targeted_ratio = clamp(hostile_to_me / (hostile_transit + EPS))

    hist = history.update(player, shares, targeted_ratio, hostile_transit, enemy_dists)

    weakest_enemy = (min(enemy_owners, key=lambda o: totals[o]["total_ships"])
                     if enemy_owners else None)

    avg_enemy_dist = (np.mean(list(enemy_dists.values())) / (math.sqrt(2) * BOARD)
                      if enemy_dists else 0.0)
    ang_spread = angular_spread(my_com, list(enemy_coms.values()))

    weakest_colony = 1.0
    my_planets = [p for p in planets if int(p[1]) == player]
    if my_planets:
        for p in my_planets:
            hostile_p = sum(float(f[6]) for f in fleets
                            if int(f[1]) not in (-1, player)
                            and int(f[5]) == int(p[0]))
            weakest_colony = min(weakest_colony,
                                 clamp((float(p[5]) - hostile_p) / (float(p[5]) + EPS)))
    else:
        weakest_colony = 0.0

    return {
        "totals":          totals,
        "total_ships":     total_ships,
        "total_prod":      total_prod,
        "my_total":        my_total,
        "my_prod":         my_prod,
        "max_enemy_ships": max_enemy_ships,
        "enemy_owners":    enemy_owners,
        "weakest_enemy":   weakest_enemy,
        "targeted_ratio":  targeted_ratio,
        "shares":          shares,
        "my_planets":      my_planets,
        "enemy_planets":   [p for p in planets if int(p[1]) not in (-1, player)],
        "num_planets":     len(planets),
        "avg_enemy_dist":  clamp(avg_enemy_dist),
        "angular_spread":  ang_spread,
        "weakest_colony":  weakest_colony,
        "hist":            hist,
        "game_progress":   clamp(step_idx / total_steps),
        "players_alive":   clamp(len(alive) / 4.0),
        "planets":         planets,
        "fleets":          fleets,
        "step_idx":        step_idx,
        "total_steps":     total_steps,
    }
# =============================================================================
# BƯỚC 6 – FEATURE EXTRACTION (46 cũ + 14 geometry + Hellburner features)
# =============================================================================

EXTRA_FEATURE_NAMES = [
    "src_x_norm",
    "src_y_norm",
    "tgt_x_norm",
    "tgt_y_norm",
    "dx_norm",
    "dy_norm",
    "dist_norm",
    "angle_sin",
    "angle_cos",
    "tgt_ships_norm",
    "src_ships_norm",
    "is_neutral",
    "is_self",
    "is_enemy",
]

# Chạy cell này sau khi đã khai báo FEATURE_NAMES 46 chiều cũ
FEATURE_NAMES = FEATURE_NAMES[:46] + EXTRA_FEATURE_NAMES + HELLBURNER_FEATURE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)


def extract_candidate_features(src, tgt, raw_distance, g):
    """
    Trích feature cho cặp (src, tgt).
    - 46 feature đầu: chiến thuật/tổng quan cũ.
    - 14 feature tiếp theo: hình học + owner + ships.
    - phần cuối: Hellburner-style heuristics.
    """
    player    = int(src[1])
    src_id    = int(src[0])
    src_ships = max(1.0, float(src[5]))
    src_prod  = float(src[6])

    tgt_id    = int(tgt[0])
    tgt_owner = int(tgt[1])
    tgt_ships = max(0.0, float(tgt[5]))
    tgt_prod  = float(tgt[6])

    src_x = float(src[2])
    src_y = float(src[3])
    tgt_x = float(tgt[2])
    tgt_y = float(tgt[3])

    dx_raw = tgt_x - src_x
    dy_raw = tgt_y - src_y
    geom_dist = math.sqrt(dx_raw * dx_raw + dy_raw * dy_raw)
    angle_to_tgt = math.atan2(dy_raw, dx_raw)

    eta = max(1, int(math.ceil(raw_distance / SHIP_SPEED)))

    # Hellburner-style metrics cho candidate này.
    # Dùng lại trong phần feature cuối và cũng có thể dùng để mask candidate khi parse replay.
    hb = hb_candidate_metrics(src, tgt, g)
    hb_features = hb["features"]


    need   = 0.0 if tgt_owner == player else tgt_ships + 1.0
    margin = src_ships - need

    target_total = g["totals"][tgt_owner]["total_ships"] if tgt_owner != -1 else 0.0
    owner_power  = clamp(target_total / g["total_ships"])

    src_front = min(
        (dist_xy(src_x, src_y, e[2], e[3]) for e in g["enemy_planets"]),
        default=BOARD,
    )
    tgt_front = min(
        (dist_xy(tgt_x, tgt_y, e[2], e[3]) for e in g["enemy_planets"]),
        default=BOARD,
    )

    threat_potential = clamp(min(
        sum(
            ship_sigmoid(e[5]) * math.exp(
                -LAMBDA_POT * max(
                    1,
                    int(math.ceil(dist_xy(e[2], e[3], tgt_x, tgt_y) / SHIP_SPEED))
                )
            )
            for e in g["enemy_planets"]
        ),
        2.0
    ) / 2.0)

    support_potential = clamp(min(
        sum(
            ship_sigmoid(p[5]) * math.exp(
                -LAMBDA_POT * max(
                    1,
                    int(math.ceil(dist_xy(p[2], p[3], tgt_x, tgt_y) / SHIP_SPEED))
                )
            )
            for p in g["my_planets"]
            if int(p[0]) != src_id
        ),
        2.0
    ) / 2.0)

    if tgt_owner == player:
        diplomacy = 1.0
        survivors = float(src_ships)
    elif tgt_owner == -1:
        diplomacy = 0.0
        survivors = max(0.0, float(src_ships) - tgt_ships)
    else:
        diplomacy = -1.0
        survivors = max(0.0, float(src_ships) - tgt_ships)

    garrison_strength   = ship_sigmoid(survivors)
    defense_sust        = clamp(garrison_strength / (threat_potential + EPS))
    local_superiority   = clamp(min(support_potential / (threat_potential + 0.1), 3.0) / 3.0)

    projected_owner      = 1.0 if margin > 0 else (0.0 if tgt_owner == -1 else -1.0)
    projected_resistance = ship_sigmoid(max(tgt_ships - src_ships, 0.0))

    enemy_commitment = 0.0
    if tgt_owner not in (-1, player):
        t = g["totals"][tgt_owner]
        enemy_commitment = clamp(t["transit"] / (t["total_ships"] + EPS))

    enemy_exposed = clamp(enemy_commitment * (1.0 - owner_power))
    src_safety    = 1.0

    hist = g["hist"]

    row = [
        g["game_progress"],                                            # 0
        g["players_alive"],                                            # 1
        clamp(len(g["my_planets"]) / max(1, g["num_planets"])),        # 2
        clamp(g["my_prod"] / g["total_prod"]),                        # 3
        clamp(g["my_total"] / g["total_ships"]),                      # 4
        hist["momentum"],                                              # 5
        clamp(g["totals"][player]["transit"] / (g["my_total"] + EPS)), # 6
        clamp((g["max_enemy_ships"] - g["my_total"])
              / (g["my_total"] + K_SHIPS), -1.0, 2.0),                # 7
        g["targeted_ratio"],                                           # 8
        hist["fastest_grower_gap"],                                    # 9
        hist["aggression_trend"],                                      # 10
        hist["enemy_rhythm"],                                          # 11
        ship_sigmoid(src_ships),                                       # 12
        clamp(src_prod / 5.0),                                         # 13
        src_safety,                                                    # 14
        diplomacy,                                                     # 15
        clamp(tgt_prod / 5.0),                                         # 16
        1.0 if is_orbiting_planet(tgt) else 0.0,                       # 17
        owner_power,                                                   # 18
        1.0 if (tgt_owner == g["weakest_enemy"]
                and tgt_owner not in (-1, player)) else 0.0,           # 19
        projected_owner,                                               # 20
        projected_resistance,                                          # 21
        0.0,                                                           # 22
        0.0,                                                           # 23
        0.0,                                                           # 24
        threat_potential,                                              # 25
        support_potential,                                             # 26
        clamp(eta / MAX_ETA),                                          # 27
        clamp(src_ships / (g["my_total"] + EPS)),                     # 28
        clamp(need / (src_ships + EPS), 0.0, 2.0),                     # 29
        clamp(margin / (src_ships + K_SHIPS), -1.0, 1.0),              # 30
        1.0 if src_ships > need else 0.0,                              # 31
        garrison_strength,                                             # 32
        defense_sust,                                                  # 33
        clamp(tgt_prod / (need + 1.0)),                                # 34
        clamp((src_front - tgt_front) / 60.0, -1.0, 1.0),              # 35
        g["avg_enemy_dist"],                                           # 36
        g["angular_spread"],                                           # 37
        0.0,                                                           # 38
        0.0,                                                           # 39
        hist["approach_rate"],                                         # 40
        enemy_commitment,                                              # 41
        g["weakest_colony"],                                           # 42
        local_superiority,                                             # 43
        clamp(tgt_prod / (g["my_prod"] + 1.0)),                       # 44
        enemy_exposed,                                                 # 45

        # ===== 14 feature mới =====
        src_x / BOARD,                                                 # 46
        src_y / BOARD,                                                 # 47
        tgt_x / BOARD,                                                 # 48
        tgt_y / BOARD,                                                 # 49
        dx_raw / BOARD,                                                # 50
        dy_raw / BOARD,                                                # 51
        geom_dist / BOARD,                                             # 52
        math.sin(angle_to_tgt),                                        # 53
        math.cos(angle_to_tgt),                                        # 54
        min(tgt_ships, 1000.0) / 1000.0,                               # 55
        min(src_ships, 1000.0) / 1000.0,                               # 56
        1.0 if tgt_owner == -1 else 0.0,                               # 57
        1.0 if tgt_owner == player else 0.0,                           # 58
        1.0 if tgt_owner not in (-1, player) else 0.0,                 # 59

        # ===== Hellburner-style heuristic features =====
        *hb_features,
    ]

    return [0.0 if not math.isfinite(float(x)) else float(x) for x in row]
class CandidateTransformer(nn.Module):
    def __init__(self, feat_dim=None, d_model=128, nhead=4,
                 n_layers=6, dim_ff=256, dropout=0.1):
        super().__init__()
        feat_dim = FEATURE_DIM if feat_dim is None else int(feat_dim)
        self.feat_dim = feat_dim
        self.input_proj = nn.Linear(feat_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, mask):
        mask = mask.bool()
        x = self.input_proj(x)
        pad_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        scores = self.score_head(x).squeeze(-1)
        return scores.masked_fill(pad_mask, float("-inf"))

# =============================================================================
# SUBMISSION LOAD + INFERENCE
# =============================================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = "/kaggle_simulations/agent"

MODEL_PATH = os.path.join(BASE_DIR, "winner_each_replay_transformer_hb_best.pt")
STATS_PATH = os.path.join(BASE_DIR, "feature_stats.npz")
DEVICE = torch.device("cpu")

MODEL_OK = False
MODEL_ERROR = None

try:
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if os.path.exists(STATS_PATH):
        _stats = np.load(STATS_PATH, allow_pickle=True)
        FEATURE_MEAN = _stats["mean"].astype(np.float32)
        FEATURE_STD = _stats["std"].astype(np.float32)
        K = int(_stats["K"])
        FEAT_DIM = int(_stats["feat_dim"])
    else:
        FEATURE_MEAN = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        FEATURE_STD = np.asarray(ckpt["feature_std"], dtype=np.float32)
        K = int(ckpt["K"])
        FEAT_DIM = int(ckpt["feat_dim"])

    # Nếu notebook train đúng bản Hellburner features thì FEAT_DIM thường là 78.
    # Vẫn đọc động từ stats/checkpoint để tránh hard-code.
    model = CandidateTransformer(feat_dim=FEAT_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERROR = repr(e)
    FEATURE_MEAN = np.zeros(FEATURE_DIM, dtype=np.float32)
    FEATURE_STD = np.ones(FEATURE_DIM, dtype=np.float32)
    K = 24
    FEAT_DIM = FEATURE_DIM
    model = None


HISTORY = ReplayHistory()


def _source_garrison_keep(src, g):
    scene_step = int(g.get("step_idx", 0))
    keep = 1.0 if scene_step >= 440 else 8.0
    if hb_source_is_frontline(src, g.get("planets", []), int(src[1])) > 0.5:
        keep += 4.0
    return keep


def heuristic_ship_count_from_hb(src, tgt, hb, g):
    """Ship sizing dùng Hellburner metrics thay vì để model đoán số quân."""
    src_ships = float(src[5])
    if src_ships <= 1:
        return 0

    tgt_owner = int(tgt[1])
    player = int(src[1])

    if tgt_owner == player:
        # Reinforce: chỉ gửi surplus, ưu tiên giữ source an toàn.
        keep = _source_garrison_keep(src, g)
        surplus = max(0.0, src_ships - keep)
        if surplus <= 0:
            return 0
        # Nếu target của mình có defense urgency cao thì gửi nhiều hơn.
        urgency = hb["features"][HELLBURNER_FEATURE_NAMES.index("hb_defense_urgency")]
        frac = 0.35 + 0.35 * float(urgency)
        ships = int(max(1, min(surplus, src_ships * frac)))
        return max(0, min(int(src_ships - 1), ships))

    required = float(hb.get("required", 0.0))
    if not math.isfinite(required) or required <= 0:
        required = hb_required_ships(src, tgt, hb.get("travel", math.inf))

    # Attack/capture: gửi đủ required, thêm buffer nhỏ nhưng không rút hở source.
    keep = _source_garrison_keep(src, g)
    max_send = int(max(0.0, src_ships - keep))
    if max_send <= 0:
        return 0

    buffer = 1.0
    if tgt_owner != -1:
        buffer += 1.0

    ships = int(math.ceil(required + buffer))
    ships = max(1, min(max_send, ships))

    # Nếu không đủ gần required thì bỏ, tránh gửi quân "vớ vẩn".
    if ships + 0.5 < required:
        return 0

    return ships



# =============================================================================
# ROBUST GLOBAL PLANNER v2
# =============================================================================

def _fleet_first_planet_hit(fleet, planets, angular_velocity, scene_step):
    """
    Suy ra fleet hiện tại đang bay vào planet nào.
    Chú ý: fleet[5] trong Orbit Wars là from_planet_id, KHÔNG phải destination_id.
    """
    fx, fy = float(fleet[2]), float(fleet[3])
    angle = float(fleet[4])
    from_id = int(fleet[5])
    ships = max(1.0, float(fleet[6]))

    pseudo_src = [from_id, int(fleet[1]), fx, fy, 0.0, ships, 0.0]

    best = None
    best_t = float("inf")

    for p in planets:
        if int(p[0]) == from_id:
            continue

        need_angle, px, py, travel = hb_intercept_planet(
            pseudo_src, p, ships, angular_velocity, scene_step
        )
        if not math.isfinite(travel):
            continue

        d = dist_xy(fx, fy, px, py)
        radius = float(p[4])
        half_cone = math.pi if d < radius else math.asin(min(1.0, radius / max(d, EPS)))
        if angle_diff(angle, need_angle) <= half_cone and travel < best_t:
            best_t = travel
            best = p

    if best is None:
        return None, math.inf

    ex = fx + best_t * hb_fleet_speed(ships) * math.cos(angle)
    ey = fy + best_t * hb_fleet_speed(ships) * math.sin(angle)
    if hb_segment_crosses_sun(fx, fy, ex, ey):
        return None, math.inf

    return best, best_t


def _compute_incoming(obs, player):
    """Reconstruct incoming ships by destination planet id."""
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    step = int(obs.get("step", 0))

    incoming_enemy = defaultdict(float)
    incoming_ally = defaultdict(float)
    incoming_enemy_eta = defaultdict(lambda: 999.0)

    for f in fleets:
        dest, eta = _fleet_first_planet_hit(f, planets, angular_velocity, step)
        if dest is None:
            continue
        did = int(dest[0])
        ships = float(f[6])
        if int(f[1]) == player:
            incoming_ally[did] += ships
        elif int(f[1]) != -1:
            incoming_enemy[did] += ships
            incoming_enemy_eta[did] = min(incoming_enemy_eta[did], float(eta))

    return incoming_enemy, incoming_ally, incoming_enemy_eta


def _is_friendly_urgent(tgt, incoming_enemy, incoming_ally):
    """Planet của mình chỉ đáng reinforce nếu thật sự sắp nguy."""
    pid = int(tgt[0])
    enemy_in = float(incoming_enemy.get(pid, 0.0))
    ally_in = float(incoming_ally.get(pid, 0.0))
    if enemy_in <= 0:
        return False
    # Cần quân nếu incoming enemy có thể vượt qua garrison hiện tại + ally incoming.
    return enemy_in + 1.0 > float(tgt[5]) + ally_in


def _action_keep(src, tgt, g):
    """Garrison giữ lại ở source, nới lỏng early game để không bị chậm expand."""
    step = int(g.get("step_idx", 0))
    player = int(src[1])
    tgt_owner = int(tgt[1])

    if step >= 440:
        keep = 1.0
    elif tgt_owner == -1 and step <= 70:
        # Mấu chốt: đầu game phải dám rời 1-2 quân để chiếm neutral nhanh.
        keep = 1.0 if float(src[6]) <= 3 else 2.0
    elif tgt_owner == -1 and step <= 150:
        keep = 4.0
    else:
        keep = 7.0

    if tgt_owner not in (-1, player):
        keep += 2.0

    # Frontline thì giữ thêm, nhưng không quá gắt đầu game.
    if step > 70 and hb_source_is_frontline(src, g.get("planets", []), player) > 0.5:
        keep += 3.0

    return keep


def _ships_for_action(src, tgt, travel, g, incoming_enemy, incoming_ally):
    """
    Ship sizing mới:
    - Early neutral expansion được ưu tiên mạnh.
    - Không gửi quân thiếu required.
    - Không reinforce friendly nếu không urgent.
    """
    src_ships = float(src[5])
    player = int(src[1])
    tgt_owner = int(tgt[1])
    step = int(g.get("step_idx", 0))
    keep = _action_keep(src, tgt, g)
    max_send = int(max(0.0, src_ships - keep))
    if max_send <= 0:
        return 0

    eta_turns = max(1, int(math.ceil(travel))) if math.isfinite(travel) else 99

    if tgt_owner == player:
        if not _is_friendly_urgent(tgt, incoming_enemy, incoming_ally):
            return 0
        pid = int(tgt[0])
        deficit = max(0.0, float(incoming_enemy.get(pid, 0.0)) + 2.0 - float(tgt[5]) - float(incoming_ally.get(pid, 0.0)))
        ships = int(min(max_send, max(5.0, deficit)))
        return ships if ships >= 4 else 0

    if tgt_owner == -1:
        # Neutral không sinh quân, nên đừng over-send quá nhiều.
        buffer = 1.0
        if step <= 80 and float(tgt[6]) >= 4:
            buffer = 1.0
        elif step > 150:
            buffer = 2.0
        required = float(tgt[5]) + buffer
        ships = int(math.ceil(required))
        if ships > max_send:
            return 0
        # Cho phép fleet nhỏ nếu đó là đúng required để chiếm neutral nhỏ.
        if step > 120 and ships < 5:
            return 0
        return max(1, ships)

    # Enemy planet: tính production trong ETA.
    required = float(tgt[5]) + float(tgt[6]) * eta_turns + 3.0
    ships = int(math.ceil(required))
    if ships > max_send:
        return 0
    if ships < 6 and step < 430:
        return 0
    return max(1, ships)


def _proposal_heuristic_score(src, tgt, ships, travel, model_score, g, incoming_enemy, incoming_ally):
    player = int(src[1])
    owner = int(tgt[1])
    step = int(g.get("step_idx", 0))
    prod = float(tgt[6])
    tgt_ships = float(tgt[5])
    turns_left = max(1.0, 500.0 - step - float(travel))
    roi = prod * turns_left / (ships + 1.0)
    dist_pen = float(travel) / 25.0

    # Convert raw model score into a bounded additive term.
    try:
        ms = float(model_score)
        if not math.isfinite(ms):
            ms = 0.0
    except Exception:
        ms = 0.0
    ms = max(-5.0, min(5.0, ms))

    if owner == player:
        urgent = 1.0 if _is_friendly_urgent(tgt, incoming_enemy, incoming_ally) else 0.0
        return 25.0 * urgent + 0.25 * ms - dist_pen

    if owner == -1:
        # Early game: expansion tempo quan trọng hơn việc model bắt chước replay.
        early_bonus = 0.0
        if step <= 90:
            early_bonus = 4.0 * prod - 0.15 * tgt_ships
        corner_bonus = 0.4 if (float(tgt[2]) < 15 or float(tgt[2]) > 85 or float(tgt[3]) < 15 or float(tgt[3]) > 85) else 0.0
        return early_bonus + 1.25 * roi + 0.35 * ms - 1.2 * dist_pen + corner_bonus

    # Attack enemy: chỉ thực sự ưu tiên khi ROI tốt hoặc enemy yếu.
    enemy_power = g["totals"][owner]["total_ships"] if owner in g["totals"] else 0.0
    weak_bonus = 1.0 if enemy_power < g.get("my_total", 0.0) else 0.0
    return 1.0 * roi + 0.45 * ms - 1.4 * dist_pen + weak_bonus


def _build_global_proposals(obs, g, use_model=True):
    player = int(obs["player"])
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    step = int(obs.get("step", 0))
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    incoming_enemy, incoming_ally, incoming_eta = _compute_incoming(obs, player)

    proposals = []
    my_planets = [p for p in planets if int(p[1]) == player]

    for src in sorted(my_planets, key=lambda p: float(p[5]), reverse=True):
        src_ships = float(src[5])
        if src_ships < 4:
            continue

        candidates = build_candidates(src, planets, K=K)
        if not candidates:
            continue

        model_scores = np.zeros(len(candidates), dtype=np.float32)

        if use_model and MODEL_OK:
            features = np.zeros((K, FEAT_DIM), dtype=np.float32)
            mask = np.zeros(K, dtype=np.bool_)

            for i, (tgt, d) in enumerate(candidates[:K]):
                try:
                    # Không mask bằng hb["valid"] nữa, vì hb source_safe quá gắt đầu game.
                    # Chỉ feature hóa; safety sẽ kiểm tra bằng ship/action logic bên dưới.
                    row = extract_candidate_features(src, tgt, d, g)
                    if len(row) != FEAT_DIM:
                        continue
                    features[i] = np.asarray(row, dtype=np.float32)
                    mask[i] = True
                except Exception:
                    continue

            if mask.any():
                features = (features - FEATURE_MEAN[None, :]) / (FEATURE_STD[None, :] + 1e-6)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                x = torch.tensor(features[None, :, :], dtype=torch.float32, device=DEVICE)
                m = torch.tensor(mask[None, :], dtype=torch.bool, device=DEVICE)
                with torch.no_grad():
                    raw = model(x, m)[0].detach().cpu().numpy()
                for i in range(min(len(candidates), K)):
                    if np.isfinite(raw[i]):
                        model_scores[i] = float(raw[i])

        for i, (tgt, d) in enumerate(candidates[:K]):
            owner = int(tgt[1])

            # Friendly chỉ cho reinforce thật sự khẩn cấp.
            if owner == player and not _is_friendly_urgent(tgt, incoming_enemy, incoming_ally):
                continue

            # Không đánh enemy quá sớm khi còn nhiều neutral ngon.
            if owner not in (-1, player) and step < 80:
                continue

            # Tính intercept bằng lượng quân sơ bộ lớn, sau đó tính ships, rồi intercept lại.
            _, _, _, coarse_travel = hb_intercept_planet(src, tgt, max(1.0, src_ships), angular_velocity, step)
            if not math.isfinite(coarse_travel):
                continue

            ships = _ships_for_action(src, tgt, coarse_travel, g, incoming_enemy, incoming_ally)
            if ships <= 0:
                continue

            angle, ix, iy, travel = hb_intercept_planet(src, tgt, ships, angular_velocity, step)
            if not math.isfinite(travel):
                continue

            # Final strict path safety: không xuyên Sun, không bắn nhầm planet.
            if hb_segment_crosses_sun(float(src[2]), float(src[3]), ix, iy, margin=0.75):
                continue
            hit = hb_first_planet_hit(src, angle, ships, planets, angular_velocity, step)
            if hit is None or int(hit[0]) != int(tgt[0]):
                continue

            score = _proposal_heuristic_score(
                src, tgt, ships, travel, model_scores[i] if i < len(model_scores) else 0.0,
                g, incoming_enemy, incoming_ally
            )

            proposals.append({
                "score": float(score),
                "src_id": int(src[0]),
                "tgt_id": int(tgt[0]),
                "src": src,
                "tgt": tgt,
                "angle": float(angle),
                "ships": int(ships),
                "travel": float(travel),
                "owner": owner,
            })

    return proposals



def _filtered_obs(obs):
    """
    Remove comet planets before target selection/intercept.
    Hellburner gốc làm việc này bằng obs["comet_planet_ids"].
    Nếu giữ comet trong danh sách planet, heuristic có thể bắn vào comet như planet thường
    trong khi quỹ đạo comet không được hb_intercept_planet dự đoán đúng -> dễ miss.
    """
    try:
        comet_ids = set(int(x) for x in obs.get("comet_planet_ids", []))
    except Exception:
        comet_ids = set()
    if not comet_ids:
        return obs

    new_obs = dict(obs)
    new_obs["planets"] = [p for p in obs.get("planets", []) if int(p[0]) not in comet_ids]
    if "initial_planets" in obs:
        new_obs["initial_planets"] = [p for p in obs.get("initial_planets", []) if int(p[0]) not in comet_ids]
    return new_obs


def fallback_rule_agent(obs):
    obs = _filtered_obs(obs)
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    if not planets:
        return []
    g = compute_step_globals(int(obs.get("step", 0)), 500.0, player, planets, fleets, HISTORY)
    g["angular_velocity"] = float(obs.get("angular_velocity", 0.0))
    proposals = _build_global_proposals(obs, g, use_model=False)
    return _commit_global_proposals(proposals, obs)


def _commit_global_proposals(proposals, obs):
    """Commit proposal toàn cục, có cập nhật source ships giả lập để tránh spam."""
    step = int(obs.get("step", 0))
    player = int(obs.get("player", 0))
    planets = obs.get("planets", [])

    if not proposals:
        return []

    # Giới hạn số move theo phase. Early cần expand nhưng không spam toàn map.
    if step < 80:
        max_moves = 2
    elif step < 250:
        max_moves = 3
    else:
        max_moves = 5

    proposals = sorted(proposals, key=lambda z: z["score"], reverse=True)

    remaining = {int(p[0]): float(p[5]) for p in planets}
    used_sources = set()
    target_counts = defaultdict(int)
    moves = []

    for pr in proposals:
        if len(moves) >= max_moves:
            break
        sid = pr["src_id"]
        tid = pr["tgt_id"]
        src = pr["src"]
        tgt = pr["tgt"]
        ships = int(pr["ships"])

        if sid in used_sources:
            continue
        if remaining.get(sid, 0.0) < ships + _action_keep(src, tgt, {"step_idx": step, "planets": planets}):
            continue

        # Không dồn quá nhiều source vào cùng một target trong bản đơn giản này.
        # Riêng enemy target có thể cần 2 source.
        if target_counts[tid] >= (2 if int(tgt[1]) not in (-1, player) else 1):
            continue

        # Score quá thấp thì thôi, tránh gửi action chỉ vì có proposal.
        if pr["score"] < 0.2 and step < 430:
            continue

        remaining[sid] -= ships
        used_sources.add(sid)
        target_counts[tid] += 1
        moves.append([int(sid), float(pr["angle"]), int(ships)])

    return moves


def agent(obs):
    try:
        obs = _filtered_obs(obs)
        player = int(obs.get("player", 0))
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        step = int(obs.get("step", 0))
        if not planets:
            return []

        my_planets = [p for p in planets if int(p[1]) == player]
        if not my_planets:
            return []

        g = compute_step_globals(
            step_idx=step,
            total_steps=500.0,
            player=player,
            planets=planets,
            fleets=fleets,
            history=HISTORY,
        )
        g["angular_velocity"] = float(obs.get("angular_velocity", 0.0))

        if not MODEL_OK:
            return fallback_rule_agent(obs)

        proposals = _build_global_proposals(obs, g, use_model=True)
        moves = _commit_global_proposals(proposals, obs)
        return moves

    except Exception:
        return []
