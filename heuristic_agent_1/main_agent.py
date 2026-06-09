# -*- coding: utf-8 -*-
"""
Luồng xử lý tổng quát của agent:
1. agent(obs) được Kaggle gọi mỗi turn.
2. HeuristicAgent.main(obs) đọc observation hiện tại.
3. Chuyển planet/fleet thô từ obs thành object dễ thao tác.
4. Xây thông tin quỹ đạo cho planet quay quanh mặt trời.
5. Xây graph gần-kề giữa các planet để giới hạn phạm vi xét nước đi.
6. Dự đoán các fleet đang bay sẽ đâm vào planet nào.
7. Nếu đang ở early game, chạy planner riêng để chiếm neutral planet.
8. Nếu đã qua early game, đánh giá từng target bằng mô phỏng timeline + ROI.
9. Commit các move tốt nhất vào bản mô phỏng nội bộ để tránh gửi trùng/lố quân.
10. Gửi thêm reinforce từ hậu phương ra tiền tuyến.
11. Trả về danh sách move dạng [source_planet_id, angle, ships].
"""

import math
import copy
from collections import defaultdict
from dataclasses import dataclass, field

"""
Import các class và hàm từ môi trường:
- Fleet: class đại diện cho một fleet đang bay trong game.
- CENTER: tọa độ trung tâm của bản đồ.
- ROTATION_RADIUS_LIMIT: bán kính tối đa để một planet có thể quay quanh mặt trời
- SUN_RADIUS: bán kính của mặt trời.
- distance: hàm tính khoảng cách giữa hai điểm.
- point_to_segment_distance: hàm tính khoảng cách từ một điểm đến một đoạn thẳng
"""
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance
)

# Hằng số game
GAME_LENGTH = 500

# -----------------------------------------------------------------------------
# Planet
# -----------------------------------------------------------------------------
# Class này là lớp biểu diễn planet nội bộ của agent.
# Dữ liệu planet trong obs ban đầu là list/tuple dạng:
#     [id, owner, x, y, radius, ships, production]
# Việc đóng gói thành object giúp code dễ đọc hơn, ví dụ dùng p.ships thay vì p[5].
class Planet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        # id: định danh duy nhất của planet, dùng khi tạo move [source_id, angle, ships].
        self.id = id

        # owner: chủ sở hữu planet.
        # - owner == self.player: planet của mình.
        # - owner == -1: planet trung lập.
        # - owner khác self.player và khác -1: planet của đối thủ.
        self.owner = owner

        # x, y: tọa độ hiện tại của planet trên bản đồ.
        # Với planet quay, đây là vị trí hiện tại; agent còn phải dự đoán vị trí tương lai.
        self.x = x
        self.y = y

        # radius: bán kính planet, dùng để tính fleet có bắn trúng planet hay không.
        self.radius = radius

        # ships: số quân hiện có trên planet. Đây là tài nguyên dùng để attack/defend.
        self.ships = ships

        # production: tốc độ sinh quân mỗi turn. Planet production cao thường đáng chiếm/giữ hơn.
        self.production = production

        # reinforcement_target: planet mà planet này nên gửi tiếp viện tới.
        # Ban đầu None; sau đó build_reinforcement_targets() sẽ gán nếu planet này là hậu phương.
        self.reinforcement_target: 'Planet | None' = None


# -----------------------------------------------------------------------------
# EarlyGameFleet
# -----------------------------------------------------------------------------
# Class này KHÔNG phải Fleet thật của Kaggle.
# Đây là fleet giả lập chỉ dùng trong early-game planner để thử trước các phương án.
@dataclass(slots=True)
class EarlyGameFleet:
    # source_id: id planet xuất phát.
    # Trong một số fleet giả lập tổng hợp, code dùng -1 nếu không cần truy vết nguồn cụ thể.
    source_id: int

    # destination_id: id planet đích mà fleet giả lập sẽ bay tới.
    destination_id: int

    # fleet_size: số quân được gửi đi trong fleet giả lập.
    fleet_size: int

    # garrison_on_arrival: số quân còn lại tại planet đích sau khi fleet tới nơi.
    # Ví dụ target có 10 quân, ta gửi 25 quân => còn 15 quân sau khi chiếm.
    garrison_on_arrival: int

    # arrival_turn: turn dự kiến fleet tới nơi trong mô phỏng early game.
    arrival_turn: int

    # is_capture:
    # - True: fleet này dùng để chiếm planet mới, khi tới nơi sẽ thêm planet vào state.owned.
    # - False: fleet này chỉ là tiếp viện cho planet đã thuộc về mình.
    is_capture: bool


# -----------------------------------------------------------------------------
# EarlyGameState
# -----------------------------------------------------------------------------
# Đây là snapshot trạng thái game giả lập dùng cho search/planning ở đầu game.
# Agent dùng object này để thử nhiều chuỗi chiếm neutral planet mà không sửa trạng thái thật.
@dataclass(slots=True)
class EarlyGameState:
    # turn: turn hiện tại trong trạng thái giả lập.
    turn: int

    # garrison: dict ánh xạ planet_id -> số quân hiện có trên planet đó.
    # Ví dụ {3: 20, 7: 15} nghĩa là planet 3 có 20 quân, planet 7 có 15 quân.
    garrison: dict

    # production: dict ánh xạ planet_id -> production của planet đó.
    # Khi mô phỏng đi qua mỗi turn, garrison sẽ được cộng thêm production.
    production: dict

    # owned: tập id các planet hiện đang thuộc về agent trong mô phỏng.
    owned: set

    # fleets: danh sách EarlyGameFleet đang bay trong mô phỏng.
    # default_factory=list tránh lỗi dùng chung list giữa nhiều EarlyGameState khác nhau.
    fleets: list = field(default_factory=list)


class HeuristicAgent:
    # -------------------------------------------------------------------------
    # Các tham số chiến thuật chính của agent
    # -------------------------------------------------------------------------

    # Tốc độ tối đa của fleet. Fleet càng nhiều quân thì càng nhanh, nhưng không vượt quá mốc này.
    SHIP_SPEED_MAX: float = 6.0

    # Số turn đầu dùng chiến lược early-game riêng để mở rộng nhanh vào planet neutral.
    EARLY_ROUNDS: int = 3

    # Horizon của early-game planner: agent nhìn trước bao nhiêu turn để tính lợi ích chiếm planet.
    EARLY_LOOK_AHEAD: int = 33

    # Khoảng cách tối đa để coi hai planet là "có thể tương tác" trong proximity graph.
    # Giúp giảm số target/source phải xét, tránh tấn công quá xa ở mid game.
    MAX_DISTANCE: int = 38

    # Khi xây graph, agent ước lượng vị trí planet quay sau vài turn để graph thực tế hơn.
    ROTATION_LOOK_AHEAD: int = 10

    # Ngưỡng quân tối thiểu muốn gửi trong một lần reinforcement bình thường.
    REINFORCEMENT_SIZE: int = 17

    # Số quân agent cố giữ lại ở mỗi planet sau khi gửi quân đi.
    GARRISON_SIZE: int = 8

    def __init__(self):
        """Khởi tạo các biến trạng thái được reset lại mỗi lần agent xử lý một obs."""

        # id người chơi hiện tại do môi trường truyền vào obs['player'].
        self.player: int = 0

        # turn hiện tại, code dùng obs['step'] - 1 để quy về chỉ số bắt đầu từ 0.
        self.scene_step: int = 0

        # tốc độ góc của các planet quay quanh tâm bản đồ.
        self.angular_velocity: float = 0.0

        # danh sách tất cả planet thường, đã loại bỏ comet planet.
        self.planets: list = []

        # subset các planet đang thuộc về mình.
        self.owned_planets: list = []

        # subset các planet không thuộc về mình, bao gồm neutral và enemy.
        self.enemy_planets: list = []

        # danh sách fleet thật đang bay trong game, chuyển từ obs['fleets'] sang object Fleet.
        self.fleets: list = []

        # orbital_info[p] = None nếu p đứng yên; ngược lại = (bán kính quay, góc ban đầu).
        self.orbital_info: dict = {}

        # inbound_edges[dst] = [(src, distance), ...]: các planet có thể đi tới dst trong ngưỡng gần.
        self.inbound_edges: dict = {}

        # outbound_edges[src] = [(dst, distance), ...]: các planet mà src có thể đi tới.
        self.outbound_edges: dict = {}

        # future_pos[p] = vị trí ước lượng của planet p sau một khoảng lookahead.
        self.future_pos: dict = {}

        # destination_list[planet] = danh sách fleet dự đoán sẽ tới planet đó.
        # Mỗi entry có dạng (owner, ships, travel, start_x, start_y, hit_x, hit_y).
        self.destination_list: dict = {}

        # True nếu còn ít nhất 2 đối thủ khác nhau trên map; khi đó agent cẩn thận hơn.
        self._multifront: bool = False

        # True nếu phát hiện hai enemy đang đánh nhau; dùng để kích hoạt FFA exploit.
        self._inter_enemy_conflict: bool = False

    def fleet_speed(self, ships):
        """
        Tính tốc độ bay của fleet dựa trên số quân.

        Tham số:
        - ships: số quân trong fleet.

        Ý nghĩa:
        - Trong Orbit Wars, fleet nhiều quân thường bay nhanh hơn fleet ít quân.
        - Hàm này mô phỏng quy luật đó bằng công thức log, rồi chặn trên bởi SHIP_SPEED_MAX.
        - Tốc độ này ảnh hưởng trực tiếp tới việc bắn đón đầu planet quay.
        """
        # math.log(ships) làm tốc độ tăng chậm dần theo số quân.
        # Lũy thừa 1.5 làm fleet lớn nhanh hơn rõ hơn, nhưng vẫn bị giới hạn bởi SHIP_SPEED_MAX.
        return min(self.SHIP_SPEED_MAX, 1.0 + (self.SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)

    def _count_enemy_owners(self):
        """
        Đếm số người chơi đối thủ còn sở hữu planet trên bản đồ.

        Kết quả dùng để phát hiện multi-front / FFA:
        - Nếu chỉ có 1 enemy: game gần giống 1v1.
        - Nếu có >= 2 enemy: cần cẩn thận hơn vì có thể bị nhiều hướng tấn công.
        """
        owners = set()
        for p in self.planets:
            if p.owner not in (-1, self.player):
                owners.add(p.owner)
        return len(owners)

    # Detect nếu 2 enemy đang có fleet bay về hành tinh của nhau
    def _detect_inter_enemy_conflict(self):
        """
        Phát hiện tình huống hai đối thủ đang đánh nhau.

        Cách làm:
        - Duyệt các fleet đã được dự đoán đích đến trong destination_list.
        - Nếu planet đích thuộc enemy A nhưng fleet tới từ enemy B, tức là enemy B đang tấn công enemy A.

        Trả về:
        - True nếu có ít nhất một cuộc tấn công enemy-vs-enemy.
        - False nếu không phát hiện.
        """
        """Returns True nếu có ít nhất 1 enemy fleet đang tấn công planet của enemy khác."""
        for dest_planet, arrivals in self.destination_list.items():
            planet_owner = dest_planet.owner
            if planet_owner in (-1, self.player):
                continue
            for owner, _, _, _, _, _, _ in arrivals:
                if owner not in (-1, self.player) and owner != planet_owner:
                    return True
        return False

    def build_orbital_info(self, initial_planets):
        """
        Xác định planet nào là planet quay và lưu thông tin quỹ đạo của nó.

        Tham số:
        - initial_planets: danh sách planet ở trạng thái ban đầu, dùng để lấy góc ban đầu.

        Kết quả:
        - self.orbital_info[p] = None nếu p là planet đứng yên.
        - self.orbital_info[p] = (r, initial_angle) nếu p quay quanh CENTER.

        Vai trò trong pipeline:
        - Các hàm intercept_planet và build_proximity_graph cần biết quỹ đạo để dự đoán vị trí tương lai.
        """
        cx = cy = CENTER
        ip_by_id = {ip[0]: ip for ip in initial_planets}
        self.orbital_info = {}
        for p in self.planets:
            # r là khoảng cách từ planet tới tâm bản đồ/mặt trời.
            # Nếu r + radius nằm trong ROTATION_RADIUS_LIMIT thì planet thuộc vùng quay.
            r = distance((p.x, p.y), (cx, cy))
            if r + p.radius < ROTATION_RADIUS_LIMIT and p.id in ip_by_id:
                ip = ip_by_id[p.id]
                # Lưu bán kính quay và góc ban đầu.
                # ip[2], ip[3] là tọa độ ban đầu của planet trong initial_planets.
                self.orbital_info[p] = (r, math.atan2(ip[3] - cy, ip[2] - cx))
            else:
                # None nghĩa là planet đứng yên, không cần dự đoán quỹ đạo.
                self.orbital_info[p] = None

    def build_proximity_graph(self):
        """
        Xây graph gần-kề giữa các planet.

        Ý tưởng:
        - Không xét mọi cặp source-target trên bản đồ vì sẽ tốn và dễ tạo move quá xa.
        - Chỉ nối cạnh nếu khoảng cách từ source tới target <= MAX_DISTANCE.

        Kết quả:
        - inbound_edges[dst]: các source có thể đi tới dst.
        - outbound_edges[src]: các target mà src có thể đi tới.

        Vai trò:
        - Giới hạn không gian hành động.
        - Xác định frontline và reinforcement route.
        """
        cx = cy = CENTER
        self.future_pos = {}
        for p in self.planets:
            orb = self.orbital_info[p]
            if orb is not None:
                r, ia = orb
                # Góc tương lai = góc ban đầu + vận tốc góc * thời gian.
                # Dùng lookahead để graph phản ánh vị trí gần tương lai thay vì chỉ vị trí hiện tại.
                a = ia + self.angular_velocity * (self.scene_step + 1 + self.ROTATION_LOOK_AHEAD)
                self.future_pos[p] = (cx + r * math.cos(a), cy + r * math.sin(a))
            else:
                self.future_pos[p] = (p.x, p.y)
        self.inbound_edges = {p: [] for p in self.planets}
        for src in self.planets:
            for dst in self.planets:
                if dst is src:
                    continue
                travel = distance((src.x, src.y), self.future_pos[dst])
                # Chỉ tạo cạnh nếu khoảng cách nằm trong ngưỡng MAX_DISTANCE.
                # Đây là cách agent hạn chế việc xét các move quá xa/rủi ro thấp.
                if travel <= self.MAX_DISTANCE:
                    self.inbound_edges[dst].append((src, travel))
        self.outbound_edges = {p: [] for p in self.planets}
        for dst, inbound in self.inbound_edges.items():
            for src, travel in inbound:
                self.outbound_edges[src].append((dst, travel))

    def build_reinforcement_targets(self):
        """
        Gán mục tiêu tiếp viện cho các planet hậu phương.

        Cách làm:
        1. Xác định front_line: planet của mình nằm gần enemy/neutral.
        2. BFS ngược từ frontline để biết planet hậu phương cách frontline bao nhiêu hop.
        3. Với mỗi planet hậu phương, chọn reinforcement_target hợp lý.

        Vai trò:
        - Tạo luồng quân từ hậu phương ra tiền tuyến.
        - Tránh để quân nằm yên ở các planet xa chiến trường.
        """
        # Một planet của mình được coi là frontline nếu nó tiếp xúc với planet không thuộc mình,
        # theo cả chiều inbound và outbound trong proximity graph.
        front_line = {
            p for p in self.owned_planets
            if any(src.owner != self.player for src, _ in self.inbound_edges[p])
            or any(dst.owner != self.player for dst, _ in self.outbound_edges[p])
        }
        hops_to_front = {p: 0 for p in front_line}
        queue = list(front_line)
        head = 0
        # BFS ngược từ frontline để tìm các planet hậu phương có đường dẫn về frontline.
        while head < len(queue):
            node = queue[head]; head += 1
            for src, _ in self.inbound_edges[node]:
                if src.owner != self.player or src in hops_to_front:
                    continue
                hops_to_front[src] = hops_to_front[node] + 1
                queue.append(src)
        for p in self.owned_planets:
            p.reinforcement_target = None
            # Frontline tự nó không cần reinforcement_target; nó là nơi nhận tiếp viện.
            if p in front_line:
                continue
            direct_front = [dst for dst, _ in self.outbound_edges[p] if dst in front_line]
            if direct_front:
                p.reinforcement_target = min(direct_front, key=lambda d: d.ships)
                continue
            reachable = [
                dst for dst, _ in self.outbound_edges[p]
                if dst.owner == self.player and dst not in front_line and dst in hops_to_front
            ]
            if reachable:
                p.reinforcement_target = min(reachable, key=lambda d: (hops_to_front[d], d.ships))

    def intercept_planet(self, sx, sy, target, ships, tol=1e-6, max_iters=30):
        """
        Tính góc bắn để fleet từ (sx, sy) có thể chạm target.

        Tham số:
        - sx, sy: tọa độ xuất phát của fleet.
        - target: planet đích.
        - ships: số quân gửi đi, dùng để tính tốc độ fleet.
        - tol: sai số hội tụ khi target là planet quay.
        - max_iters: số vòng lặp tối đa để giải bài toán đón đầu.

        Trả về:
        - angle: góc cần bắn.
        - tx, ty: tọa độ va chạm dự kiến.
        - travel: thời gian bay dự kiến.

        Vai trò:
        - Đây là hàm vật lý quan trọng nhất để bắn trúng planet quay.
        """
        # Tốc độ fleet phụ thuộc số quân gửi đi.
        speed = self.fleet_speed(ships)
        orb = self.orbital_info[target]
        # Nếu target đứng yên, bài toán đơn giản: bắn thẳng vào vị trí hiện tại.
        if orb is None:
            tx, ty = target.x, target.y
            travel = distance((sx, sy), (tx, ty)) / speed
        else:
            cx = cy = CENTER
            r, ia = orb
            travel = distance((sx, sy), (target.x, target.y)) / speed
            # Nếu target quay, cần giải lặp: thời gian bay phụ thuộc vị trí target tương lai,
            # còn vị trí target tương lai lại phụ thuộc thời gian bay.
            for _ in range(max_iters):
                a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
                new_tx, new_ty = cx + r * math.cos(a), cy + r * math.sin(a)
                new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
                new_travel = 0.5 * (travel + new_travel - 0.5)
                # Nếu thời gian bay mới gần bằng thời gian bay cũ, coi như đã hội tụ.
                if abs(new_travel - travel) < tol:
                    travel = new_travel
                    break
                travel = new_travel
            else:
                return 0.0, target.x, target.y, math.inf
            a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
        # atan2 cho góc bắn từ source tới điểm va chạm dự kiến.
        angle = math.atan2(ty - sy, tx - sx)
        return angle, tx, ty, travel

    def first_planet_hit(self, sx, sy, angle, ships, source):
        """
        Dự đoán planet đầu tiên mà fleet sẽ va vào nếu bắn theo angle.

        Tham số:
        - sx, sy: tọa độ xuất phát.
        - angle: góc bắn đang kiểm tra.
        - ships: số quân trong fleet, dùng để tính tốc độ.
        - source: planet xuất phát, cần bỏ qua vì fleet xuất phát từ đó.

        Trả về:
        - Planet đầu tiên bị bắn trúng, hoặc None nếu không trúng planet hợp lệ.

        Kiểm tra quan trọng:
        - Nếu đường bay cắt qua mặt trời thì trả None để tránh fleet chết.
        """
        best = None
        best_t = float('inf')
        for planet in self.planets:
            if planet is source:
                continue
            needed_angle, px, py, travel = self.intercept_planet(sx, sy, planet, ships)
            dist = distance((sx, sy), (px, py))
            # half_cone là nửa góc va chạm: planet càng lớn/càng gần thì góc cho phép càng rộng.
            half_cone = math.pi if dist < planet.radius else math.asin(min(1.0, planet.radius / dist))
            # delta là độ lệch góc nhỏ nhất giữa hướng bắn thật và hướng cần để trúng planet.
            delta = abs(math.atan2(math.sin(angle - needed_angle), math.cos(angle - needed_angle)))
            if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                best_t = travel
                best = planet
        if best is None:
            return None
        ex = sx + best_t * self.fleet_speed(ships) * math.cos(angle)
        ey = sy + best_t * self.fleet_speed(ships) * math.sin(angle)
        # Kiểm tra đường bay có đi qua vùng mặt trời không.
        # Nếu có, fleet có thể bị mặt trời chặn nên move bị loại.
        if point_to_segment_distance((CENTER, CENTER), (sx, sy), (ex, ey)) <= SUN_RADIUS:
            return None
        return best

    def build_destination_list(self):
        """
        Dự đoán đích đến của tất cả fleet thật đang bay trong game.

        Cách làm:
        - Với mỗi fleet hiện có, thử xem nó sẽ chạm planet nào đầu tiên.
        - Lưu fleet đó vào destination_list của planet tương ứng.

        Vai trò:
        - Cung cấp dữ liệu cho simulate_planet_timeline.
        - Giúp agent biết planet nào đang bị tấn công hoặc được tiếp viện.
        """
        # Reset danh sách đích đến dự đoán ở mỗi turn.
        self.destination_list = defaultdict(list)
        for fleet in self.fleets:
            best = None
            best_t = float('inf')
            for planet in self.planets:
                needed_angle, px, py, travel = self.intercept_planet(fleet.x, fleet.y, planet, fleet.ships)
                dist = distance((fleet.x, fleet.y), (px, py))
                # half_cone là nửa góc va chạm: planet càng lớn/càng gần thì góc cho phép càng rộng.
                half_cone = math.pi if dist < planet.radius else math.asin(min(1.0, planet.radius / dist))
                delta = abs(math.atan2(math.sin(fleet.angle - needed_angle), math.cos(fleet.angle - needed_angle)))
                if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                    best_t = travel
                    best = (planet, travel, px, py)
            if best is not None:
                planet, travel, px, py = best
                self.destination_list[planet].append((fleet.owner, fleet.ships, travel, fleet.x, fleet.y, px, py))

    def simulate_planet_timeline(self, planet, destination_list):
        """
        Mô phỏng tương lai gần của một planet dựa trên các fleet sẽ tới nó.

        Tham số:
        - planet: planet cần mô phỏng.
        - destination_list: dict chứa các fleet dự kiến tới từng planet.

        Trả về:
        - cur_owner: chủ sở hữu cuối cùng sau khi xử lý các fleet.
        - excess_ships: lượng quân dư/margin của mình trong mô phỏng.

        Vai trò:
        - Dùng để quyết định có cần defend không.
        - Dùng để biết tấn công target đã đủ thắng chưa.
        - Dùng để đánh giá rủi ro mất source planet.
        """
        cur_owner = planet.owner
        entries = destination_list.get(planet)
        if not bool(entries):
            return cur_owner, 0
        # Gom các fleet theo turn đến nơi.
        # buckets[turn] = [(owner, ships), ...]
        buckets = defaultdict(list)
        for owner, ships, t, _, _, _, _ in entries:
            turn = max(1, math.ceil(t))
            buckets[turn].append((owner, ships))
        last_ships, last_t = entries[-1][1], entries[-1][2]
        last_turn = max(1, math.ceil(last_t))
        cur_ships = float(planet.ships)
        prod = planet.production
        cur_t = 0
        excess_ships = float('inf')
        for turn in sorted(buckets):
            elapsed = turn - cur_t
            # Nếu planet đang có chủ, nó sinh thêm quân trong khoảng thời gian elapsed.
            # Planet neutral owner == -1 thì không cộng production cho ai.
            if elapsed > 0 and cur_owner != -1:
                cur_ships += prod * elapsed
            cur_t = turn
            owner_ships = defaultdict(float)
            for owner, ships in buckets[turn]:
                owner_ships[owner] += ships
            if owner_ships:
                # Nếu nhiều phe cùng tới trong một turn, phe có tổng ships lớn nhất thắng phần giao tranh giữa fleets.
                sorted_owners = sorted(owner_ships.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_owners) == 1:
                    survivor_owner, survivor_ships = sorted_owners[0]
                else:
                    top_owner, top_ships = sorted_owners[0]
                    second_ships = sorted_owners[1][1]
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner if survivor_ships > 0 else -1
                if survivor_ships > 0:
                    if survivor_owner == cur_owner:
                        cur_ships += survivor_ships
                    else:
                        cur_ships -= survivor_ships
                        if cur_ships < 0:
                            cur_owner = survivor_owner
                            cur_ships = abs(cur_ships)
            # Sau lần fleet cuối cùng tới, tính margin còn lại nếu planet thuộc về mình.
            if turn >= last_turn:
                margin = cur_ships if cur_owner == self.player else 0.0
                excess_ships = min(excess_ships, margin)
        if excess_ships == float('inf'):
            excess_ships = 0.0
        excess_ships = min(excess_ships, last_ships)
        return cur_owner, excess_ships

    def _enemy_counter_wins(self, target, trial_dl):
        """
        Kiểm tra sau khi mình chiếm target thì enemy gần nhất có thể retake không.

        Tham số:
        - target: planet đang định chiếm/giữ.
        - trial_dl: destination_list giả lập sau khi thêm fleet của mình.

        Trả về:
        - True nếu enemy có thể phản công làm target không thuộc về mình.
        - False nếu chưa thấy counter nguy hiểm.

        Vai trò:
        - Heuristic 2-ply trong multi-front: không chỉ nhìn nước đi của mình, mà nhìn thêm phản ứng gần nhất của enemy.
        """
        """Returns True nếu enemy mạnh nhất kề cận có thể retake target."""
        best_enemy = None
        best_ships = 0
        # Chỉ xét enemy gần target theo proximity graph, vì enemy xa thường retake chậm/khó hơn.
        for src, _ in self.inbound_edges.get(target, []):
            if src.owner not in (self.player, -1) and src.ships > best_ships:
                best_enemy = src
                best_ships = src.ships
        if best_enemy is None or best_ships == 0:
            return False
        _, ex, ey, enemy_travel = self.intercept_planet(
            best_enemy.x, best_enemy.y, target, best_ships)
        if not math.isfinite(enemy_travel):
            return False
        counter_dl = {k: list(v) for k, v in trial_dl.items()}
        counter_dl.setdefault(target, [])
        counter_dl[target].append(
            (best_enemy.owner, best_ships, enemy_travel,
             best_enemy.x, best_enemy.y, ex, ey))
        end_owner, _ = self.simulate_planet_timeline(target, counter_dl)
        return end_owner != self.player

    def _source_exposure_penalty(self, fleet_orders):
        """
        Tính penalty nếu việc gửi fleet làm source planet bị hở và có thể mất.

        Tham số:
        - fleet_orders: danh sách move dự kiến [source_id, angle, ships].

        Trả về:
        - Tổng production của các source planet có nguy cơ bị flip sau khi rút quân.

        Vai trò:
        - Tránh over-extend: không rút quân khỏi planet quan trọng nếu điều đó khiến nó bị enemy chiếm.
        """
        """
        Returns production sum của các source planet sẽ flip sau khi commit fleet.
        NÂNG CẤP: dùng 0.75x enemy ships thay vì 0.5x để estimate conservatively hơn.
        """
        penalty = 0
        for order in fleet_orders:
            src_id, _, ships_sent = order
            src = next((p for p in self.planets if p.id == src_id), None)
            if src is None or src.owner != self.player:
                continue
            # Copy destination_list để thử tình huống xấu mà không làm hỏng mô phỏng chính.
            worst_case_dl = {k: list(v) for k, v in self.destination_list.items()}
            worst_case_dl.setdefault(src, [])
            for attacker, _ in self.inbound_edges.get(src, []):
                if attacker.owner == self.player or attacker.owner == -1 or attacker.ships == 0:
                    continue
                _, ax, ay, atk_travel = self.intercept_planet(attacker.x, attacker.y, src, attacker.ships)
                if not math.isfinite(atk_travel):
                    continue
                # NÂNG CẤP: 0.75x thay vì 0.5x — estimate pressure realistically hơn
                # Giả định enemy có thể dùng khoảng 75% quân để gây áp lực lên source.
                pressure_ships = max(1, int(attacker.ships * 0.75))
                worst_case_dl[src].append(
                    (attacker.owner, pressure_ships, atk_travel,
                     attacker.x, attacker.y, ax, ay))
            saved = src.ships
            src.ships = max(0, src.ships - ships_sent)
            exposed_owner, _ = self.simulate_planet_timeline(src, worst_case_dl)
            src.ships = saved
            if exposed_owner != self.player:
                penalty += src.production
        return penalty

    # NEW: Tính minimum travel time từ các owned planet đến target
    def _min_travel_to_target(self, target):
        """
        Tìm thời gian bay nhỏ nhất từ một planet của mình tới target.

        Vai trò:
        - Dùng trong ROI để ước lượng target sẽ bắt đầu sinh lợi sau bao lâu.
        - Target càng xa thì payoff_turns càng nhỏ, ROI càng giảm.
        """
        min_travel = math.inf
        # Chỉ xét enemy gần target theo proximity graph, vì enemy xa thường retake chậm/khó hơn.
        for src, _ in self.inbound_edges.get(target, []):
            if src.owner != self.player or src.ships == 0:
                continue
            _, _, _, t = self.intercept_planet(src.x, src.y, target, max(1, int(src.ships)))
            if math.isfinite(t):
                min_travel = min(min_travel, t)
        return min_travel

    # NEW: ROI score cho một target
    def _compute_roi_score(self, target, fleet_orders):
        """
        Tính điểm ROI cho target.

        Công thức ý tưởng:
            ROI = production * payoff_turns / (cost + 1)

        Trong đó:
        - production: sản lượng của target.
        - payoff_turns: số turn còn lại để khai thác target sau khi fleet đến.
        - cost: tổng số quân cần gửi.

        Vai trò:
        - Ưu tiên planet sinh lợi cao, gần, và chi phí chiếm thấp.
        """
        """
        ROI = production * payoff_turns / (cost + 1)
        payoff_turns = turns còn lại trong game sau khi fleet đến nơi
        cost = số ships tổng cộng gửi đi
        """
        # Số turn còn lại của trận đấu. Target càng chiếm muộn thì càng ít thời gian sinh lợi.
        turns_remaining = max(0, GAME_LENGTH - self.scene_step)
        min_travel = self._min_travel_to_target(target)
        if not math.isfinite(min_travel):
            return float(target.production)  # fallback
        payoff_turns = max(1, turns_remaining - math.ceil(min_travel))
        # cost là số quân phải bỏ ra. Nếu chưa có fleet_orders thì dùng target.ships + 1 làm ước lượng.
        cost = sum(ships for _, _, ships in fleet_orders) if fleet_orders else (target.ships + 1)
        roi = target.production * payoff_turns / (cost + 1)
        return roi

    # NEW: Defense urgency bonus — planet mình đang bị tấn công nhận thêm weight
    def _defense_urgency_bonus(self, target):
        """
        Tính bonus phòng thủ cho planet của mình đang sắp mất.

        Nếu simulate_planet_timeline cho thấy target sẽ không còn thuộc về mình,
        agent cộng thêm điểm target.production * 2 để ưu tiên cứu planet đó.
        """
        """
        Trả về bonus score nếu target là planet của mình đang sắp mất.
        Ưu tiên defend hơn attack vào target mới.
        """
        # Nếu đang attack planet không thuộc mình, cần chú ý fleet enemy khác có thể tới target trước/sau.
        if target.owner != self.player:
            return 0
        end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
        if end_owner != self.player:
            # Planet sắp mất — bonus production * 2 để defend ưu tiên cao
            return target.production * 2
        return 0

    def evaluate_frontline_strategy(self, target):
        """
        Tìm một tập fleet_orders đủ để chiếm hoặc giữ target.

        Cách làm:
        1. Lấy các planet của mình có cạnh gần tới target.
        2. Sắp xếp source gần nhất trước.
        3. Thử gửi quân từ từng source vào target.
        4. Sau mỗi lần gửi, mô phỏng timeline target.
        5. Nếu target thuộc về mình thì coi là battle_won.

        Trả về:
        - fleet_orders: danh sách move [source_id, angle, ships].
        - intercepts: vị trí/thời gian va chạm tương ứng để commit vào mô phỏng.
        - battle_won: True nếu phương án đủ thắng.

        Vai trò:
        - Đây là hàm lập kế hoạch chiến thuật chính cho một target cụ thể.
        """
        # Các source planet của mình có thể bắn tới target, ưu tiên source gần target trước.
        possible_origins = sorted(
            [(src, travel) for src, travel in self.inbound_edges.get(target, [])
             if src.owner == self.player], key=lambda x: x[1])
        fleet_orders = []
        intercepts = []
        # trial_destination_list là bản mô phỏng thử sau khi thêm các fleet của mình.
        trial_destination_list = {}
        for _p, _entries in self.destination_list.items():
            if _p is target:
                trial_destination_list[_p] = [
                    (o, int(s * 0.5) if o != self.player else s, t, x, y, bx, by)
                    for o, s, t, x, y, bx, by in _entries
                ]
            else:
                trial_destination_list[_p] = list(_entries)
        trial_destination_list.setdefault(target, [])
        battle_won = False
        second_enemy_arrival = None
        # Nếu đang attack planet không thuộc mình, cần chú ý fleet enemy khác có thể tới target trước/sau.
        if target.owner != self.player:
            for owner, _, t, _, _, _, _ in self.destination_list.get(target, []):
                if owner != self.player and owner != target.owner:
                    turn = math.ceil(t)
                    if second_enemy_arrival is None or turn < second_enemy_arrival:
                        second_enemy_arrival = turn
        for neighbor, _ in possible_origins:
            if neighbor.ships == 0:
                continue
            ships_to_send = int(neighbor.ships)
            # Kiểm tra source planet hiện có đang an toàn không trước khi rút quân.
            baseline_owner, _ = self.simulate_planet_timeline(neighbor, self.destination_list)
            not_doomed = baseline_owner == self.player
            if not_doomed:
                # Copy destination_list để thử tình huống xấu mà không làm hỏng mô phỏng chính.
                worst_case_dl = {k: list(v) for k, v in self.destination_list.items()}
                worst_case_dl.setdefault(neighbor, [])
                half_pressure = 0
                for attacker, _ in self.inbound_edges.get(neighbor, []):
                    if attacker.owner == self.player or attacker.owner == -1 or attacker.ships == 0:
                        continue
                    _, ax, ay, atk_travel = self.intercept_planet(attacker.x, attacker.y, neighbor, attacker.ships)
                    if not math.isfinite(atk_travel):
                        continue
                    half_ships = max(1, int(attacker.ships * 0.5))
                    worst_case_dl[neighbor].append((attacker.owner, half_ships, atk_travel, attacker.x, attacker.y, ax, ay))
                    half_pressure += half_ships
                # Tạm đặt ships của source về 0 để test nếu rút hết quân thì source có mất không.
                saved_ships = neighbor.ships
                neighbor.ships = 0
                exposed_owner, _ = self.simulate_planet_timeline(neighbor, worst_case_dl)
                neighbor.ships = saved_ships
                if exposed_owner != self.player:
                    if target.production <= neighbor.production:
                        continue
                else:
                    ships_to_send = max(0, int(neighbor.ships) - half_pressure)
                    if ships_to_send == 0:
                        continue
            angle, ix, iy, travel = self.intercept_planet(neighbor.x, neighbor.y, target, ships_to_send)
            if not math.isfinite(travel):
                continue
            # Kiểm tra lại: bắn theo angle này có thật sự trúng target đầu tiên không.
            # Nếu trúng planet khác hoặc đi qua mặt trời thì bỏ phương án.
            if self.first_planet_hit(neighbor.x, neighbor.y, angle, ships_to_send, neighbor) is not target:
                continue
            if second_enemy_arrival is not None and math.ceil(travel) <= second_enemy_arrival + 1:
                continue
            # Thêm fleet của mình vào mô phỏng target, rồi kiểm tra target có đổi sang mình không.
            trial_destination_list[target].append((self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy))
            fleet_orders.append([neighbor.id, angle, ships_to_send])
            intercepts.append((ix, iy, travel))
            trial_end_owner, _ = self.simulate_planet_timeline(target, trial_destination_list)
            if trial_end_owner == self.player:
                # Conditional 2-ply: chỉ kích hoạt multi-front (từ 00066)
                if self._multifront and self._enemy_counter_wins(target, trial_destination_list):
                    continue  # enemy có thể retake — cần thêm ships
                battle_won = True
                break
        return fleet_orders, intercepts, battle_won

    def evaluate_move_orders(self):
        """
        Duyệt tất cả target và chọn phương án attack/defend tốt nhất ở thời điểm hiện tại.

        Pipeline:
        - Với planet của mình: chỉ xét nếu nó đang bị đe dọa và cần defend.
        - Với planet neutral/enemy: xét nếu nó chưa chắc thuộc về mình.
        - Gọi evaluate_frontline_strategy để tìm fleet đủ thắng.
        - Chấm điểm bằng ROI, urgency bonus, FFA bonus, source exposure penalty.

        Trả về:
        - best_move_orders = (target, value, fleet_orders, intercepts).
        """
        """
        NÂNG CẤP: Scoring dùng ROI-based value thay vì flat production.
        Thêm defense urgency bonus cho planet mình đang bị tấn công.
        Trong FFA conflict window: tạm thời ưu tiên planet enemy đang bị attack.
        """
        # best_move_orders lưu phương án tốt nhất tìm được trong turn hiện tại.
        # Format: (target, value, fleet_orders, intercepts)
        best_move_orders = (None, -65535, [], [])
        # Xét target production cao trước, vì production là nguồn tăng quân lâu dài.
        for target in sorted(self.planets, key=lambda p: p.production, reverse=True):
            if not bool(self.inbound_edges.get(target)):
                continue

            if target.owner == self.player:
                # Defend path: planet mình đang bị đe dọa
                if not bool(self.destination_list.get(target)):
                    continue
                # Kiểm tra target có đang sắp mất không. Nếu có thì cần defend.
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue
                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
                if not battle_won:
                    continue
                penalty = self._source_exposure_penalty(fleet_orders) if self._multifront else 0

                # NÂNG CẤP: ROI score + defense urgency bonus
                roi = self._compute_roi_score(target, fleet_orders)
                urgency = self._defense_urgency_bonus(target)
                value = roi + urgency - penalty

                _, best_value, best_orders, _ = best_move_orders
                if value > best_value or (value == best_value and len(fleet_orders) < len(best_orders)):
                    best_move_orders = (target, value, fleet_orders, intercepts)

            else:
                # Attack path: planet neutral hoặc enemy
                # Kiểm tra target có đang sắp mất không. Nếu có thì cần defend.
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue

                # NÂNG CẤP: FFA exploit — nếu target enemy đang bị enemy khác tấn công,
                # đây là cơ hội tốt — tăng giá trị target lên
                ffa_bonus = 0
                if self._inter_enemy_conflict and target.owner not in (-1, self.player):
                    for dest, arrivals in self.destination_list.items():
                        if dest is target:
                            for owner, _, _, _, _, _, _ in arrivals:
                                if owner not in (-1, self.player) and owner != target.owner:
                                    ffa_bonus = target.production  # bonus bằng production
                                    break

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
                if not battle_won:
                    continue
                penalty = self._source_exposure_penalty(fleet_orders) if self._multifront else 0

                # NÂNG CẤP: ROI score thay vì flat production
                roi = self._compute_roi_score(target, fleet_orders)
                # Neutral planet bị trừ nhẹ để agent không quá ham neutral nếu enemy target cũng tốt.
                neutral_penalty = 1 if target.owner == -1 else 0
                value = roi - neutral_penalty - penalty + ffa_bonus

                _, best_value, best_orders, _ = best_move_orders
                if value > best_value or (value == best_value and len(fleet_orders) < len(best_orders)):
                    best_move_orders = (target, value, fleet_orders, intercepts)

        return best_move_orders

    def send_reinforcements(self):
        """
        Gửi quân tiếp viện từ hậu phương tới reinforcement_target.

        Cách làm:
        - Duyệt planet của mình.
        - Nếu planet có reinforcement_target và đủ quân sau khi giữ GARRISON_SIZE, gửi quân đi.
        - Nếu target đang urgent/sắp mất, giảm ngưỡng gửi để cứu kịp thời.

        Vai trò:
        - Duy trì dòng quân ra frontline.
        - Tránh để hậu phương tích quá nhiều quân không tham chiến.
        """
        """
        NÂNG CẤP: Urgent reinforcement — nếu frontline target sắp bị mất,
        reinforce ngay dù chưa đủ ngưỡng REINFORCEMENT_SIZE.
        """
        orders = []
        for p in self.owned_planets:
            if p.reinforcement_target is None:
                continue

            target = p.reinforcement_target

            # Kiểm tra urgency của target
            # Kiểm tra target reinforcement có đang sắp mất không. Nếu có thì cho phép gửi sớm hơn.
            end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
            is_urgent = (end_owner != self.player)

            # Threshold: urgent thì chỉ cần đủ garrison, bình thường cần full
            if is_urgent:
                min_threshold = self.GARRISON_SIZE + 1
            else:
                min_threshold = self.REINFORCEMENT_SIZE + self.GARRISON_SIZE

            if p.ships < min_threshold:
                continue

            # Nếu không urgent: không reinforce nếu đang bị đe dọa (giữ nguyên 00081 logic)
            if not is_urgent:
                if any(src.owner != self.player for src, _ in self.inbound_edges.get(p, [])):
                    continue

            # Gửi phần quân dư, giữ lại GARRISON_SIZE để planet nguồn không trống.
            ships = int(p.ships - self.GARRISON_SIZE)
            if ships <= 0:
                continue

            angle, ix, iy, travel = self.intercept_planet(p.x, p.y, target, ships)
            if not math.isfinite(travel):
                continue
            orders.append([p.id, angle, ships])
        return orders

    def commit_move_orders(self, move):
        """
        Cập nhật mô phỏng nội bộ sau khi đã chọn một move.

        Lưu ý:
        - Hàm này không gửi lệnh thật ra môi trường.
        - Nó chỉ trừ ships trong object nội bộ và thêm fleet giả định vào destination_list.

        Vai trò:
        - Khi vòng while tiếp tục chọn move tiếp theo, agent biết quân đã được dùng rồi, tránh gửi quá tay.
        """
        # move có dạng (target, value, fleet_orders, intercepts).
        target, _, fleet_orders, intercepts = move
        for (from_id, _, ships), (ix, iy, travel) in zip(fleet_orders, intercepts):
            src = next((p for p in self.planets if p.id == from_id), None)
            if src is None:
                continue
            # Trừ quân trong trạng thái nội bộ để các lần chọn move sau biết source đã dùng quân.
            src.ships = max(0, src.ships - ships)
            # Thêm fleet vừa chọn vào destination_list nội bộ để mô phỏng các target sau chính xác hơn.
            self.destination_list.setdefault(target, [])
            self.destination_list[target].append((self.player, ships, travel, src.x, src.y, ix, iy))

    def early_game_compute_travel_turns(self, source_id, target, fleet_size, launch_turn):
        """
        Tính thời gian bay trong early-game planner nếu launch ở một turn tương lai.

        Khác với intercept bình thường:
        - Nếu source là planet quay, vị trí source tại launch_turn cũng phải được dự đoán.

        Vai trò:
        - Dùng để đánh giá khi nào có thể chiếm một neutral planet.
        """
        src = next(p for p in self.planets if p.id == source_id)
        orb = self.orbital_info.get(src)
        if orb is not None:
            cx = cy = CENTER
            r, ia = orb
            a = ia + self.angular_velocity * (launch_turn - 0.5)
            sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
        else:
            sx, sy = src.x, src.y
        _, _, _, travel = self.intercept_planet(sx, sy, target, fleet_size)
        return travel

    def early_game_find_capture_turn(self, state, target):
        """
        Tìm turn sớm nhất mà trạng thái early-game giả lập có thể chiếm target.

        Cách làm:
        - Với mỗi source đang sở hữu trong state.owned.
        - Thử chờ 0, 1, 2, ... turn để tích thêm quân.
        - Nếu đủ quân và kịp tới trong horizon, cập nhật capture turn tốt nhất.
        """
        garrison_size = target.ships
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        best = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(self.EARLY_LOOK_AHEAD):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                if launch_turn >= horizon:
                    break
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= horizon:
                    best = min(best, arrival_turn)
                    break
        return best

    def early_game_assign_fleets(self, state, target, capture_turn):
        """
        Chọn source tốt nhất để gửi quân chiếm target đúng trước capture_turn.

        Trả về:
        - dict {source_id: (fleet_size, launch_turn, arrival_turn)}.

        Vai trò:
        - Sau khi biết target có thể bị chiếm ở turn nào, hàm này xác định gửi từ đâu và gửi bao nhiêu.
        """
        garrison_size = target.ships
        best_source = None
        best_entry = None
        best_arrival = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(capture_turn - state.turn):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= capture_turn and arrival_turn < best_arrival:
                    best_arrival = arrival_turn
                    best_source = source
                    best_entry = (fleet_size, launch_turn, arrival_turn)
                break
        if best_source is None:
            return {}
        return {best_source: best_entry}

    def early_game_advance(self, state, from_turn, to_turn):
        """
        Tiến trạng thái early-game giả lập từ from_turn tới to_turn.

        Trong mỗi turn:
        - Xử lý các fleet tới nơi.
        - Nếu fleet là capture thì thêm planet vào owned.
        - Cộng production cho tất cả planet đang sở hữu.

        Vai trò:
        - Đây là hàm cập nhật động lực học cho early-game simulation.
        """
        for current_turn in range(from_turn + 1, to_turn + 1):
            for fleet in list(state.fleets):
                if fleet.arrival_turn == current_turn:
                    if fleet.is_capture:
                        state.garrison[fleet.destination_id] = fleet.garrison_on_arrival
                        state.owned.add(fleet.destination_id)
                        if fleet.destination_id not in state.production:
                            state.production[fleet.destination_id] = self.early_game_production(fleet.destination_id)
                    else:
                        state.garrison[fleet.destination_id] += fleet.garrison_on_arrival
                    state.fleets.remove(fleet)
            for planet_id in state.owned:
                state.garrison[planet_id] += state.production[planet_id]
        return state

    def early_game_execute_attack(self, state, target, fleet_assignment, capture_turn):
        """
        Áp dụng một kế hoạch chiếm target vào EarlyGameState giả lập.

        Cách làm:
        - Advance tới thời điểm launch.
        - Trừ quân ở source.
        - Thêm EarlyGameFleet sẽ tới target ở capture_turn.
        - Advance tới capture_turn để cập nhật kết quả.
        """
        garrison_size = target.ships
        total_fleet = sum(fs for fs, _, _ in fleet_assignment.values())
        current_turn = state.turn
        for source, (fleet_size, launch_turn, _) in sorted(fleet_assignment.items(), key=lambda se: se[1][1]):
            state = self.early_game_advance(state, current_turn, launch_turn)
            current_turn = launch_turn
            state.garrison[source] -= fleet_size
        state.fleets.append(EarlyGameFleet(
            source_id=-1, destination_id=target.id, fleet_size=total_fleet,
            garrison_on_arrival=total_fleet - garrison_size, arrival_turn=capture_turn, is_capture=True,
        ))
        state = self.early_game_advance(state, current_turn, capture_turn)
        return state

    def early_game_score(self, state):
        """
        Chấm điểm một EarlyGameState.

        Điểm gồm:
        - Quân hiện có trên các planet owned.
        - Production dự kiến trong horizon còn lại.
        - Quân/fleet đang bay và lợi ích từ planet sẽ chiếm.

        Vai trò:
        - Hàm objective cho DFS early-game planner.
        """
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        total = 0
        for planet_id in state.owned:
            total += state.garrison[planet_id] + state.production[planet_id] * (horizon - state.turn)
        for fleet in state.fleets:
            total += fleet.garrison_on_arrival
            if fleet.is_capture:
                total += self.early_game_production(fleet.destination_id) * max(0, horizon - fleet.arrival_turn)
        return total

    def early_game_production(self, planet_id):
        """
        Trả về production của planet theo id.

        Dùng khi một planet mới được chiếm trong mô phỏng và cần thêm nó vào bảng production.
        """
        p = next((pl for pl in self.planets if pl.id == planet_id), None)
        return p.production if p else 0

    def run_early_game(self):
        """
        Chạy chiến lược riêng cho vài turn đầu trận.

        Mục tiêu:
        - Chiếm nhanh neutral planet có lợi ích dương.
        - Dùng DFS + branch-and-bound để thử nhiều thứ tự chiếm planet.

        Output:
        - Danh sách move thật cần thực hiện ngay ở turn hiện tại.

        Vai trò:
        - Early game quyết định tốc độ mở rộng ban đầu, ảnh hưởng lớn tới toàn trận.
        """
        # Tập id planet đang sở hữu ở trạng thái thật hiện tại.
        owned_ids = {p.id for p in self.owned_planets}
        # Chỉ xét neutral planet có thể đi tới từ planet của mình theo graph gần-kề.
        neutral_candidates = [
            p for p in self.planets
            if p.owner == -1 and any(src.id in owned_ids for src, _ in self.inbound_edges.get(p, []))
        ]
        in_flight = []
        for dest_planet, arrivals in self.destination_list.items():
            for owner, ships, t, _, _, _, _ in arrivals:
                if owner != self.player:
                    continue
                arrival = self.scene_step + math.ceil(t)
                is_cap = dest_planet.owner != self.player
                surplus = ships - dest_planet.ships
                in_flight.append(EarlyGameFleet(
                    source_id=-1, destination_id=dest_planet.id, fleet_size=int(ships),
                    garrison_on_arrival=int(surplus) if is_cap else int(ships),
                    arrival_turn=arrival, is_capture=is_cap,
                ))
        # Tạo snapshot early game ban đầu để DFS thử các chuỗi chiếm planet.
        initial_state = EarlyGameState(
            turn=self.scene_step,
            garrison={p.id: float(p.ships) for p in self.owned_planets},
            production={p.id: p.production for p in self.owned_planets},
            owned=owned_ids.copy(),
            fleets=in_flight,
        )
        # Hàm gain nhanh để lọc/sắp xếp neutral candidate trước khi DFS.
        def initial_gain(planet):
            ct = self.early_game_find_capture_turn(initial_state, planet)
            horizon = initial_state.turn + self.EARLY_LOOK_AHEAD
            return planet.production * (horizon - ct) - planet.ships if math.isfinite(ct) else -math.inf
        candidates = sorted(neutral_candidates, key=initial_gain, reverse=True)
        candidates = [p for p in candidates if initial_gain(p) > 0]
        if not candidates:
            return []
        best = [self.early_game_score(initial_state), []]
        # upper_bound dùng để cắt nhánh DFS: nếu kịch bản tốt nhất cũng không vượt best thì bỏ.
        def upper_bound(state, remaining):
            horizon = state.turn + self.EARLY_LOOK_AHEAD
            bound = self.early_game_score(state)
            for planet in remaining:
                ct = self.early_game_find_capture_turn(state, planet)
                gain = planet.production * (horizon - ct) - planet.ships
                if gain > 0:
                    bound += gain
            return bound
        # DFS thử từng thứ tự chiếm neutral planet.
        # sequence lưu chuỗi kế hoạch đã chọn.
        def dfs(state, remaining, sequence):
            current_score = self.early_game_score(state)
            if current_score > best[0]:
                best[0] = current_score
                best[1] = list(sequence)
            if upper_bound(state, remaining) <= best[0]:
                return
            already_targeted = {f.destination_id for f in state.fleets if f.is_capture}
            for index, planet in enumerate(remaining):
                if planet.id in already_targeted:
                    continue
                horizon = state.turn + self.EARLY_LOOK_AHEAD
                ct = self.early_game_find_capture_turn(state, planet)
                if not math.isfinite(ct):
                    continue
                if planet.production * (horizon - ct) - planet.ships <= 0:
                    continue
                fleet_assignment = self.early_game_assign_fleets(state, planet, ct)
                if not fleet_assignment:
                    continue
                next_state = self.early_game_execute_attack(copy.deepcopy(state), planet, fleet_assignment, ct)
                dfs(next_state, remaining[:index] + remaining[index + 1:], sequence + [(planet, fleet_assignment, ct)])
        dfs(initial_state, candidates, [])
        _, best_sequence = best
        if not best_sequence:
            return []
        moves = []
        for target_planet, fleet_assignment, _ in best_sequence:
            for source_id, (fleet_size, launch_turn, _) in fleet_assignment.items():
                if launch_turn != self.scene_step:
                    continue
                src = next((p for p in self.planets if p.id == source_id), None)
                if src is None:
                    continue
                angle, _, _, travel = self.intercept_planet(src.x, src.y, target_planet, fleet_size)
                if not math.isfinite(travel):
                    continue
                hit = self.first_planet_hit(src.x, src.y, angle, fleet_size, src)
                if hit is not target_planet:
                    continue
                moves.append([source_id, angle, fleet_size])
        return moves

    def main(self, obs):
        """
        Hàm chính xử lý một observation và trả về danh sách hành động.

        Đây là pipeline tổng:
        1. Đọc obs và tạo object Planet/Fleet.
        2. Thiết lập mode multi-front/endgame.
        3. Xây orbital_info, proximity_graph, destination_list.
        4. Nếu early game thì chạy run_early_game.
        5. Nếu mid/late game thì chọn attack/defense bằng evaluate_move_orders.
        6. Gửi reinforcement.
        7. Return moves.
        """
        # Đọc id người chơi hiện tại từ observation.
        self.player = obs['player']
        self.scene_step = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        # Comet planet bị loại khỏi danh sách planet thường vì có hành vi đặc biệt/không nên xử lý như target bình thường.
        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [Planet(*p) for p in obs['planets']]
        self.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        self.owned_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner != self.player]
        self.fleets = [Fleet(*f) for f in obs['fleets']]

        if not self.enemy_planets:
            return []

        # Multi-front detection
        # Đếm số enemy còn sống để bật/tắt logic multi-front.
        enemy_count = self._count_enemy_owners()
        self._multifront = enemy_count >= 2

        # NÂNG CẤP: Adaptive endgame threshold
        # Kích hoạt endgame sớm hơn nếu đang thắng lớn (ownership > 65%)
        # Tỉ lệ planet đang sở hữu, dùng để phát hiện đang thắng lớn và kích hoạt endgame sớm.
        ownership_ratio = len(self.owned_planets) / max(1, len(self.planets))
        self.GARRISON_SIZE = 8
        self.REINFORCEMENT_SIZE = 17
        self.MAX_DISTANCE = 38

        # Endgame mode: giảm giữ quân, tăng tầm đánh để chiếm thêm planet trước khi hết game.
        if self.scene_step >= 440 or (ownership_ratio >= 0.65 and self.scene_step >= 350):
            self.GARRISON_SIZE = 1
            self.REINFORCEMENT_SIZE = 2
            self.MAX_DISTANCE = 55

        self.build_orbital_info(obs.get('initial_planets', []))
        self.build_proximity_graph()
        self.build_destination_list()

        # NÂNG CẤP: Detect inter-enemy conflict cho FFA exploit
        self._inter_enemy_conflict = self._multifront and self._detect_inter_enemy_conflict()

        if self.scene_step < self.EARLY_ROUNDS:
            return self.run_early_game()

        self.build_reinforcement_targets()

        moves = []
        # Lặp chọn nhiều move trong cùng một turn.
        # Sau mỗi move, commit_move_orders cập nhật mô phỏng nội bộ rồi mới chọn tiếp.
        while True:
            move_orders = self.evaluate_move_orders()
            target_planet, _, fleet_orders, _ = move_orders
            if target_planet is None:
                break
            self.commit_move_orders(move_orders)
            moves.extend(fleet_orders)

        moves.extend(self.send_reinforcements())
        return moves


def agent(obs):
    """
    Entry point bắt buộc của Kaggle Orbit Wars.

    Kaggle sẽ gọi agent(obs) ở mỗi turn.
    Hàm này tạo một HeuristicAgent mới, chạy main(obs), rồi trả về danh sách move.

    try/except giúp submission không bị crash nếu có lỗi bất ngờ; khi lỗi thì agent không làm gì.
    """
    try:
        return HeuristicAgent().main(obs)
    except Exception:
        return []
