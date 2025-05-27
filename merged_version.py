import random
import re
from collections import defaultdict
from game.avalon_game_helper import (
    askLLM, read_public_lib,
    read_private_lib, write_into_private
)

MAP_SIZE = 9

#### 辅助函数
def calculate_hearing_range(role: str) -> int:
    """根据角色计算听力范围"""
    if role in {"Knight", "Oberon"}:
        return 5  # 5x5范围
    else:
        return 3  # 3x3范围

def is_in_hearing_range(speaker_pos: tuple, listener_pos: tuple, hearing_range: int) -> bool:
    """检查说话者是否在听者的听力范围内"""
    if not speaker_pos or not listener_pos:
        return False
    sx, sy = speaker_pos
    lx, ly = listener_pos
    half_range = (hearing_range - 1) // 2
    return abs(sx - lx) <= half_range and abs(sy - ly) <= half_range


# 待修复
def analyze_message_content(msg: str, role: str, trusted_evil: set, trusted_good: set) -> dict:
    """分析消息内容并返回信任度变化"""
    changes = defaultdict(int)
    msg_lower = msg.lower()
    mentioned_players = [int(w[1:]) for w in msg_lower.split() if w.startswith("p") and w[1:].isdigit()]
    
    # 分析指控和支持
    if any(word in msg_lower for word in ["evil", "spy", "fail", "bad", "suspicious"]):
        for p in mentioned_players:
            if role == "Merlin":
                if p in trusted_evil and "not" not in msg_lower:
                    changes['speaker_trust'] += 2  # 准确指控邪恶
                elif p not in trusted_evil and "not" not in msg_lower:
                    changes['speaker_suspicion'] += 2  # 错误指控好人
            elif role in {"Morgana", "Assassin"}:
                if p not in trusted_evil and "not" not in msg_lower:
                    changes['speaker_trust'] += 1  # 指控好人有利于邪恶
    
    if any(word in msg_lower for word in ["trust", "good", "approve", "loyal", "support"]):
        for p in mentioned_players:
            if role == "Merlin":
                if p in trusted_evil and "not" not in msg_lower:
                    changes['speaker_suspicion'] += 2  # 支持邪恶
                elif p not in trusted_evil and "not" not in msg_lower:
                    changes['speaker_trust'] += 1  # 支持好人
    
    return changes

def calculate_mission_performance_score(player_id: int, team_history: list, mission_results: list) -> float:
    """计算玩家在任务中的表现得分"""
    score = 0
    participation_count = 0
    
    for i, (team, result) in enumerate(zip(team_history, mission_results)):
        if player_id in team:
            participation_count += 1
            if result:  # 任务成功
                score += 2
            else:  # 任务失败
                score -= 1
    
    return score / max(participation_count, 1)

def get_merlin_candidates_from_messages(memory: set, known_roles: dict) -> list:
    """从消息中分析梅林候选人"""
    candidates = []
    merlin_indicators = [
        "梅林", "知道身份", "信息量", "观察到", "从行为来看",
        "太合理", "隐藏", "不该这么说"
    ]
    
    for speaker, msg in memory:
        if any(indicator in msg for indicator in merlin_indicators):
            candidates.append(speaker)
    
    # 结合派西维尔的视野信息
    if 'Merlin_Morgana_candidates' in known_roles:
        candidates.extend(known_roles['Merlin_Morgana_candidates'])
    
    return list(set(candidates))

def get_nearby_players(self_pos: tuple, player_positions: dict, hearing_range: int) -> list:
    """获取听力范围内的玩家"""
    nearby = []
    if not self_pos:
        return nearby
    
    for player_id, pos in player_positions.items():
        if pos and is_in_hearing_range(pos, self_pos, hearing_range):
            nearby.append(player_id)
    
    return nearby

def random_walk(current_pos: tuple, player_positions: dict, game_map, max_steps: int = 3) -> tuple:
    """通用随机移动函数"""
    if not current_pos or not game_map:
        return tuple()
    
    x, y = current_pos
    others_pos = set(pos for pid, pos in player_positions.items() if pos != current_pos)
    
    # 检查是否被完全包围
    directions = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]
    blocked_count = 0
    for _, dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if (new_x < 0 or new_x >= MAP_SIZE or new_y < 0 or new_y >= MAP_SIZE or 
            (new_x, new_y) in others_pos):
            blocked_count += 1
    
    if blocked_count == 4:  # 完全被包围
        return tuple()
    
    # 随机移动
    total_steps = random.randint(0, max_steps)
    valid_moves = []
    current_x, current_y = x, y
    
    for _ in range(total_steps):
        possible_moves = []
        for direction, dx, dy in directions:
            new_x, new_y = current_x + dx, current_y + dy
            if (0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE and 
                (new_x, new_y) not in others_pos):
                possible_moves.append((direction, new_x, new_y))
        
        if not possible_moves:
            break
            
        direction, new_x, new_y = random.choice(possible_moves)
        valid_moves.append(direction)
        current_x, current_y = new_x, new_y
        others_pos.add((new_x, new_y))  # 更新位置避免重复占用
    
    return tuple(valid_moves)

def strategic_walk_towards(current_pos: tuple, target_positions: list, player_positions: dict, 
                          game_map, max_steps: int = 3) -> tuple:
    """策略性移动：朝向目标位置"""
    if not current_pos or not target_positions or not game_map:
        return random_walk(current_pos, player_positions, game_map, max_steps)
    
    x, y = current_pos
    others_pos = set(pos for pid, pos in player_positions.items() if pos != current_pos)
    
    # 找到最近的目标
    min_distance = float('inf')
    best_target = None
    for target_pos in target_positions:
        if target_pos and target_pos != current_pos:
            distance = abs(x - target_pos[0]) + abs(y - target_pos[1])
            if distance < min_distance:
                min_distance = distance
                best_target = target_pos
    
    if not best_target:
        return random_walk(current_pos, player_positions, game_map, max_steps)
    
    target_x, target_y = best_target
    valid_moves = []
    current_x, current_y = x, y
    
    for _ in range(max_steps):
        # 计算朝向目标的最佳方向
        dx = target_x - current_x
        dy = target_y - current_y
        
        possible_moves = []
        if dx > 0 and current_x + 1 < MAP_SIZE and (current_x + 1, current_y) not in others_pos:
            possible_moves.append(("Down", current_x + 1, current_y))
        elif dx < 0 and current_x - 1 >= 0 and (current_x - 1, current_y) not in others_pos:
            possible_moves.append(("Up", current_x - 1, current_y))
        
        if dy > 0 and current_y + 1 < MAP_SIZE and (current_x, current_y + 1) not in others_pos:
            possible_moves.append(("Right", current_x, current_y + 1))
        elif dy < 0 and current_y - 1 >= 0 and (current_x, current_y - 1) not in others_pos:
            possible_moves.append(("Left", current_x, current_y - 1))
        
        # 如果没有朝向目标的可行移动，随机选择
        if not possible_moves:
            directions = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]
            for direction, dx_rand, dy_rand in directions:
                new_x, new_y = current_x + dx_rand, current_y + dy_rand
                if (0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE and 
                    (new_x, new_y) not in others_pos):
                    possible_moves.append((direction, new_x, new_y))
        
        if not possible_moves:
            break
            
        direction, new_x, new_y = random.choice(possible_moves)
        valid_moves.append(direction)
        current_x, current_y = new_x, new_y
        others_pos.add((new_x, new_y))
        
        # 如果已经到达目标附近，停止移动
        if abs(current_x - target_x) + abs(current_y - target_y) <= 1:
            break
    
    return tuple(valid_moves)

#### 角色策略类
# 基类
class RoleStrategy:
    def __init__(self, player):
        self.player = player
    
    def decide_mission_member(self, team_size: int) -> list:
        """默认组队策略"""
        return random.sample(self.player.players, team_size)
    
    def say(self) -> str:
        """默认发言策略"""
        return "我在观察局势。"
    
    def walk(self) -> tuple:
        """默认移动策略：随机移动"""
        current_pos = self.player.player_positions.get(self.player.index)
        return random_walk(current_pos, self.player.player_positions, self.player.map)
    
    def mission_vote1(self) -> bool:
        """默认投票策略"""
        return True
    
    def mission_vote2(self) -> bool:
        """默认任务执行策略：好人成功，邪恶破坏"""
        if self.player.role in {"Morgana", "Assassin", "Oberon"}:
            # 邪恶角色破坏任务
            current_round = len(self.player.mission_results) + 1
            if current_round == 1:
                return False if random.random() < 0.8 else True
            return False
        else:
            # 好人角色执行成功
            return True

    def assassinate(self, player) -> int:  # Target player index for assassination
        # Most roles don't assassinate. Assassin role will override this.
        return -1

# 梅林策略
class MerlinRoleStrategy(RoleStrategy):
    def say(self, player) -> str:
        self_pos = player.player_positions.get(player.index)
        if not self_pos:
            return "我还没有确定位置，暂时无法发言。"

        if player.round_num <= 2:  # Early game, very subtle
            if len(player.trusted_evil) > 0:
                evil_player = random.choice(list(player.trusted_evil))
                evil_player_pos = player.player_positions.get(evil_player)
                # Only hint if evil player's position is known and they are NOT in Merlin's hearing range
                if evil_player_pos and not player._is_in_range(evil_player_pos, self_pos, player.hearing_range):
                    return f"我觉得P{evil_player}今天的发言有点奇怪，大家可以多观察一下他/她。"
            return "大家要多沟通，找出队伍中的不稳定因素。"
        else:  # Mid-late game, slightly more direct but still cautious
            if len(player.trusted_evil) > 0:
                evil_player = random.choice(list(player.trusted_evil))
                evil_player_pos = player.player_positions.get(evil_player)
                # Even in later game, prefer to hint about players not directly listening to Merlin, if possible
                if evil_player_pos and not player._is_in_range(evil_player_pos, self_pos, player.hearing_range):
                    return f"我们必须警惕P{evil_player}，他/她可能不是好人。"
                else:  # If evil is in range, be more generic
                    return "邪恶阵营的成员正在试图混淆视听，请大家警惕！"
            if len(player.trusted_good) > 0:  # trusted_good is populated by Merlin based on observations
                good_player = random.choice(list(player.trusted_good))
                good_player_pos = player.player_positions.get(good_player)
                if good_player_pos and not player._is_in_range(good_player_pos, self_pos, player.hearing_range):
                    return f"P{good_player}一直是值得信任的，我支持他/她。"
                else:
                    return "我们应该信任那些为团队付出的人。"
        return "我们需要一个能通过任务的队伍。请大家谨慎选择。"

    def decide_mission_member(self, player, team_size: int) -> list:
        candidates = []
        if player.index not in candidates:
            candidates.append(player.index)

        # Get players Merlin knows are not evil
        safe_players = [p for p in player.players if p not in player.trusted_evil]

        # Prioritize players Merlin trusts (in player.trusted_good) or has low suspicion of among safe players
        # Sort by: 1. Is in trusted_good (True first), 2. Suspicion level (lower first)
        sorted_safe_candidates = sorted(safe_players, key=lambda p_id: (
        not (p_id in player.trusted_good), player.suspicion_level[p_id]))

        for p_id in sorted_safe_candidates:
            if p_id not in candidates and len(candidates) < team_size:
                candidates.append(p_id)

        # If team is still not full, add remaining safe players (excluding self if already added)
        remaining_safe_to_add = [p_id for p_id in safe_players if p_id not in candidates]
        while len(candidates) < team_size and remaining_safe_to_add:
            candidates.append(remaining_safe_to_add.pop(0))  # Add from the start of the sorted list

        # Fallback: if still not enough (e.g. too many evil players), fill with any available player not already on team
        # This part should ideally not be reached if team_size is reasonable for # of good players
        all_player_ids = [p_id for p_id in player.players]
        while len(candidates) < team_size:
            available_players = [p_id for p_id in all_player_ids if p_id not in candidates]
            if not available_players: break
            # Add players Merlin is less certain about but are not known evil
            player_to_add = random.choice(available_players)  # Or pick from less suspicious non-evil
            candidates.append(player_to_add)

        return sorted(list(set(candidates))[:team_size])

    def mission_vote1(self, player) -> bool:  # Team approval
        current_team = player.team_history[-1] if player.team_history else []
        if not current_team: return random.choice([True, False])

        if any(p in player.trusted_evil for p in current_team):
            return False  # Reject if known evil is on the team

        # If a seemingly good team, approve.
        # Calculate suspicion of team members not known to be good by Merlin
        suspicion_on_team = sum(player.suspicion_level[p] for p in current_team if
                                p not in player.trusted_good and p not in player.trusted_evil)

        # Heuristic: if high suspicion and early/mid game, might reject unless vote track is late
        # (More complex vote track logic would require more state from Player class)
        if suspicion_on_team > (len(current_team) * 1.5) and player.round_num < 4:
            # Check if it's a late proposal in the round (e.g. 4th or 5th vote)
            # This info isn't directly available in a simple way, so Merlin is generally cautious
            return False

        return True  # Approve if no known evil and not overly suspicious

    # assassinate is inherited from RoleStrategy (returns -1 for Merlin)
    def mission_vote2(self) -> bool:
        """梅林任务执行：永远成功"""
        return True

    def walk(self, player) -> tuple:
        """
        Generic walking logic. Roles can override this if they have specific
        walking patterns, but typically this is common.
        The 'player' parameter is an instance of the main Player class.
        """
        origin_pos = player.player_positions.get(player.index)
        if not origin_pos:
            return tuple()  # Cannot walk if position is unknown

        x, y = origin_pos
        # Correctly get other players' current positions
        others_pos = [pos for pid, pos in player.player_positions.items()
                      if pid != player.index and pos is not None]

        total_step = random.randint(0, 3)

        # Check if completely surrounded by map edges or other players
        up_blocked = (x == 0 or (x - 1, y) in others_pos)
        down_blocked = (x == MAP_SIZE - 1 or (x + 1, y) in others_pos)  # Using global MAP_SIZE
        left_blocked = (y == 0 or (x, y - 1) in others_pos)
        right_blocked = (y == MAP_SIZE - 1 or (x, y + 1) in others_pos)

        if up_blocked and down_blocked and left_blocked and right_blocked:
            total_step = 0  # Player is stuck

        valid_moves = []
        current_x, current_y = x, y  # Simulate moves locally for this turn's plan

        for _ in range(total_step):
            possible_directions = []
            # Check based on current_x, current_y after previous potential step in this walk
            if current_x > 0 and (current_x - 1, current_y) not in others_pos:
                possible_directions.append("Up")
            if current_x < MAP_SIZE - 1 and (current_x + 1, current_y) not in others_pos:
                possible_directions.append("Down")
            if current_y > 0 and (current_x, current_y - 1) not in others_pos:
                possible_directions.append("Left")
            if current_y < MAP_SIZE - 1 and (current_x, current_y + 1) not in others_pos:
                possible_directions.append("Right")

            if not possible_directions:
                break  # No valid moves from current_x, current_y

            direction = random.choice(possible_directions)

            # Update current_x, current_y to reflect the chosen move for this step
            if direction == "Up":
                current_x -= 1
            elif direction == "Down":
                current_x += 1
            elif direction == "Left":
                current_y -= 1
            elif direction == "Right":
                current_y += 1

            valid_moves.append(direction)
        return tuple(valid_moves)
# 派西维尔策略
class PercivalRoleStrategy(RoleStrategy):
    def say(self, player) -> str:
        self_pos = player.player_positions.get(player.index)
        if not self_pos:
            return "我尚未确定方位。"

        if 'Merlin_Morgana_candidates' in player.known_roles and \
                len(player.known_roles['Merlin_Morgana_candidates']) == 2:
            candidates = player.known_roles['Merlin_Morgana_candidates']
            c1_pos = player.player_positions.get(candidates[0])
            c2_pos = player.player_positions.get(candidates[1])

            # Try to communicate to candidates if they are in range
            if c1_pos and player._is_in_range(c1_pos, self_pos, player.hearing_range) and \
                    c2_pos and player._is_in_range(c2_pos, self_pos, player.hearing_range):
                return f"P{candidates[0]}和P{candidates[1]}，你们中的一位是梅林，另一位是莫甘娜。请用你们的行动证明身份！"
            elif c1_pos and player._is_in_range(c1_pos, self_pos, player.hearing_range):
                return f"P{candidates[0]}，你的责任重大。我希望你能指引我们。"
            elif c2_pos and player._is_in_range(c2_pos, self_pos, player.hearing_range):
                return f"P{candidates[1]}，我正密切关注你。希望你是忠诚的一方。"
            else:  # If candidates are not in range, make a general statement for others
                return f"我已见到梅林与莫甘娜的幻象，他们是P{candidates[0]}和P{candidates[1]}。大家要仔细观察他们的行为！"
        return "寻找梅林是我的责任。我会尽力保护他，并揭露莫甘娜的伪装。"

    def decide_mission_member(self, player, team_size: int) -> list:
        candidates = []
        if player.index not in candidates:
            candidates.append(player.index)

        # Percival knows Merlin/Morgana candidates
        merlin_morgana_pair = player.known_roles.get('Merlin_Morgana_candidates', [])

        # Strategy: Try to include one of the Merlin/Morgana candidates to test them.
        # Prefer the one Percival suspects less (potential Merlin).
        if len(merlin_morgana_pair) == 2 and len(candidates) < team_size:
            # Sort candidates by suspicion level (lower is better for Percival's Merlin guess)
            sorted_mm_pair = sorted(merlin_morgana_pair, key=lambda p_id: player.suspicion_level[p_id])
            potential_merlin_on_team = sorted_mm_pair[0]
            if potential_merlin_on_team not in candidates:
                candidates.append(potential_merlin_on_team)

        # Fill with other trusted players (low suspicion, or in player.trusted_good if Percival populates this)
        # Exclude self and already added M/M candidate from this pool for now.
        other_players_to_consider = [p for p in player.players if p not in candidates]

        # Sort by: 1. Is in trusted_good (True first), 2. Suspicion level (lower first)
        # Percival might not have a robust 'trusted_good' like Merlin, relies more on suspicion.
        sorted_other_players = sorted(other_players_to_consider, key=lambda p_id: (
        not (p_id in player.trusted_good), player.suspicion_level[p_id]))

        for p_id in sorted_other_players:
            if len(candidates) < team_size:
                # Avoid putting both Merlin/Morgana candidates on the same team if Percival is leading
                # especially if the team is small or it's early game.
                if p_id in merlin_morgana_pair and any(mm_cand in candidates for mm_cand in merlin_morgana_pair):
                    if team_size <= 3 or player.round_num <= 2:  # Be cautious putting both on small/early teams
                        continue
                candidates.append(p_id)

        # Fallback fill if necessary
        all_player_ids = [p_id for p_id in player.players]
        while len(candidates) < team_size:
            available_players = [p for p in all_player_ids if p not in candidates]
            if not available_players: break
            # Add least suspicious from remaining
            player_to_add = sorted(available_players, key=lambda p_add: player.suspicion_level[p_add])[0]
            candidates.append(player_to_add)

        return sorted(list(set(candidates))[:team_size])

    def mission_vote1(self, player) -> bool:  # Team approval
        current_team = player.team_history[-1] if player.team_history else []
        if not current_team: return random.choice([True, False])

        merlin_morgana_pair = player.known_roles.get('Merlin_Morgana_candidates', [])

        # Count how many of the M/M pair are on the proposed team
        mm_on_team_count = sum(1 for p_id in current_team if p_id in merlin_morgana_pair)

        # If Percival is on the team, he's generally more likely to approve if it's not terrible
        if player.index in current_team:
            if mm_on_team_count == 1: return True  # Good for testing one candidate
            if mm_on_team_count == 0 and len(merlin_morgana_pair) == 2:  # Neither M/M on team, but Percival is.
                # Approve if team seems good otherwise
                suspicion_on_team_no_mm = sum(
                    player.suspicion_level[p] for p in current_team if p not in merlin_morgana_pair)
                if suspicion_on_team_no_mm < (
                        len(current_team) - (1 if player.index in current_team else 0)) * 0.5: return True  # Low sus

        # If one of the M/M candidates is on the team, Percival usually approves to see their mission play.
        if mm_on_team_count == 1:
            return True

        # If both M/M candidates are on the team, Percival might be cautious.
        if mm_on_team_count == 2:
            # Could be a good team if one is Merlin and other is good, or risky if one is Morgana.
            # Percival might reject if he suspects Morgana could easily fail it or confuse things.
            # Let's say reject if team is small and both are on it.
            if len(current_team) <= 3: return False
            return random.random() < 0.4  # Otherwise, less likely to approve both together

        # If no M/M candidates are on the team (and Percival isn't leading/on it to test them this way)
        # Percival might reject to try and get one of them onto a future team,
        # or approve if the team looks very trustworthy otherwise.
        if mm_on_team_count == 0 and len(merlin_morgana_pair) == 2:
            if random.random() < 0.6:  # Higher chance to reject to try and get M/M on a team
                return False

        # General good player logic: approve if low suspicion
        team_suspicion = sum(player.suspicion_level[p] for p in current_team)
        return team_suspicion < (len(current_team) * 0.75)  # Threshold for approval

    # assassinate is inherited from RoleStrategy (returns -1 for Percival)
    def mission_vote2(self) -> bool:
        """派西维尔任务执行：永远成功"""
        return True

    def walk(self) -> tuple:
        """派西维尔移动策略：尝试靠近梅林/莫甘娜候选人观察"""
        current_pos = self.player.player_positions.get(self.player.index)
        if not current_pos:
            return tuple()
        
        # 获取梅林/莫甘娜候选人位置
        candidates = self.player.known_roles.get('Merlin_Morgana_candidates', [])
        candidate_positions = [self.player.player_positions.get(p) for p in candidates 
                              if p in self.player.player_positions]
        candidate_positions = [pos for pos in candidate_positions if pos]
        
        if candidate_positions:
            return strategic_walk_towards(current_pos, candidate_positions, 
                                        self.player.player_positions, self.player.map, 2)
        else:
            return random_walk(current_pos, self.player.player_positions, self.player.map, 2)

# 骑士策略
class KnightStrategy(RoleStrategy):
    def decide_mission_member(self, team_size: int) -> list:
        """骑士组队：选择信任的玩家"""
        candidates = [self.player.index]
        trusted_players = sorted(self.player.players, key=lambda p: self.player.suspicion_level[p])
        
        for p in trusted_players:
            if len(candidates) < team_size and p not in candidates:
                candidates.append(p)
        
        return sorted(candidates[:team_size])
    
    def say(self) -> str:
        """骑士发言：积极识别可疑玩家"""
        # 利用更广的听力范围获得的信息
        nearby_players = get_nearby_players(
            self.player.player_positions.get(self.player.index),
            self.player.player_positions,
            self.player.hearing_range
        )
        
        suspicious = [p for p in nearby_players if self.player.suspicion_level[p] > 2]
        if suspicious:
            target = random.choice(suspicious)
            return f"根据我的观察，P{target}的行为模式值得怀疑。"
        
        return "我会继续观察大家的行为。"
    
    def mission_vote1(self) -> bool:
        """骑士投票：基于怀疑度判断"""
        current_team = self.player.team_history[-1][1] if self.player.team_history else []
        
        if self.player.index in current_team:
            suspicion_score = sum(self.player.suspicion_level[p] for p in current_team)
            return suspicion_score < len(current_team) * 2
        
        suspicion_score = sum(self.player.suspicion_level[p] for p in current_team)
        return suspicion_score <= len(current_team) * 1.0

    def mission_vote2(self) -> bool:
        """骑士任务执行：永远成功"""
        return True

    def walk(self) -> tuple:
        """骑士移动策略：由于听力范围大，移动较少以保持覆盖"""
        current_pos = self.player.player_positions.get(self.player.index)
        return random_walk(current_pos, self.player.player_positions, self.player.map, 2)

# 莫甘娜策略
class MorganaStrategy(RoleStrategy):
    def decide_mission_member(self, team_size: int) -> list:
        """莫甘娜组队：混入邪恶成员，模仿好人行为"""
        candidates = [self.player.index]
        current_round = len(self.player.team_history) + 1
        
        # 后期增加邪恶盟友
        evil_allies = [p for p in self.player.trusted_evil if p != self.player.index]
        if evil_allies and current_round >= 3:
            candidates.append(random.choice(evil_allies))
        
        # 选择看似好人的玩家获取信任
        sorted_by_suspicion = sorted(self.player.players, key=lambda p: self.player.suspicion_level[p])
        for p in sorted_by_suspicion:
            if p not in candidates and len(candidates) < team_size:
                candidates.append(p)
        
        return sorted(candidates[:team_size])
    
    def say(self) -> str:
        """莫甘娜发言：模仿梅林，指控好人"""
        # 模仿梅林的发言模式
        good_players = [p for p in self.player.players 
                       if p not in self.player.trusted_evil and p != self.player.index]
        
        if good_players and random.random() < 0.4:
            target = random.choice(good_players)
            return f"我观察到P{target}的行为有些异常，可能需要注意。"
        
        return "我们需要保持警惕，仔细观察。"
    
    def mission_vote1(self) -> bool:
        """莫甘娜投票：支持有利于破坏的队伍"""
        current_team = self.player.team_history[-1][1] if self.player.team_history else []
        evil_on_team = [p for p in current_team if p in self.player.trusted_evil or p == self.player.index]
        
        if self.player.index in current_team and len(evil_on_team) >= 1:
            return True
        
        if len(evil_on_team) == 0:
            return random.random() < 0.3  # 低概率支持纯好人队伍
        
        needed_evil = 2 if len(current_team) >= 5 else 1
        return len(evil_on_team) >= needed_evil
    
    def mission_vote2(self) -> bool:
        """莫甘娜任务执行：破坏任务"""
        current_round = len(self.player.mission_results) + 1
        if current_round == 1:
            return False if random.random() < 0.8 else True
        return False

    def walk(self) -> tuple:
        """莫甘娜移动策略：模仿梅林行为，适度移动"""
        current_pos = self.player.player_positions.get(self.player.index)
        return random_walk(current_pos, self.player.player_positions, self.player.map, 2)

# 刺客策略
class AssassinStrategy(RoleStrategy):
    def decide_mission_member(self, team_size: int) -> list:
        """刺客组队：主动加入队伍伺机破坏"""
        candidates = [self.player.index]
        
        # 优先选择邪恶盟友
        evil_allies = [p for p in self.player.trusted_evil if p != self.player.index]
        if evil_allies:
            candidates.append(random.choice(evil_allies))
        
        # 随机选择其他玩家
        remaining = [p for p in self.player.players if p not in candidates]
        while len(candidates) < team_size and remaining:
            candidates.append(random.choice(remaining))
            remaining.remove(candidates[-1])
        
        return sorted(candidates[:team_size])
    
    def say(self) -> str:
        """刺客发言：试图陷害好人玩家"""
        good_players = [p for p in self.player.players 
                       if p not in self.player.trusted_evil and p != self.player.index]
        
        if good_players and random.random() < 0.5:
            target = random.choice(good_players)
            return f"我怀疑P{target}可能是间谍，大家要小心。"
        
        # 支持邪恶盟友
        evil_allies = [p for p in self.player.trusted_evil if p != self.player.index]
        if evil_allies and random.random() < 0.3:
            ally = random.choice(evil_allies)
            return f"我觉得P{ally}的分析很有道理，值得信任。"
        
        return "我们要团结起来，找出真正的间谍。"
    
    def mission_vote1(self) -> bool:
        """刺客投票：支持有利于破坏的队伍"""
        current_team = self.player.team_history[-1][1] if self.player.team_history else []
        evil_on_team = [p for p in current_team if p in self.player.trusted_evil or p == self.player.index]
        
        if self.player.index in current_team and len(evil_on_team) >= 1:
            return True
        
        needed_evil = 2 if len(current_team) >= 5 else 1
        return len(evil_on_team) >= needed_evil
    
    def mission_vote2(self) -> bool:
        """刺客任务执行：破坏任务"""
        current_round = len(self.player.mission_results) + 1
        if current_round == 1:
            return False if random.random() < 0.8 else True
        return False
    
    def assass(self) -> int:
        """刺客刺杀：分析梅林特征"""
        candidate_scores = defaultdict(float)
        
        # 基于任务表现分析
        for i, (team, result) in enumerate(zip(self.player.team_history, self.player.mission_results)):
            for p in team:
                if result:  # 任务成功
                    candidate_scores[p] += 2
                else:  # 任务失败
                    candidate_scores[p] -= 1
        
        # 基于发言分析
        merlin_candidates = get_merlin_candidates_from_messages(self.player.memory, self.player.known_roles)
        for candidate in merlin_candidates:
            candidate_scores[candidate] += 3
        
        # 基于信任度分析
        for p in self.player.players:
            if p == self.player.index or p in self.player.trusted_evil:
                continue
            candidate_scores[p] += (5 - self.player.suspicion_level[p])
        
        # 派西维尔的候选人优先级更高
        if 'Merlin_Morgana_candidates' in self.player.known_roles:
            for p in self.player.known_roles['Merlin_Morgana_candidates']:
                if p not in self.player.trusted_evil:
                    candidate_scores[p] += 15
        
        # 排除已知邪恶玩家
        for evil in self.player.trusted_evil:
            candidate_scores.pop(evil, None)
        candidate_scores.pop(self.player.index, None)
        
        # 选择得分最高的目标
        if candidate_scores:
            max_score = max(candidate_scores.values())
            best_candidates = [p for p, s in candidate_scores.items() if s == max_score]
            target = random.choice(best_candidates)
            write_into_private(f"刺杀目标：P{target}，得分：{candidate_scores[target]}")
            return target
        
        # 备选方案：随机选择非邪恶玩家
        valid_targets = [p for p in self.player.players if p != self.player.index and p not in self.player.trusted_evil]
        return random.choice(valid_targets) if valid_targets else 1

    def walk(self) -> tuple:
        """刺客移动策略：尝试靠近疑似梅林的玩家"""
        current_pos = self.player.player_positions.get(self.player.index)
        if not current_pos:
            return tuple()
        
        # 寻找疑似梅林的玩家
        merlin_suspects = []
        if 'Merlin_Morgana_candidates' in self.player.known_roles:
            # 派西维尔的候选人中排除已知邪恶
            for p in self.player.known_roles['Merlin_Morgana_candidates']:
                if p not in self.player.trusted_evil:
                    merlin_suspects.append(p)
        
        # 如果没有明确目标，选择信任度低的好人
        if not merlin_suspects:
            merlin_suspects = [p for p in self.player.players 
                              if p != self.player.index and p not in self.player.trusted_evil 
                              and self.player.suspicion_level[p] < 3]
        
        if merlin_suspects:
            suspect_positions = [self.player.player_positions.get(p) for p in merlin_suspects 
                               if p in self.player.player_positions]
            suspect_positions = [pos for pos in suspect_positions if pos]
            
            if suspect_positions:
                return strategic_walk_towards(current_pos, suspect_positions, 
                                            self.player.player_positions, self.player.map, 3)
        
        return random_walk(current_pos, self.player.player_positions, self.player.map, 3)

# 奥伯伦策略
class OberonStrategy(RoleStrategy):
    def decide_mission_member(self, team_size: int) -> list:
        """奥伯伦组队：随机选择，制造混乱"""
        candidates = []
        if random.random() < 0.6:
            candidates.append(self.player.index)
        
        while len(candidates) < team_size:
            remaining = [p for p in self.player.players if p not in candidates]
            if remaining:
                candidates.append(random.choice(remaining))
            else:
                break
        
        return sorted(candidates[:team_size])
    
    def say(self) -> str:
        """奥伯伦发言：混乱发言，制造困扰"""
        chaos_phrases = [
            "我觉得情况很复杂，需要更多信息。",
            "有人的行为很奇怪，但我说不清楚。",
            "我们可能都被误导了。",
            "局势不明朗，大家要小心。",
            "我有种不好的预感。"
        ]
        return random.choice(chaos_phrases)
    
    def mission_vote1(self) -> bool:
        """奥伯伦投票：基于当前轮次和队伍情况"""
        current_team = self.player.team_history[-1][1] if self.player.team_history else []
        current_round = len(self.player.team_history)
        
        if self.player.index in current_team:
            return True
        
        if current_round == 1:
            return random.random() < 0.6
        elif current_round != 4:
            # 选择信任度最低的玩家
            consider_dict = {p: self.player.kind_level[p] for p in self.player.players 
                           if p != self.player.index}
            if consider_dict:
                least_trusted = min(consider_dict.items(), key=lambda x: x[1])
                return least_trusted[0] in current_team
            return False
        else:  # 第4轮
            return random.random() < 0.7
    
    def mission_vote2(self) -> bool:
        """奥伯伦任务执行：破坏任务"""
        current_round = len(self.player.mission_results) + 1
        if current_round == 1:
            return False if random.random() < 0.8 else True
        return False

    def walk(self) -> tuple:
        """奥伯伦移动策略：由于听力范围大且独狼，随机移动保持神秘"""
        current_pos = self.player.player_positions.get(self.player.index)
        return random_walk(current_pos, self.player.player_positions, self.player.map, 
                          random.randint(0, 3))  # 更随机的移动步数

#### 主Player类
class Player:
    def __init__(self):
        '''初始化，设置必要参数'''
        self.index = None
        self.role = None
        self.map = None
        self.memory = set()
        self.trusted_evil = set()
        self.trusted_good = set()
        self.team_history = []
        self.vote_history = defaultdict(list)
        self.mission_results = []
        self.assassination_target = None
        self.suspicion_level = defaultdict(float)
        self.kind_level = defaultdict(float)
        self.wise_level = defaultdict(float)
        self.players = [1, 2, 3, 4, 5, 6, 7]
        self.player_positions = {}
        self.known_roles = {}
        self.hearing_range = 3
        self.current_round = 0
        self.consecutive_rejections = 0
        self.mission_vote_history = {}
        self.past_team = {}
        self.past_leader = {}
        self.strategy = None  # 角色策略对象
        
    def set_player_index(self, index: int):
        '''设置玩家索引'''
        self.index = index
        
    def set_role_type(self, role_type: str):
        '''设置玩家角色'''
        self.role = role_type
        self.hearing_range = calculate_hearing_range(role_type)
        
        # 根据角色设置策略
        if self.role == "Merlin":
            self.strategy = MerlinStrategy(self)
            write_into_private(f"我是梅林，可以看到所有红方玩家。听力范围：{self.hearing_range}x{self.hearing_range}")
        elif self.role == "Percival":
            self.strategy = PercivalStrategy(self)
            write_into_private(f"我是派西维尔，可以看到梅林和莫甘娜但无法区分。听力范围：{self.hearing_range}x{self.hearing_range}")
        elif self.role == "Knight":
            self.strategy = KnightStrategy(self)
            write_into_private(f"我是骑士，拥有更广的听力范围。听力范围：{self.hearing_range}x{self.hearing_range}")
        elif self.role == "Morgana":
            self.strategy = MorganaStrategy(self)
            write_into_private(f"我是莫甘娜，邪恶阵营。听力范围：{self.hearing_range}x{self.hearing_range}")
        elif self.role == "Assassin":
            self.strategy = AssassinStrategy(self)
            write_into_private(f"我是刺客，邪恶阵营。听力范围：{self.hearing_range}x{self.hearing_range}")
        elif self.role == "Oberon":
            self.strategy = OberonStrategy(self)
            write_into_private(f"我是奥伯伦，邪恶独狼。听力范围：{self.hearing_range}x{self.hearing_range}")
        else:
            self.strategy = RoleStrategy(self)  # 默认策略
        
    def pass_role_sight(self, role_sight: dict[str, int]):
        '''处理夜晚阶段的视野信息'''
        self.sight = role_sight
        
        if self.role == "Merlin":
            self.trusted_evil.update(role_sight.values())
            write_into_private(f"梅林视野：红方玩家是 {list(role_sight.values())}")
        elif self.role == "Percival":
            self.known_roles['Merlin_Morgana_candidates'] = list(role_sight.values())
            write_into_private(f"派西维尔视野：梅林/莫甘娜候选人 {list(role_sight.values())}")
        elif self.role in {"Morgana", "Assassin"}:
            self.trusted_evil.update(role_sight.values())
            write_into_private(f"红方互认：队友是 {list(role_sight.values())}")
        
    def pass_map(self, game_map):
        self.map = game_map
        
    def pass_position_data(self, player_positions: dict[int, tuple]):
        self.player_positions = player_positions

    def pass_message(self, content: tuple[int, str]):
        speaker, msg = content
        speaker_pos = self.player_positions.get(speaker)
        self_pos = self.player_positions.get(self.index)
        
        # 只处理听力范围内的消息
        if not is_in_hearing_range(speaker_pos, self_pos, self.hearing_range):
            return
            
        self.memory.add(content)
        
        # 分析消息内容并更新信任度
        changes = analyze_message_content(msg, self.role, self.trusted_evil, self.trusted_good)
        
        if changes['speaker_trust'] > 0:
            self.kind_level[speaker] += changes['speaker_trust']
        if changes['speaker_suspicion'] > 0:
            self.suspicion_level[speaker] += changes['speaker_suspicion']

    def pass_mission_members(self, leader: int, mission_members: list):
        self.team_history.append((leader, mission_members))

    # 委托给策略对象的方法
    def decide_mission_member(self, team_size: int) -> list:
        return self.strategy.decide_mission_member(team_size)
    
    def say(self) -> str:
        return self.strategy.say()
    
    def walk(self) -> tuple:
        return self.strategy.walk()
    
    def mission_vote1(self) -> bool:
        return self.strategy.mission_vote1()
    
    def mission_vote2(self) -> bool:
        return self.strategy.mission_vote2()
    
    def assass(self) -> int:
        # 只有刺客才有刺杀方法
        if hasattr(self.strategy, 'assass'):
            return self.strategy.assass()
        return 1  # 默认返回
    
