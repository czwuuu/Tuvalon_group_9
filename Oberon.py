class OberonStrategy(RoleStrategy):
    def decide_mission_member(self, team_size: int) -> list:
        """奥伯伦组队策略：始终带自己，其余选最可信蓝方"""
        candidates = [self.player.index]
        others = [p for p in self.player.players if p != self.player.index]

        # 优先 trusted_good 中的高 kind_level 玩家
        trusted_candidates = sorted(
            [p for p in self.player.trusted_good if p in others],
            key=lambda x: self.player.kind_level[x],
            reverse=True
        )

        # 再按 kind_level 排序，排除 trusted_evil
        fallback_candidates = sorted(
            [p for p in others if p not in self.player.trusted_evil and p not in trusted_candidates],
            key=lambda x: self.player.kind_level[x],
            reverse=True
        )

        for p in trusted_candidates + fallback_candidates:
            if len(candidates) < team_size:
                candidates.append(p)
            else:
                break

        return sorted(candidates)

    def mission_vote1(self) -> bool:
        """奥伯伦第一轮两人任务通过，其余全部反对"""
        current_round = len(self.player.team_history) + 1
        current_team = self.player.team_history[-1][1] if self.player.team_history else []
        
        # 第一轮且任务人数为2
        if current_round == 1 and len(current_team) == 2:
            return True
        return False  # 其他轮次一律反对

    def mission_vote2(self) -> bool:
        """任务执行策略：第一轮有 80% 概率破坏，其余全破坏"""
        current_round = len(self.player.mission_results) + 1
        if current_round == 1:
            return False if random.random() < 0.8 else True
        return False

    def say(self) -> str:
        """奥伯伦发言：像好人但不主动接触他人"""
        neutral_phrases = [
            "我还在观察大家的表现。",
            "这一轮的信息还不够，我不轻易下判断。",
            "我们要谨慎，每个细节都不能忽视。",
            "希望这次任务能顺利完成。",
            "我不确定谁是坏人，但我会认真分析。"
        ]
        return random.choice(neutral_phrases)

    def walk(self) -> tuple:
        """奥伯伦移动策略：更随机的走位保持神秘"""
        current_pos = self.player.player_positions.get(self.player.index)
        return random_walk(current_pos, self.player.player_positions, self.player.map, 
                           random.randint(0, 3))
