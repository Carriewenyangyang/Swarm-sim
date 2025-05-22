import numpy as np

class Agent:
    def __init__(self, id, position, speed, perception_radius):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.perception_radius = perception_radius

        self.target_A = None
        self.target_B = None

        self.broadcast_message = None
        self.received_messages = []

    def broadcast(self):
        self.broadcast_message = {
            "id": self.id,
            "position": self.position.copy(),
            "target_ids": (
                self.target_A.id if self.target_A else None,
                self.target_B.id if self.target_B else None
            )
        }

    def receive_broadcasts(self, all_agents, communication_radius):
        self.received_messages = []
        for agent in all_agents:
            if agent is self or agent.broadcast_message is None:
                continue
            if np.linalg.norm(agent.position - self.position) <= communication_radius:
                self.received_messages.append(agent.broadcast_message)

    def select_targets(self, all_agents):
        neighbors = [a for a in all_agents if a is not self and
                     np.linalg.norm(self.position - a.position) <= self.perception_radius]

        # 提取邻居中已被别人选为目标的 ID
        used_ids = {tid for msg in self.received_messages for tid in msg["target_ids"] if tid is not None}
        candidates = [a for a in neighbors if a.id not in used_ids]

        if len(candidates) >= 2:
            self.target_A, self.target_B = np.random.choice(candidates, 2)
        elif len(neighbors) >= 2:
            self.target_A, self.target_B = np.random.choice(neighbors, 2)
        else:
            self.target_A, self.target_B = None, None

    def get_direction(self, strategy="between"):
        if self.target_A is None or self.target_B is None:
            return np.zeros(2)

        if strategy == "between":
            midpoint = (self.target_A.position + self.target_B.position) / 2
            return midpoint - self.position
        elif strategy == "behind":
            vector = self.target_B.position - self.target_A.position
            target = self.target_B.position + vector
            return target - self.position

        return np.zeros(2)

    def update_position(self, strategy="between"):
        direction = self.get_direction(strategy)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            self.position += direction * self.speed


    def select_targets_upgrade(self, all_agents):
        # 感知范围内的 agent（有位置）
        local_neighbors = [a for a in all_agents if a is not self and
                        np.linalg.norm(self.position - a.position) <= self.perception_radius]

        # 通过通信接收到的远处 agent 信息（有 ID 和位置）
        remote_candidates = [a for a in all_agents if a is not self and
                            a.id in {msg["id"] for msg in self.received_messages}
                            and a not in local_neighbors]

        # 总候选
        all_candidates = local_neighbors + remote_candidates

        # 去掉已被别人选过的
        used_ids = {tid for msg in self.received_messages for tid in msg["target_ids"] if tid is not None}
        final_candidates = [a for a in all_candidates if a.id not in used_ids]
        
        # Step 5: 尝试选择目标
        if len(final_candidates) >= 2:
            # 优先选最干净的
            self.target_A, self.target_B = np.random.choice(final_candidates, 2, replace=False)
        elif len(all_candidates) >= 2:
            # 如果没得选干净的，退而选重复的也行
            self.target_A, self.target_B = np.random.choice(all_candidates, 2, replace=False)
        else:
            self.target_A, self.target_B = None, None
