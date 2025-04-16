# Agent Class
import numpy as np

class Agent:
    def __init__(self, id, position, speed, perception_radius):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.perception_radius = perception_radius

        self.target_A = None
        self.target_B = None

    def set_targets(self, A, B):
        self.target_A = A
        self.target_B = B

    def get_direction(self, strategy="between"):
        if self.target_A is None or self.target_B is None:
            return np.zeros(2)

        if strategy == "between":
            midpoint = (self.target_A.position + self.target_B.position) / 2
            return midpoint - self.position

        elif strategy == "behind":
            # Try to hide behind B from A
            vector = self.target_B.position - self.target_A.position
            target = self.target_B.position + vector
            return target - self.position

        # default no movement
        return np.zeros(2)

    def update_position(self, strategy="between"):
        direction = self.get_direction(strategy)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm  # normalize
            self.position += direction * self.speed
