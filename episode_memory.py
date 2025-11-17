# ==========================
# episode_memory.py
# Joystick of Thought V4.x
# ==========================

import json
import math
import os


class EpisodeMemory:
    def __init__(self, filename="episodes.json"):
        self.filename = filename

        # No episodes file yet → create with initial tor and infinite step
        if not os.path.exists(self.filename):
            self.episodes = [{
                "tor_id": 0,
                "path_length": 0.0,
                "spiral_step": "∞"
            }]
            self._save()
        else:
            self._load()

    # ----------------------------------------
    def _load(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            self.episodes = json.load(f)

    def _save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, indent=4, ensure_ascii=False)

    # ----------------------------------------
    def add_episode(self, path_length, new_spiral_step):
        """
        Add new episode:
        - path_length: длина пути follower от предыдущего тора
        - new_spiral_step: шаг спирали φ* следующего тора
        """

        new_id = len(self.episodes)

        ep = {
            "tor_id": new_id,
            "path_length": float(path_length),
            "spiral_step": float(new_spiral_step)
        }

        self.episodes.append(ep)
        self._save()

    # ----------------------------------------
    @staticmethod
    def compute_path_length(points):
        """
        Получаем список точек движения follower:
        points = [(x0,y0), (x1,y1), ...]
        Возвращаем суммарный путь.
        """
        total = 0.0
        for i in range(1, len(points)):
            x0, y0 = points[i-1]
            x1, y1 = points[i]
            total += math.dist((x0, y0), (x1, y1))
        return total

    # ----------------------------------------
    @staticmethod
    def compute_new_spiral_step(prev_step, path_length, k=0.01):
        """
        Главная формула V4.x:
        φ*_{i+1} = φ*_i * g(L_i)
        где g(L) = 1 + k·L
        """

        # Если предыдущий шаг был бесконечностью
        if prev_step == "∞":
            return 1.0  # Начальный реальный шаг

        return float(prev_step) * (1.0 + k * path_length)

    # ----------------------------------------
    def last_spiral_step(self):
        """Возвращает шаг последнего тора."""
        return self.episodes[-1]["spiral_step"]

    # ----------------------------------------
    def __len__(self):
        return len(self.episodes)
