# envs/multi_step_limited_attack_env.py (V3 con límite de "zona de ataque")
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AttackEnvLimitedMultiStep(gym.Env):
    """
    Entorno donde el atacante puede modificar un mismo ataque durante varios pasos.
    - Estado: punto 2D actual (x_adv).
    - Acción: delta_x, delta_y en [-epsilon, epsilon].
    - Recompensa:
        - señal densa si reduce la prob de ser ataque,
        - recompensa final fuerte si evade al clasificador SIN alejarse demasiado del ataque original,
        - penalización por perturbación grande.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        attack_samples: np.ndarray,
        clf,
        threshold: float = 0.5,
        epsilon: float = 0.2,   # Más pequeño para no pegar saltos enormes
        penalty: float = 0.05,  # Penalización un poco más fuerte
        max_steps: int = 5,
        max_norm: float = 1,  # Radio máximo desde el ataque original
    ):
        super().__init__()

        assert attack_samples.shape[1] == 2, "Esperaba datos 2D"
        self.attack_samples = attack_samples
        self.clf = clf
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.penalty = float(penalty)
        self.max_steps = int(max_steps)
        self.max_norm = float(max_norm)

        self.action_space = spaces.Box(
            low=-self.epsilon,
            high=self.epsilon,
            shape=(2,),
            dtype=np.float32,
        )

        low = np.array([-10.0, -10.0], dtype=np.float32)
        high = np.array([10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.rng = np.random.default_rng(42)
        self.x_orig = None
        self.x_adv = None
        self.step_count = 0

    def _sample_attack(self):
        idx = self.rng.integers(0, len(self.attack_samples))
        return self.attack_samples[idx]

    def _p_attack(self, x):
        return float(self.clf.predict_proba(x.reshape(1, -1))[0, 1])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.x_orig = self._sample_attack()
        self.x_adv = self.x_orig.copy()
        self.step_count = 0

        obs = self.x_adv.astype(np.float32)
        info = {
            "x_orig": self.x_orig.copy(),
            "x_adv": self.x_adv.copy(),
            "p_attack": self._p_attack(self.x_adv),
        }
        return obs, info

    def step(self, action):

        self.step_count += 1

        delta = np.clip(action, -self.epsilon, self.epsilon)

        prev_p_attack = self._p_attack(self.x_adv)

        self.x_adv = self.x_adv + delta

        # Cálculamos la distancia al ataque original
        dist = float(np.linalg.norm(self.x_adv - self.x_orig))

        # Comprobamos si estamos fuera del radio máximo permitido
        if dist > self.max_norm:
            # Proyectamos el punto de vuelta al borde del círculo permitido
            direction = (self.x_adv - self.x_orig) / (dist + 1e-8) # Calculamos la dirección desde el original al adversarial
            self.x_adv = self.x_orig + direction * self.max_norm # Proyectamos el punto adversarial al borde del círculo permitido
            dist = self.max_norm # Actualizamos la distancia al máximo permitido

        p_attack = self._p_attack(self.x_adv)
        pred = int(p_attack >= self.threshold)

        reward = 0.0

        reward += (prev_p_attack - p_attack)

        reward -= self.penalty * float(np.sum(delta**2))

        terminated = False
        truncated = False
        success = 0

        if self.step_count >= self.max_steps:
            truncated = True

            # Damos por valida la evasión si el clasificador no lo detecta como ataque y estamos dentro del radio permitido
            if pred == 0 and dist <= self.max_norm:
                reward += 1.0
                success = 1

        info = {
            "x_orig": self.x_orig.copy(),
            "x_adv": self.x_adv.copy(),
            "p_attack": p_attack,
            "step_count": self.step_count,
            "pred": pred,
            "success": success,
            "dist": dist, # Distancia al ataque original
        }

        obs = self.x_adv.astype(np.float32)
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
