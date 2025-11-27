# envs/v3/defender_env_v3.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DefenderEnvV3(gym.Env):
    """
    Entorno avanzado para entrenar al DEFENSOR RL contra un ATACANTE SAC.

    Mejora sobre V2:
    - Observación extendida con info útil del ataque.
    - Dos acciones:
        a0 = delta_threshold
        a1 = sensitivity_boost
    - Reward densa (shaping)
    - Considera distancia adversarial, progreso del ataque,
      densidad normal, etc.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        normal_samples: np.ndarray,
        attack_env,
        attacker_model,
        init_threshold: float = 0.5,
        delta_max: float = 0.05,
        min_threshold: float = 0.2,
        max_threshold: float = 0.8,
        attack_prob: float = 0.3,
        episode_length: int = 50,
        # reward shaping
        tp_reward: float = 3.0,
        fn_penalty: float = -6.0,
        tn_reward: float = 1.0,
        fp_penalty: float = -2.0,
        move_penalty: float = 0.1,
        sensitivity_penalty: float = 0.05,
        extreme_penalty: float = 0.5,
    ):
        super().__init__()

        assert normal_samples.shape[1] == 2, "Esperaba datos 2D"

        # Dataset normal
        self.normal_samples = normal_samples
        self.normal_center = np.mean(normal_samples, axis=0)
        self.normal_radius = np.mean(
            np.linalg.norm(normal_samples - self.normal_center, axis=1)
        )

        # Atacante RL
        self.attack_env = attack_env
        self.attacker_model = attacker_model

        # Threshold
        self.init_threshold = float(init_threshold)
        self.threshold = float(init_threshold)
        self.delta_max = float(delta_max)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)

        # Otros parámetros
        self.attack_prob = float(attack_prob)
        self.episode_length = int(episode_length)

        # Reward shaping
        self.tp_reward = tp_reward
        self.fn_penalty = fn_penalty
        self.tn_reward = tn_reward
        self.fp_penalty = fp_penalty
        self.move_penalty = move_penalty
        self.sensitivity_penalty = sensitivity_penalty
        self.extreme_penalty = extreme_penalty

        # RNG
        self.rng = np.random.default_rng(123)
        self.step_count = 0

        # -------------------------------------------
        # ACTION SPACE: 2 acciones continuas
        # a0 = delta threshold
        # a1 = sensitivity boost (-1,1)
        # -------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-self.delta_max, -1.0], dtype=np.float32),
            high=np.array([self.delta_max, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # -------------------------------------------
        # OBSERVATION SPACE:
        # [x0, x1, threshold, p_attack,
        #  adv_distance_norm, adv_progress,
        #  dist_center_norm, cluster_score, step_norm]
        # -------------------------------------------
        low = np.array(
            [-10, -10, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
        )
        high = np.array(
            [10, 10, 1, 1, 3, 1, 5, 1, 1], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Estado actual
        self.x_sample = None
        self.y_sample = None

        # Variables del ataque actual
        self.last_adv_distance = 0.0
        self.last_adv_progress = 0.0

    # -------------------------------------------
    # Funciones auxiliares
    # -------------------------------------------

    def _dist_to_center(self, x):
        return np.linalg.norm(x - self.normal_center)

    def _sample_normal(self):
        idx = self.rng.integers(0, len(self.normal_samples))
        return self.normal_samples[idx]

    def _generate_attack_sample(self):
        """
        Ejecuta el atacante SAC multi-step y devuelve:
        - x_adv final
        - dist adversarial
        - progreso normalizado (step_count/max_steps)
        """
        obs, info = self.attack_env.reset()
        done = False
        truncated = False
        last_info = info

        while not (done or truncated):
            action, _ = self.attacker_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, last_info = self.attack_env.step(action)

        x_adv = last_info["x_adv"]
        adv_dist = last_info.get("dist", 0.0)
        max_steps = getattr(self.attack_env, "max_steps", 5)
        progress = min(last_info.get("step_count", 0) / max_steps, 1.0)

        return x_adv, adv_dist, progress

    def _sample_sample(self):
        if self.rng.random() < self.attack_prob:
            x_adv, dist, prog = self._generate_attack_sample()
            return x_adv, 1, dist, prog
        else:
            return (
                self._sample_normal(),
                0,
                0.0,
                0.0  # no hay progreso adversarial
            )

    def _p_attack(self, x):
        clf = self.attack_env.clf
        return float(clf.predict_proba(x.reshape(1, -1))[0, 1])

    # -------------------------------------------
    # reset()
    # -------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.threshold = self.init_threshold

        self.x_sample, self.y_sample, self.last_adv_distance, self.last_adv_progress = (
            self._sample_sample()
        )

        p = self._p_attack(self.x_sample)
        dist_center = self._dist_to_center(self.x_sample)
        dist_norm = dist_center / self.normal_radius
        cluster_score = 1.0 / (1.0 + dist_norm)

        obs = np.array(
            [
                self.x_sample[0],
                self.x_sample[1],
                self.threshold,
                p,
                self.last_adv_distance,
                self.last_adv_progress,
                dist_norm,
                cluster_score,
                0.0,  # step_norm
            ],
            dtype=np.float32,
        )

        info = {
            "x_sample": self.x_sample.copy(),
            "y_sample": int(self.y_sample),
            "threshold": float(self.threshold),
            "p_attack": p,
        }
        return obs, info

    # -------------------------------------------
    # step()
    # -------------------------------------------
    def step(self, action):

        self.step_count += 1

        # Acciones
        delta_th = float(action[0])
        sens_boost = float(action[1])

        # Actualizamos threshold global
        self.threshold = float(
            np.clip(self.threshold + delta_th, self.min_threshold, self.max_threshold)
        )

        # threshold dinámico (local)
        dynamic_threshold = float(
            np.clip(self.threshold + sens_boost * 0.1, 0.05, 0.95)
        )

        # Predicción del clasificador base
        p_attack = self._p_attack(self.x_sample)
        pred = int(p_attack >= dynamic_threshold)

        # -------------------------------------------
        # Reward principal
        # -------------------------------------------
        reward = 0.0

        if self.y_sample == 1:  # ataque
            reward += self.tp_reward if pred == 1 else self.fn_penalty
        else:  # normal
            reward += self.tn_reward if pred == 0 else self.fp_penalty

        # -------------------------------------------
        # Reward densa
        # -------------------------------------------
        # penalizar movimiento excesivo
        reward -= self.move_penalty * (abs(delta_th) / self.delta_max)

        # penalizar sensibilidad exagerada
        reward -= self.sensitivity_penalty * abs(sens_boost)

        # premiar detección en zonas difíciles
        reward += 0.2 * abs(p_attack - self.threshold)

        # penalización si threshold global se va a extremos
        if self.threshold < 0.25 or self.threshold > 0.75:
            reward -= self.extreme_penalty

        # premiar más si el ataque es muy adversarial
        reward += 0.1 * self.last_adv_distance

        # -------------------------------------------
        # Terminar episodio
        # -------------------------------------------
        done = False
        truncated = self.step_count >= self.episode_length

        # -------------------------------------------
        # siguiente muestra
        # -------------------------------------------
        (
            self.x_sample,
            self.y_sample,
            self.last_adv_distance,
            self.last_adv_progress,
        ) = self._sample_sample()

        next_p = self._p_attack(self.x_sample)
        dist_center = self._dist_to_center(self.x_sample)
        dist_norm = dist_center / self.normal_radius
        cluster_score = 1.0 / (1.0 + dist_norm)
        step_norm = self.step_count / self.episode_length

        obs = np.array(
            [
                self.x_sample[0],
                self.x_sample[1],
                self.threshold,
                next_p,
                self.last_adv_distance,
                self.last_adv_progress,
                dist_norm,
                cluster_score,
                step_norm,
            ],
            dtype=np.float32,
        )

        info = {
            "x_sample": self.x_sample.copy(),
            "y_sample": int(self.y_sample),
            "threshold": float(self.threshold),
            "p_attack": next_p,
            "pred": pred,
            "step_count": self.step_count,
        }

        return obs, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass
