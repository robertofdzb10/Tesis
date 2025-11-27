# envs/defender_env_v2.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DefenderEnvV2(gym.Env):

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
        move_penalty: float = 0.05, # 
        extreme_penalty: float = 0.5,
        fn_penalty: float = -4.0, #
        fp_penalty: float = -3.0, #
        tp_reward: float = 2.0, #
        tn_reward: float = 1.0, #
    ):
        super().__init__()

        assert normal_samples.shape[1] == 2
        self.normal_samples = normal_samples
        self.normal_center = np.mean(normal_samples, axis=0)
        self.normal_radius = np.mean(np.linalg.norm(normal_samples - self.normal_center, axis=1))

        self.attack_env = attack_env
        self.attacker_model = attacker_model

        self.init_threshold = float(init_threshold)
        self.threshold = float(init_threshold)
        self.delta_max = float(delta_max)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.attack_prob = float(attack_prob)
        self.episode_length = int(episode_length)

        self.move_penalty = move_penalty
        self.extreme_penalty = extreme_penalty

        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        self.tp_reward = tp_reward
        self.tn_reward = tn_reward

        self.rng = np.random.default_rng(123)
        self.step_count = 0

        self.action_space = spaces.Box(
            low=-self.delta_max,
            high=self.delta_max,
            shape=(1,),
            dtype=np.float32,
        )

        # Extendemos el espacio de observaciiones extendida [x0, x1, threshold, p_attack, distancia del punto al centro de los datos normales, número de paso dentro del episodio (le da contexto al agente]
        low = np.array([-10, -10, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([10, 10, 1, 1, 5, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.x_sample = None
        self.y_sample = None

    # Calcula la distancia euclídea entre el punto x y el centro del cluster de muestras normales
    def _dist_to_center(self, x):
        return np.linalg.norm(x - self.normal_center)

    def _sample_normal(self):
        idx = self.rng.integers(0, len(self.normal_samples))
        return self.normal_samples[idx]

    def _generate_attack_sample(self):
        obs, info = self.attack_env.reset()
        done = False
        truncated = False
        last_info = info

        while not (done or truncated):
            action, _ = self.attacker_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, last_info = self.attack_env.step(action)

        return last_info["x_adv"]

    def _sample_sample(self):
        if self.rng.random() < self.attack_prob:
            return self._generate_attack_sample(), 1
        return self._sample_normal(), 0

    def _p_attack(self, x):
        clf = self.attack_env.clf
        return float(clf.predict_proba(x.reshape(1, -1))[0, 1])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.threshold = self.init_threshold

        self.x_sample, self.y_sample = self._sample_sample()
        p = self._p_attack(self.x_sample)

        obs = np.array([
            self.x_sample[0],
            self.x_sample[1],
            self.threshold,
            p,
            self._dist_to_center(self.x_sample) / self.normal_radius,
            self.step_count / self.episode_length
        ], dtype=np.float32)

        info = {
            "x_sample": self.x_sample.copy(),
            "y_sample": int(self.y_sample),
            "threshold": float(self.threshold),
            "p_attack": p,
        }
        return obs, info

    def step(self, action):

        self.step_count += 1

        # Clip action
        delta_th = float(np.clip(action[0], -self.delta_max, self.delta_max))
        old_threshold = self.threshold

        # Update threshold
        self.threshold = float(np.clip(self.threshold + delta_th, self.min_threshold, self.max_threshold))

        # Prediction
        p_attack = self._p_attack(self.x_sample)
        pred = int(p_attack >= self.threshold)

        # ------------------ REWARD ------------------
        reward = 0.0

        if self.y_sample == 1:  # ataque
            reward += self.tp_reward if pred == 1 else self.fn_penalty
        else:
            reward += self.tn_reward if pred == 0 else self.fp_penalty

        # Penalización por moverse demasiado
        reward -= self.move_penalty * (abs(delta_th) / self.delta_max)

        # Penalización progresiva por extremos
        if self.threshold < 0.25 or self.threshold > 0.75:
            reward -= self.extreme_penalty

        # ------------------------------------------------

        done = False
        truncated = self.step_count >= self.episode_length

        # New sample
        self.x_sample, self.y_sample = self._sample_sample()
        p2 = self._p_attack(self.x_sample)

        obs = np.array([
            self.x_sample[0],
            self.x_sample[1],
            self.threshold,
            p2,
            self._dist_to_center(self.x_sample) / self.normal_radius,
            self.step_count / self.episode_length
        ], dtype=np.float32)

        info = {
            "x_sample": self.x_sample.copy(),
            "y_sample": int(self.y_sample),
            "threshold": float(self.threshold),
            "p_attack": p2,
            "pred": pred,
            "step_count": self.step_count
        }

        return obs, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass
