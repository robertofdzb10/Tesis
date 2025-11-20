# envs/multi_step_attack_env.py (V2 con multi-step)
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AttackEnvMultiStep(gym.Env):
    """
    Entorno donde el atacante puede modificar un mismo ataque durante varios pasos.
    - Estado: punto 2D actual (x_adv) y, opcionalmente, el original.
    - Acción: delta_x, delta_y en [-epsilon, epsilon].
    - Recompensa:
        - pequeña señal en cada paso si reduce la prob de ser ataque,
        - recompensa final fuerte si evade al clasificador,
        - penalización por perturbación grande acumulada.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        attack_samples: np.ndarray,
        clf,
        threshold: float = 0.5,
        epsilon: float = 0.5,
        penalty: float = 0.01,
        max_steps: int = 5, # Número máximo de pasos por episodio
    ):
        super().__init__()

        assert attack_samples.shape[1] == 2, "Esperaba datos 2D"
        self.attack_samples = attack_samples
        self.clf = clf
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.penalty = float(penalty)
        self.max_steps = int(max_steps)

        self.action_space = spaces.Box(
            low=-self.epsilon,
            high=self.epsilon,
            shape=(2,),
            dtype=np.float32,
        )

        low = np.array([-10.0, -10.0], dtype=np.float32)
        high = np.array([10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Añadimos nuevas variables para el entorno multi-step
        self.rng = np.random.default_rng(42)
        self.x_orig = None
        self.x_adv = None # Estado actual (punto adversarial)
        self.step_count = 0 # Contador de pasos en el episodio

    def _sample_attack(self):
        idx = self.rng.integers(0, len(self.attack_samples))
        return self.attack_samples[idx]

    # Función para obtener la probabilidad de que un punto sea clasificado como ataque
    def _p_attack(self, x):
        return float(self.clf.predict_proba(x.reshape(1, -1))[0, 1]) # La función devuevle la probabilidad de que x sea un ataque segun el clasificador 

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Añadimos nuevas variables para el entorno multi-step
        self.x_orig = self._sample_attack()
        self.x_adv = self.x_orig.copy() # Estado actual (punto adversarial)
        self.step_count = 0

        obs = self.x_adv.astype(np.float32)
        info = {
            "x_orig": self.x_orig.copy(),
            "x_adv": self.x_adv.copy(),
            "p_attack": self._p_attack(self.x_adv),
        }
        return obs, info

    def step(self, action):

        # Incrementamos el contador de pasos
        self.step_count += 1

        # Aplicamos la acción con clipping para no exceder epsilon
        delta = np.clip(action, -self.epsilon, self.epsilon)

        # Probabilidad del ataque antes de aplicar la acción
        prev_p_attack = self._p_attack(self.x_adv)

        # Actualizamos el punto adversarial
        self.x_adv = self.x_adv + delta

        # Calculamos la nueva probabilidad de ser clasificado como ataque
        p_attack = self._p_attack(self.x_adv)

        # Predicción del clasificador con el nuevo punto adversarial
        pred = int(p_attack >= self.threshold)

        # Recompensa inicial
        reward = 0.0

        # Si logramos que la probabilidad de ataque final sea menor que la inicial, damos recompensa
        reward += (prev_p_attack - p_attack) # La recompensa es cuanto hemos conseguido reducir la probabilidad de ser clasificado como ataque, cuanto más reducción, más recompensa

        # Penalización por perturbación grande, queremos que el agente sea sigiloso
        reward -= self.penalty * float(np.sum(delta**2)) # La penalización es proporcional al tamaño del cambio realizado

        # Episodio no termina hasta que se alcance el máximo de pasos
        terminated = False
        truncated = False

        # Inicializamos el indicador de éxito
        success = 0

        # Si llegamos al último paso del episodio:
        if self.step_count >= self.max_steps:

            # Marcamos el episodio como terminado
            truncated = True

            # Si hemos evadido al clasificador, damos una recompensa fuerte
            if pred == 0:
                reward += 1.0 # Recompensa fuerte por evadir al clasificador
                success = 1 # Indicador de éxito

        # Información adicional 
        info = {
            "x_orig": self.x_orig.copy(),
            "x_adv": self.x_adv.copy(),
            "p_attack": p_attack,
            "step_count": self.step_count,
            "pred": pred,
            "success": success,
        }

        # Preparamos la siguiente observación
        obs = self.x_adv.astype(np.float32)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
