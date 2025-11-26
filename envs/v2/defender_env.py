# envs/defender_env.py (V0 sencillo)
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DefenderEnv(gym.Env):
    """
    Entorno para entrenar al DEFENSOR.
    - Hay dos tipos de muestras: normal (y=0) y ataque (y=1).
    - Si es normal: x_sample = x_normal
    - Si es ataque: x_sample = x_adv generado por el ATACANTE SAC (multi-step)
    - El defensor observa: [x_sample[0], x_sample[1], threshold_actual]
    - Acción: delta_threshold en [-delta_max, delta_max]
    - Recompensa:
        - y=1 (ataque):
            - pred=1 (detecta)   -> +1
            - pred=0 (se escapa) -> -2
        - y=0 (normal):
            - pred=0 (ok)        -> +0.5
            - pred=1 (FP)        -> -1
      Además se penalizan thresholds extremos.
    """

    # Metadatos del entorno, indica los modos de render disponibles, ninguno por ahora
    metadata = {"render_modes": []}

    def __init__(
        self,
        normal_samples: np.ndarray, # Lista de ejemplos normales
        attack_env,             # Instancia del entorno AttackEnvMultiStep
        attacker_model,         # Instancia del modelo SAC del atacante
        init_threshold: float = 0.5,  # Threshold inicial del defensor, apartid de que probabilidad considera ataque
        delta_max: float = 0.1, # Máximo cambio que el defensor puede hacer al threshold en cada paso
        min_threshold: float = 0.05, # Mínimo valor permitido para el threshold
        max_threshold: float = 0.95, # Máximo valor permitido para el threshold
        attack_prob: float = 0.5, # Probabilidad de que la siguiente muestra sea un ataque
        episode_length: int = 50, # Longitud del episodio en pasos
        extremal_penalty: float = 0.1, # Penalización por usar thresholds demasiado extremos
    ):
        
        super().__init__()

        # Asignamos parámetros
        assert normal_samples.shape[1] == 2, "Esperaba datos 2D"
        self.normal_samples = normal_samples
        self.attack_env = attack_env
        self.attacker_model = attacker_model
        self.init_threshold = float(init_threshold)
        self.threshold = float(init_threshold)
        self.delta_max = float(delta_max)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.attack_prob = float(attack_prob)
        self.episode_length = int(episode_length)
        self.extremal_penalty = float(extremal_penalty)

        # Generador de números aleatorios con semilla fija para reproducibilidad
        self.rng = np.random.default_rng(123)

        # Contador de pasos en el episodio
        self.step_count = 0

        # Espacio de acciones del defensor
        self.action_space = spaces.Box( # Acción: cambiar threshold en [-delta_max, delta_max]
            low=-self.delta_max, # El valor mínimo de la acción
            high=self.delta_max, # El valor máximo de la acción
            shape=(1,), # El defensor entrega un solo número
            dtype=np.float32, # Tipo de dato de los números
        )

        # Indicamos el espacio de observaciones
        low = np.array([-10.0, -10.0, 0.0], dtype=np.float32) # Rango mínimo de las observaciones [x_sample[0], x_sample[1], threshold]
        high = np.array([10.0, 10.0, 1.0], dtype=np.float32) # Rango máximo de las observaciones
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) # Espacio de observaciones: [x_sample[0], x_sample[1], threshold]

        # Guardamos el ultimo sample generado
        self.x_sample = None # punto 2D actual
        self.y_sample = None  # etiqueta actual (0=normal, 1=ataque)

    # Función para muestrear una muestra normal aleatoria
    def _sample_normal(self):
        
        # Elegimos un número al hazar entre 0 y el número de muestras normales disponibles
        idx = self.rng.integers(0, len(self.normal_samples))

        # Devolvemos la muestra normal correspondiente a ese índice
        return self.normal_samples[idx]

    # Función para generar una muestra de ataque usando el entorno del atacante y su modelo SAC
    def _generate_attack_sample(self):
        """
        Usa el AttackEnvMultiStep + attacker SAC para generar un x_adv.
        No nos importa la recompensa del atacante, solo su política.
        """
        
        # Reiniciamos el entorno del atacante
        obs, info = self.attack_env.reset()
        done = False
        truncated = False

        # Guardamos la última info
        last_info = info

        # Ejecutamos la política del atacante durante max_steps
        while not (done or truncated):

            # Obtener acción del modelo SAC del atacante dada la observación actual (determinística)
            action, _ = self.attacker_model.predict(obs, deterministic=True)

            # Ejecutar la acción en el entorno del atacante
            obs, reward, done, truncated, last_info = self.attack_env.step(action)

        # Obtenemos el punto adversarial generado
        x_adv = last_info["x_adv"]

        # Devolvemos el punto adversarial generado
        return x_adv

    # Función que decide de forma aleatoria si el proximo punto que vera el defensor es normal o de ataque
    def _sample_sample(self):
        """
        Decide si genera normal o ataque, y devuelve (x_sample, y_sample).
        """
        # Si un número aleatorio es menor que la probabilidad de ataque, generamos un ataque
        if self.rng.random() < self.attack_prob:
            x_sample = self._generate_attack_sample()
            y_sample = 1
        # Si no, generamos una muestra normal
        else:
            x_sample = self._sample_normal()
            y_sample = 0

        # Devolvemos la muestra y su etiqueta
        return x_sample, y_sample

    # Función para obtener la probabilidad de ataque según el clasificador con el que se entrenó el atacante
    def _p_attack(self, x):

        # Extraemos el clasificador del atacante
        clf = self.attack_env.clf

        # Devolvemos la probabilidad de que x sea un ataque según el clasificador del cual se ha entrenado el atacante
        return float(clf.predict_proba(x.reshape(1, -1))[0, 1])

    # Función de reinicio del entorno
    def reset(self, *, seed=None, options=None):

        # Reiniciamos el entorno con la semilla dada
        super().reset(seed=seed)

        # Si se proporciona una semilla, la usamos para el generador de números aleatorios
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reiniciamos el contador de pasos y el threshold
        self.step_count = 0
        self.threshold = self.init_threshold # Reiniciamos el threshold al valor inicial 0.5

        # Preparamos la primera muestra para el defensor
        self.x_sample, self.y_sample = self._sample_sample()

        # Preparamos la observación a devolver
        obs = np.array(
            [self.x_sample[0], self.x_sample[1], self.threshold], dtype=np.float32
        )

        # Información adicional
        info = {
            "x_sample": self.x_sample.copy(), # Punto 2D de la muestra
            "y_sample": int(self.y_sample), # Etiqueta de la muestra (0=normal, 1=ataque)
            "threshold": float(self.threshold), # Threshold actual
            "p_attack": self._p_attack(self.x_sample), # Probabilidad de que la muestra sea ataque según el clasificador del atacante
        }

        # Devolvemos la observación y la información adicional
        return obs, info

    # Función para ejecutar un paso del entorno
    def step(self, action):

        # Incrementamos el contador de pasos
        self.step_count += 1

        # Clip de la acción para no exceder delta_max
        delta_th = float(np.clip(action[0], -self.delta_max, self.delta_max))

        # Actualizamos el threshold con el cambio aplicado y aplicamos otro clip para que no exceda los límites min y max
        self.threshold = float(
            np.clip(self.threshold + delta_th, self.min_threshold, self.max_threshold)
        )

        # Calculamos la predicción del defensor sobre la muestra actual
        p_attack = self._p_attack(self.x_sample)

        # Predicción del defensor basada en el threshold actual (actualizado por el agente)
        pred = int(p_attack >= self.threshold)  # 1 = ataque, 0 = normal

        # Calculamos la recompensa según las reglas definidas
        reward = 0.0

        # Si es ataque (en la muestra) 
        if self.y_sample == 1:
            # Si es detectado
            if pred == 1:
                reward += 1.0 # Recompensa por detectar ataque
            # Si no es detectado
            else:  
                reward -= 2.0 # Penalización por no detectar ataque

        # Si es normal
        else: 
            # Si es bien clasificado
            if pred == 0:
                reward += 0.5 # Recompensa por clasificar bien normal
            # Si es falso positivo
            else:
                reward -= 1.0 # Penalización por falso positivo

        # Penalización si el threshold está demasiado extremo
        if self.threshold < 0.1 or self.threshold > 0.9:
            reward -= self.extremal_penalty  # Penalización por usar thresholds extremos

        # El episodio no termina hasta que se alcance la longitud máxima
        terminated = False
        truncated = False

        # Si alcanza la longitud máxima del episodio termina
        if self.step_count >= self.episode_length:
            truncated = True

        # Preparamos siguiente muestra para el próximo paso
        self.x_sample, self.y_sample = self._sample_sample()

        # Preparamos la siguiente observación a devolver
        obs = np.array(
            [self.x_sample[0], self.x_sample[1], self.threshold], dtype=np.float32
        )

        # Información adicional
        info = {
            "x_sample": self.x_sample.copy(), # Punto 2D de la muestra
            "y_sample": int(self.y_sample), # Etiqueta de la muestra (0=normal, 1=ataque)
            "threshold": float(self.threshold), # Threshold actual
            "p_attack": self._p_attack(self.x_sample), # Probabilidad de que la muestra sea ataque según el clasificador del atacante
            "pred": pred, # Predicción del defensor (0=normal, 1=ataque)
            "step_count": self.step_count, # Contador de pasos en el episodio
        } 

        # Devolvemos la siguiente observación, recompensa, estado de terminado, truncado e información adicional
        return obs, reward, terminated, truncated, info

    # Render del entorno
    def render(self):
        pass

    # Cierre del entorno
    def close(self):
        pass
