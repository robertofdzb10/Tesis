# envs/attack_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AttackEnv(gym.Env):
    """
    Entorno sencillo donde:
    - El estado es un punto 2D que representa un ataque.
    - La acción es una perturbación (delta_x, delta_y) acotada.
    - El agente intenta que el clasificador NO detecte el ataque.
    """

    # Metadatos del entorno, indica los modos de render disponibles, ninguno por ahora
    metadata = {"render_modes": []}

    def __init__(
        self,
        attack_samples: np.ndarray, # Lista de ejemplos de ataques
        clf, # Modelo de ML que predice si un punto es ataque o no (La "victima")
        threshold: float = 0.5, # Indica a partir de qué probabilidad se considera ataque
        epsilon: float = 0.5, # Distancia máxima que el agente mover el punto orginal (Si es 0.5, +0.5 -0.5)
        penalty: float = 0.1, # Penalización por moverse mucho
    ):
        super().__init__()

        # Asignamos parámetros
        assert attack_samples.shape[1] == 2, "Esperaba datos 2D" # Comprobación de seguridad, los datos deben ser 2D
        self.attack_samples = attack_samples
        self.clf = clf
        self.threshold = threshold
        self.epsilon = float(epsilon)
        self.penalty = float(penalty)

        # Espacio de acciones del agente
        self.action_space = spaces.Box( # Los "botones" que el agente puede usar, spaces.Box indica que es un espacio continuo, es decir puede tomar cualquier valor dentro de un rango
            low=-self.epsilon, # El valor mínimo de cada número
            high=self.epsilon, # El valor máximo de cada número
            shape=(2,), # El agente entrega dos números
            dtype=np.float32, # Tipo de dato de los números, con el que trabajan las redes neuronales
        )

        # Indicamos el espacio de observaciones
        low = np.array([-10.0, -10.0], dtype=np.float32) # Rango mínimo de las observaciones
        high = np.array([10.0, 10.0], dtype=np.float32) # Rango máximo de las observaciones
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) # Espacio de observaciones, en este caso puntos 2D entre -10 y 10

        # Estado actual (punto de ataque), inicialmente None
        self.current_x = None

        # Generador de números aleatorios con semilla 
        self.rng = np.random.default_rng(42)

    # Función para muestrear un ataque aleatorio
    def _sample_attack(self):

        # Elegimos un número al hazar entre 0 y el número de ataques disponibles
        idx = self.rng.integers(0, len(self.attack_samples)) 

        # Devolvemos el ataque correspondiente a ese índice
        return self.attack_samples[idx] 

    # Reinicio del entorno
    def reset(self, *, seed=None, options=None):

        # Reiniciamos el entorno con la semilla dada
        super().reset(seed=seed) 

        # Si se proporciona una semilla, la usamos para el generador de números aleatorios
        if seed is not None: 
            self.rng = np.random.default_rng(seed)

        # Escogemos un ataque aleatorio como estado inicial
        self.current_x = self._sample_attack()

        # Convertimos el estado a float32 para que lo entienda el agente
        obs = self.current_x.astype(np.float32)

        # Información adicional (vacía por ahora)
        info = {}

        # Devolvemos la observación y la información adicional
        return obs, info

    # Cada paso del entorno
    def step(self, action):

        # Clip de la acción
        delta = np.clip(action, -self.epsilon, self.epsilon) # El agente propone un movimiento (action) que se limita a estar dentro de los límites permitidos, por si este propone algo fuera de rango

        # Generamos el nuevo punto
        x_adv = self.current_x + delta # Sumamos la variación al punto original, y ese es el nuevo valor

        # Clasificación del defensor
        p_attack = self.clf.predict_proba(x_adv.reshape(1, -1))[0, 1] # Adaptamos el dato al formato correcto (reshape(1, -1)), obtenemos la probabilidad (0.48 en vez de 0 o 1) de que el punto sea un ataque (fila 0, columna 1, ataque)
        pred = int(p_attack >= self.threshold)  # Predicción final (0 o 1) según el umbral establecido

        # Recompensa del atacante
        success = int(pred == 0) # Éxito si el punto NO es detectado como ataque

        # Penalización proporcional al tamaño del movimiento realizado (queremos que el agente sea sigiloso)
        penalty = self.penalty * float(np.sum(delta**2)) # Elevamos al cuadrado los cambios (independecia del signo) y los sumamos, representando cuanto cambio ha hecho el agente, y lo multplicamos por el factor de penalización
        reward = success - penalty # Recompensa final 1 punto si tiene éxito menos la penalización por moverse mucho

        # Episodio de un solo paso (por ahora)
        terminated = True # El episodio termina después de un solo paso
        truncated = False # No hay truncamiento (se usa para limitar la longitud del episodio cuando pasan x pasos)

        # Información adicional
        info = {
            "x_orig": self.current_x, # Estado original antes de la acción
            "x_adv": x_adv, # Estado final después de la variación del agente
            "p_attack": p_attack, # Probabilidad de que el defensor clasifique como ataque
            "success": success, # Indicador de éxito del ataque
        }

        # Preparamos siguiente estado (nuevo ataque)
        self.current_x = self._sample_attack() # Escogemos un nuevo ataque aleatorio para el siguiente paso
        next_obs = self.current_x.astype(np.float32) # Convertimos el estado a float32 para que lo entienda el agente

        # Devolvemos la siguiente observación, recompensa, estado de terminado, truncado e información adicional
        return next_obs, reward, terminated, truncated, info

    # Render del entorno
    def render(self):
        pass

    # Cierre del entorno
    def close(self):
        pass
