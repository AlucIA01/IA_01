import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
import json
import os

# Configuración global
class ConfiguracionBiologica:
    def __init__(self):
        self.SEED = 42
        self.NUM_NEURONAS_INICIALES = 20
        self.NUM_PASOS_SIMULACION = 50000
        self.PROB_NUEVA_CONEXION = 0.1
        self.PROB_PODA = 0.001
        self.LEARNING_RATE = 0.01
        self.THRESHOLD_ACTIVATION = 0.5
        self.DECAY_FACTOR = 0.9  # Factor de decaimiento para el estado continuo
        self.STDP_DELTA = 0.01  # Delta para ajustar las conexiones con STDP
        self.PROB_ACTIVACION_ESPONTANEA = 0.01  # Probabilidad de activación espontánea

config = ConfiguracionBiologica()

class NeuronaBiologica:
    def __init__(self, id, tipo):
        self.id = id
        self.tipo = tipo
        self.conexiones = {}
        self.potencial = 0.0
        self.umbral = random.uniform(0.3, 0.7)
        self.uso = 0
        self.estado = 0.0  # Estado continuo
        self.memoria_largo_plazo = {}  # Memoria a largo plazo para conexiones

    def recibir_señal(self, señal, intensidad):
        self.potencial += señal * intensidad

    def actualizar(self):
        # Decaimiento natural del potencial
        self.estado *= config.DECAY_FACTOR
        
        # Activación espontánea adaptativa
        if random.random() < config.PROB_ACTIVACION_ESPONTANEA * (1 - self.estado):
            self.potencial += random.uniform(0.1, 0.5)
        
        if self.potencial >= self.umbral:
            self.uso += 1
            return self.disparar()
        self.potencial *= 0.9  # Decaimiento del potencial
        return None

    def disparar(self):
        señales = []
        for neuron_id, intensidad in self.conexiones.items():
            señales.append((neuron_id, intensidad))
            # Memoria a largo plazo con decaimiento
            self.memoria_largo_plazo[neuron_id] = self.memoria_largo_plazo.get(neuron_id, 0) * 0.95 + 1
        self.potencial = 0  # Reseteo del potencial
        return señales
    
    def adaptar_conexion(self, conexion_id, delta_t):
        if conexion_id in self.conexiones:
            if delta_t > 0:  # Si la neurona pre dispara antes que la post
                self.conexiones[conexion_id] += config.STDP_DELTA * delta_t  # LTP
            else:  # Si la neurona post dispara antes que la pre
                self.conexiones[conexion_id] -= config.STDP_DELTA * abs(delta_t)  # LTD
            self.conexiones[conexion_id] = max(0, min(self.conexiones[conexion_id], 1))

class RedNeuronalBiologica:
    def __init__(self, config):
        self.config = config
        self.neuronas = {}
        self.grafo = nx.Graph()
        self.pasos_simulados = 0
        self.precision_por_paso = []  # Lista para almacenar la precisión en cada paso
        self.crear_red_inicial()

    def crear_red_inicial(self):
        for i in range(self.config.NUM_NEURONAS_INICIALES):
            tipo = random.choice(['excitadora', 'inhibidora'])
            neurona = NeuronaBiologica(i, tipo)
            self.neuronas[i] = neurona
            self.grafo.add_node(i, tipo=tipo)
        
        # Asegurar conexiones iniciales
        for _ in range(self.config.NUM_NEURONAS_INICIALES * 2):
            self.conectar_neuronas()

    def conectar_neuronas(self):
        if len(self.neuronas) < 2:
            return
        neurona = random.choice(list(self.neuronas.values()))
        targets = [n for n in self.neuronas.values() if n.id != neurona.id and neurona.id not in n.conexiones]
        if targets:
            target = random.choice(targets)
            intensidad = random.uniform(0.1, 1) if neurona.tipo == 'excitadora' else -random.uniform(0.1, 0.5)
            neurona.conexiones[target.id] = intensidad
            self.grafo.add_edge(neurona.id, target.id)
            # Retroalimentación interna: conectar salida a neuronas ocultas
            if random.random() < 0.5:
                target.conexiones[neurona.id] = intensidad

    def paso(self):
        señales = []
        for neurona in self.neuronas.values():
            if random.random() < self.config.PROB_NUEVA_CONEXION:
                self.conectar_neuronas()
            resultado = neurona.actualizar()
            if resultado:
                señales.extend(resultado)
        
        for id_destino, intensidad in señales:
            if id_destino in self.neuronas:
                self.neuronas[id_destino].recibir_señal(1, intensidad)
        
        if random.random() < self.config.PROB_PODA:
            self.poda()
        
        self.pasos_simulados += 1

    def entrenar(self, inputs, expected_output):
        # Asignar entradas a las primeras neuronas
        for i, input_value in enumerate(inputs):
            if i in self.neuronas:  # Verificación para evitar KeyError
                self.neuronas[i].potencial = input_value

        # Propagar señales
        for _ in range(10):
            self.paso()

        # Obtener salida y calcular error
        salida_actual = self.obtener_salida()
        error = expected_output[0] - salida_actual[0]

        # Ajustar conexiones basado en el error
        for neurona in self.neuronas.values():
            for conexion_id in neurona.conexiones:
                delta = self.config.LEARNING_RATE * error * neurona.potencial
                neurona.adaptar_conexion(conexion_id, delta)
        
        # Ajuste dinámico de la tasa de aprendizaje
        if self.pasos_simulados % 10000 == 0:
            self.config.LEARNING_RATE *= 0.9  # Disminuir la tasa de aprendizaje con el tiempo

    def simular_y_entrenar(self):
        aciertos = 0
        for paso in range(self.config.NUM_PASOS_SIMULACION):
            self.paso()
            
            if paso % 100 == 0:
                x = random.random()
                y = random.random()
                entrada = [x, y]
                salida_deseada = [1] if x + y > self.config.THRESHOLD_ACTIVATION else [0]
                
                self.entrenar(entrada, salida_deseada)
                
                # Evaluar rendimiento
                salida_actual = self.obtener_salida()
                if (salida_actual[0] > 0.5) == (salida_deseada[0] == 1):
                    aciertos += 1
                
                if paso % 1000 == 0:
                    precision = aciertos / 10  # Precisión en las últimas 1000 iteraciones
                    self.precision_por_paso.append(precision)
                    print(f"Paso {paso}, Precisión: {precision:.2f}%")
                    aciertos = 0

    def poda(self):
        if not self.neuronas:
            return
        neurona = random.choice(list(self.neuronas.values()))
        if neurona.conexiones:
            conexion_id = min(neurona.conexiones, key=neurona.conexiones.get)
            del neurona.conexiones[conexion_id]
            if self.grafo.has_edge(neurona.id, conexion_id):
                self.grafo.remove_edge(neurona.id, conexion_id)

    def obtener_salida(self):
        return [self.neuronas[max(self.neuronas.keys())].potencial]

    def visualizar_red(self):
        # Verificar que las listas node_sizes y node_colors tengan la longitud correcta
        uso_max = max(neurona.uso for neurona in self.neuronas.values())
        if uso_max == 0:
            uso_max = 1

        node_sizes = []
        node_colors = []

        for nodo in self.grafo.nodes():
            if nodo in self.neuronas:
                neurona = self.neuronas[nodo]
                node_sizes.append(500 * (neurona.uso / uso_max + 0.1))
                node_colors.append(cm.coolwarm(neurona.uso / uso_max))
            else:
                # Si por alguna razón el nodo no tiene una neurona asociada, usar valores predeterminados
                node_sizes.append(100)
                node_colors.append(cm.coolwarm(0))

        pos = nx.spring_layout(self.grafo)
        plt.figure(figsize=(12, 8))
        nx.draw(self.grafo, pos, with_labels=True, node_color=node_colors, 
                node_size=node_sizes, font_size=8, font_weight='bold')
        plt.title("Estructura Final de la Red Neuronal Biológica")
        plt.show()


    def visualizar_progreso_entrenamiento(self):
        # Visualizar la precisión a lo largo del tiempo
        plt.figure(figsize=(10, 5))
        plt.plot(self.precision_por_paso)
        plt.xlabel('Paso de Simulación (x1000)')
        plt.ylabel('Precisión (%)')
        plt.title('Progreso del Entrenamiento de la Red Neuronal')
        plt.show()

    def probar_red(self, inputs):
        # Asignar entradas a las primeras neuronas
        for i, input_value in enumerate(inputs):
            if i in self.neuronas:  # Verificación para evitar KeyError
                self.neuronas[i].potencial = input_value

        # Propagar señales en la red
        for _ in range(10):
            self.paso()
        
        # Obtener la salida después de la propagación
        salida = self.obtener_salida()
        return salida

    def guardar_estado(self, filename="red_estado.json"):
        try:
            estado = {
                "config": vars(self.config),
                "neuronas": {id: {"tipo": neurona.tipo, "conexiones": neurona.conexiones, 
                                  "potencial": neurona.potencial, "uso": neurona.uso,
                                  "memoria_largo_plazo": neurona.memoria_largo_plazo} 
                             for id, neurona in self.neuronas.items()},
                "pasos_simulados": self.pasos_simulados
            }
            with open(filename, "w") as file:
                json.dump(estado, file)
            print(f"Estado guardado en {filename}")
        except IOError as e:
            print(f"Error al guardar el estado de la red: {e}")

    def cargar_estado(self, filename="red_estado.json"):
        if not os.path.exists(filename):
            print(f"Archivo {filename} no encontrado.")
            return
        try:
            with open(filename, "r") as file:
                estado = json.load(file)
            self.config = ConfiguracionBiologica()
            for key, value in estado["config"].items():
                setattr(self.config, key, value)
            self.neuronas = {}
            for id, datos in estado["neuronas"].items():
                neurona = NeuronaBiologica(id, datos["tipo"])
                neurona.conexiones = datos["conexiones"]
                neurona.potencial = datos["potencial"]
                neurona.uso = datos["uso"]
                neurona.memoria_largo_plazo = datos["memoria_largo_plazo"]
                self.neuronas[id] = neurona
            self.pasos_simulados = estado["pasos_simulados"]
            print(f"Estado cargado desde {filename}")
        except IOError as e:
            print(f"Error al cargar el estado de la red: {e}")
        except json.JSONDecodeError as e:
            print(f"Error al decodificar el archivo JSON: {e}")

def probar_red_normalizada(red, inputs):
    # Normalizar entradas considerando el rango de los datos de entrenamiento
    inputs_normalizados = [x / 30.0 for x in inputs]  # Suponiendo que el valor máximo durante el entrenamiento fue 30
    salida = red.probar_red(inputs_normalizados)
    # Cualquier activación mayor que el threshold define la neurona activada como 1, sino 0
    resultado_binario = 1 if salida[0] > 0.5 else 0
    return resultado_binario, sum(inputs)

def main():
    config = ConfiguracionBiologica()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    red = RedNeuronalBiologica(config)
    
    # Cargar estado si existe
    red.cargar_estado()

    print("Iniciando simulación y entrenamiento.")
    red.simular_y_entrenar()
    
    # Guardar estado de la red
    red.guardar_estado()

    # Evaluación final
    pruebas = 1000
    correctas = 0
    for _ in range(pruebas):
        x = random.random()
        y = random.random()
        entrada = [x, y]
        resultado_esperado = x + y > config.THRESHOLD_ACTIVATION
        red.entrenar(entrada, [1] if resultado_esperado else [0])
        salida = red.obtener_salida()
        if (salida[0] > 0.5) == resultado_esperado:
            correctas += 1

    print(f"Precisión final en la clasificación: {correctas/pruebas * 100:.2f}%")

    # Visualización del grafo
    red.visualizar_red()

    # Visualizar progreso del entrenamiento
    red.visualizar_progreso_entrenamiento()
    
    # Probar la red con entradas específicas
    while True:
        try:
            x = float(input("Ingrese el valor de x: "))
            y = float(input("Ingrese el valor de y: "))
            entrada = [x, y]
            resultado_binario, valor_suma = probar_red_normalizada(red, entrada)
            print(f"Salida de la red para la entrada {entrada}: {resultado_binario} (Suma: {valor_suma})")
        except ValueError:
            print("Entrada inválida. Por favor ingrese números reales.")
            continue

if __name__ == "__main__":
    main()
