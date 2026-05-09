# AGENTS.md - TP4: Aprendizaje No Supervisado

> **Sistemas de Inteligencia Artificial 2026**

---

## Estructura del Proyecto

```
sia-tp4/
├── AGENTS.md              # Este archivo
├── README.md              # README público del proyecto
├── config.*               # Archivo(s) de configuración
├── context/               # Contexto teórico para no desviarse de la materia
│   ├── AprendizajeNoSupervisado.md
│   ├── Kohonen.md
│   ├── PCA.md
│   └── OjaSanger.md
├── ej1/                   # Ejercicio 1: Europa (Kohonen + Oja)
│   ├── ...
├── ej2/                   # Ejercicio 2: Patrones (Hopfield)
│   ├── ...
└── shared/                # Material compartido entre ejercicios
    ├── utils.py
    ├── plotting.py
    └── ...
```

- **`context/`**: Contiene la teoría vista en clase. Cualquier implementación debe respetar los algoritmos, definiciones y conceptos allí detallados. Si por algún motivo se considera desviarse de esa teoría, se debe indicar explícitamente al usuario y justificar la decisión.
- **`ej1/`**: Contiene todo lo relacionado al Ejercicio 1 (dataset `europe.csv`, implementación de Kohonen y Oja, gráficos, análisis).
- **`ej2/`**: Contiene todo lo relacionado al Ejercicio 2 (patrones de letras 5×5, modelo de Hopfield, ruido, estados espúreos).
- **`shared/`**: Código reutilizable entre ambos ejercicios (utilidades, visualización, parsers, etc.).

---

## Dependencia con Trabajos Prácticos Anteriores

> **IMPORTANTE:** Este TP **no** requiere perceptrón simple, perceptrón lineal, perceptrón no lineal ni perceptrón multicapa. Los modelos a implementar son exclusivamente de **aprendizaje no supervisado**.
>
> **Si en algún momento surge la necesidad de usar código o conceptos de los TPs anteriores** (perceptrón simple / lineal / no lineal / multicapa), **consultar al usuario antes de proceder**. Preguntar si dispone de los archivos necesarios y si desea reutilizarlos o implementar desde cero.

---

## Consigna del Trabajo Práctico 4

### Ejercicio 1: Europa

El conjunto de datos `europe.csv` corresponde a características económicas, sociales y geográficas de **28 países de Europa**. Las variables son:

- `Country`: Nombre del país.
- `Area`: área.
- `GDP`: producto bruto interno.
- `Inflation`: inflación anual.
- `Life.expect`: expectativa de vida media en años.
- `Military`: presupuesto militar.
- `Pop.growth`: tasa de crecimiento poblacional.
- `Unemployment`: tasa de desempleo.

#### 1.1 Red de Kohonen

Implementar la red de Kohonen y aplicarla para resolver los siguientes problemas:

- Asociar países que posean las mismas características geopolíticas, económicas y sociales.
- Realizar al menos un gráfico que muestre los resultados.
- Realizar un gráfico que muestre las distancias promedio entre neuronas vecinas (Matriz U).
- Analizar la cantidad de elementos que fueron asociados a cada neurona.

#### 1.2 Modelo de Oja

Implementar una red neuronal utilizando la regla de Oja para resolver los siguientes problemas:

- Calcular la primer componente principal para este conjunto de datos.
- Interpretar el resultado de la primer componente.
- Comparar el resultado del ejercicio de Oja con el resultado de calcular la primer componente principal con una librería.

---

### Ejercicio 2: Patrones

#### 2.1 Modelo de Hopfield

Construir patrones de letras del abecedario utilizando **1 y −1** y matrices de **5 × 5**. Por ejemplo, con la matriz:

```
[[ 1,  1,  1,  1,  1],
 [-1, -1, -1,  1, -1],
 [-1, -1, -1,  1, -1],
 [ 1, -1, -1,  1, -1],
 [ 1,  1,  1, -1, -1]]
```

puede dibujarse la letra J.

**a.** Almacenar **4 patrones de letras**. Implementar el modelo de Hopfield para asociar matrices ruidosas de 5×5 con los patrones de las letras almacenadas. Los patrones de consulta deben ser **alteraciones aleatorias** de los patrones originales. Mostrar los resultados que se obtienen en cada paso hasta llegar al resultado final.

**b.** Ingresar un patrón muy ruidoso e identificar un **estado espúreo**.

---

### Entrega

Entregar por Campus:
- La presentación.
- El repositorio con `README.md` y archivo de configuración.
- El **hash del commit**.

---

## Notas para el Agente

- **No asumir** que existen notebooks, scripts previos o datasets en otras rutas fuera de este repositorio.
- Si se necesita un dataset (`europe.csv`), verificar si ya existe en `ej1/` o pedir confirmación al usuario para obtenerlo.
- Priorizar la teoría de los archivos en `context/` sobre implementaciones genéricas encontradas en internet.
- Documentar decisiones de diseño e hiperparámetros elegidos.
