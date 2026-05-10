# AGENTS.md - TP4: Aprendizaje No Supervisado

> **Sistemas de Inteligencia Artificial 2026**

---

## Estructura del Proyecto

```
sia-tp4/
├── AGENTS.md              # Este archivo
├── README.md              # README público del proyecto
├── requirements.txt       # Dependencias Python (numpy, pandas, matplotlib, sklearn, pyyaml)
├── context/               # Contexto teórico para no desviarse de la materia
│   ├── AprendizajeNoSupervisado.md
│   ├── Kohonen.md
│   ├── PCA.md
│   └── OjaSanger.md
├── ej1/                   # Ejercicio 1: Europa (PCA librería + Oja + Kohonen)
│   ├── config.yaml        # Hiperparámetros y paths (secciones pca, oja, kohonen)
│   ├── data/europe.csv
│   ├── models/            # IMPLEMENTACIÓN: numpy puro (sklearn solo en pca.py)
│   │   ├── pca.py
│   │   ├── oja.py
│   │   └── kohonen.py
│   ├── analysis/          # PLOTS + INTERPRETACIÓN (consume outputs de models/)
│   │   ├── pca.py
│   │   ├── oja.py
│   │   └── kohonen.py
│   ├── main_pca.py        # Entrypoint independiente por bloque
│   ├── main_oja.py
│   ├── main_kohonen.py
│   └── results/{pca,oja,kohonen}/   # Figuras (18 PNG totales)
├── ej2/                   # Ejercicio 2: Patrones (Hopfield) — pendiente de implementar
├── shared/                # Utilidades comunes a los ejercicios
│   ├── preprocessing.py   # ZScoreScaler
│   ├── io_utils.py        # load_europe()
│   ├── plotting.py        # save_fig, paleta
│   └── config_loader.py   # load_yaml()
└── presentacion/
    ├── main.tex           # Beamer (tema Metropolis), 22 frames
    ├── main.pdf
    └── README.md          # cómo compilar
```

- **`context/`**: Contiene la teoría vista en clase. Cualquier implementación debe respetar los algoritmos, definiciones y conceptos allí detallados. Si por algún motivo se considera desviarse de esa teoría, se debe indicar explícitamente al usuario y justificar la decisión.
- **`ej1/`**: Ejercicio 1 (`europe.csv`). Subdividido en `models/` (núcleo algorítmico) y `analysis/` (plots + métricas).
- **`ej2/`**: Ejercicio 2 (patrones de letras 5×5, Hopfield, ruido, estados espúreos). **Pendiente**.
- **`shared/`**: Código reutilizable: cargadores, preprocesamiento, helpers de plotting, lector de YAML.
- **`presentacion/`**: Slides Beamer cubriendo el EJ1 completo.

### Layout `models/` vs `analysis/` (decisión de diseño)

Cada ejercicio se divide en dos capas con responsabilidades estrictas:

- **`models/`**: implementa los algoritmos (PCA con librería, Oja, Kohonen). NO importa matplotlib. Expone clases y funciones que devuelven estructuras de datos puras (numpy arrays, dicts, history lists).
- **`analysis/`**: consume los outputs de `models/` y produce figuras + métricas. Importa matplotlib.

Los `main_*.py` son la única capa que orquesta ambas: cargan datos, llaman a un modelo, le pasan los resultados a `analysis/` para graficar.

Esto permite auditar / testear el código algorítmico aisladamente, reutilizarlo en `ej2/`, y mantener cualquier cambio de visualización fuera del modelo.

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

---

## Decisiones de diseño tomadas durante EJ1

Cosas decididas al implementar el EJ1 que vale la pena registrar para mantener consistencia en EJ2 y entregas futuras:

1. **Formato de configuración: YAML** (`ej1/config.yaml`). Soporta comentarios, consistente con TP3, fácil de versionar. Una sola configuración por ejercicio con secciones por bloque (`pca`, `oja`, `kohonen`).
2. **Tres entrypoints separados** (`main_pca.py`, `main_oja.py`, `main_kohonen.py`) en lugar de un único orchestrator. Permite correr y entregar bloques de forma independiente.
3. **`results/` subdividido por bloque** (`results/pca/`, `results/oja/`, `results/kohonen/`). Cada `main_*.py` escribe solo a su subcarpeta.
4. **Librerías permitidas**: para PCA con librería se usan `sklearn.decomposition.PCA` y `numpy.linalg.eigh` (validación cruzada). Oja y Kohonen se implementan en numpy puro siguiendo la teoría literal.
5. **`ZScoreScaler` reimplementado en `shared/preprocessing.py`** (no copiado del TP3). Son ~10 líneas de numpy; mantener TP4 self-contained justifica el costo.
6. **Convención de signo en PCA**: en cada autovector, se fuerza positiva la componente de mayor `|valor|`. Esto vuelve comparables sklearn vs eigh vs Oja.
7. **Sanity checks en cada `main_*.py`**: asserts mínimos sobre `explained_variance_ratio`, `cos_sim` con la referencia, normas, NaN, etc. No hay test suite formal; los asserts inline cumplen ese rol.
8. **Presentación en LaTeX Beamer** (tema Metropolis), consistente con TP3. Las figuras se leen vía `\graphicspath` desde `ej1/results/`.

---

## Pendiente (Ejercicio 2)

- Construir 4 patrones de letras 5×5 con valores $\{-1, +1\}$.
- Implementar Hopfield: regla de Hebb para almacenar, dinámica asíncrona/síncrona para recuperar, mostrar evolución paso a paso.
- Identificar un estado espúreo con un patrón muy ruidoso.
- Replicar el layout `models/` + `analysis/` dentro de `ej2/`.
- Agregar slides al final de `presentacion/main.tex` o crear `presentacion2/`.
