# Modelo de Kohonen (SOM - Self-Organizing Map)

> **Contexto teórico para implementación**  
> Fuente: Material de clase - Sistemas de Inteligencia Artificial 2026  
> **ADVERTENCIA PARA EL LLM:** Si durante la implementación te desvías de los conceptos, algoritmos o definiciones aquí expuestos, debes indicarlo explícitamente al usuario y justificar la desviación.

---

## 1. Introducción

Las **Redes de Kohonen** son un modelo de **Aprendizaje No Supervisado**.
- **No existe información externa** que indique si la red neuronal está operando correcta o incorrectamente.
- Durante el proceso de aprendizaje **descubre por sí misma** regularidades (patrones) en los datos de entrada.
- También conocidas como **SOM: Mapas Auto-Organizados** (Self-Organizing Maps).

**Autor:** Teuvo Kohonen, investigador finlandés. Publicó su idea por primera vez en 1982 [1] y siguió trabajando en el tema [2].

---

## 2. Arquitectura

- Las neuronas están conectadas **consigo mismas positivamente**.
- Las neuronas están conectadas **con las neuronas vecinas** (radio = R).
- **INPUT:** elemento del training set (vector n-dimensional).
- **OUTPUT:** grilla/mapa (M) de dimensiones k × k.

La red pasa de un **espacio multidimensional** a un **espacio bidimensional**.

### 2.1 Grilla
- Dimensión **K × K**.
- Si los datos de entrada tienen dimensión **N**, cada neurona de la grilla tiene **N conexiones**.
- Cada neurona de salida j ∈ {1, ..., k²} tiene asociado un **vector de pesos** Wj = (wj1, ..., wjn), que es el representante de esa neurona.
- Wj tiene la **misma dimensión** que los datos de entrada.

### 2.2 Tipos de Grilla
- **Grilla Rectangular**
- **Grilla Hexagonal**

### 2.3 Vecindario
Se define un **radio R**:
- **4-vecinos** → R = 1
- **8-vecinos** → R = √2

**Propiedad topológica:** Las entradas similares se concentran en neuronas cercanas. Neuronas vecinas contienen datos con algún grado de similitud entre sí.

---

## 3. Aprendizaje Competitivo

Las neuronas compiten unas con otras. El objetivo es que finalmente **sólo una de las neuronas de salida se active** (la neurona ganadora). Las demás son forzadas a valores de respuesta mínimos.

### 3.1 Neurona Ganadora
A lo largo del tiempo (épocas), algunas unidades toman un nivel de activación mayor mientras que el nivel de las demás se anula.

Dada la unidad de entrada **x**, la neurona que tenga el vector de pesos **w** "más parecido" a x será la ganadora.

> Esto implica una **clasificación**: las entradas parecidas van hacia la misma neurona.

### 3.2 Objetivo
Agrupar los datos que se introducen en la red. El mapa mostrará un agrupamiento donde:
- Las informaciones similares son clasificadas formando parte de la misma categoría o grupo.
- Deben activar la **misma neurona de salida**.

---

## 4. Estandarización de Datos

> **IMPORTANTE:** Es fundamental estandarizar/normalizar todos los vectores de entrada antes de entrenar la red.

### 4.1 Feature Scaling (Min-Max)
Se utiliza para escalar los datos dentro un intervalo [a, b]:

```
x' = a + (x - min(x)) * (b - a) / (max(x) - min(x))
```

Entre [0, 1]:
```
x' = (x - min(x)) / (max(x) - min(x))
```

### 4.2 Estandarización (Z-Score)
Para cada variable Xi con n registros:
- Media: x̄i = (1/n) Σj xji
- Desvío estándar: σi = √[(1/n) Σj (xji - x̄i)²]
- Variable estandarizada: x̃ji = (xji - x̄i) / σi

### 4.3 Unit Length Scaling
Dividir por la norma 2:
```
x' = x / ||x||₂
```

---

## 5. Algoritmo de Kohonen

### 5.1 Inicialización
1. **Xp** = {x₁ᵖ, ..., xₙᵖ}, p = 1, ..., P son los registros de entrada con dimensión n.
2. Definir la cantidad de neuronas de salida: **k × k**.
3. Inicializar los pesos **Wj**, j = 1, ..., k², cada Wj = (wj₁, ..., wjn):
   - Con valores aleatorios con distribución uniforme.
   - **Preferible:** Con ejemplos al azar del conjunto de entrenamiento (evita unidades muertas).
4. Seleccionar un tamaño de entorno inicial con radio **R(0)**.
5. Seleccionar la tasa de aprendizaje inicial **η(0) < 1**.

### 5.2 Iteración i
1. Seleccionar un registro de entrada **Xp**.
2. Encontrar la **neurona ganadora k̂** que tenga el vector de pesos Wk̂ más cercano a Xp.
   - Se define una **medida de similitud d**:
   ```
   Wk̂ = arg min {d(Xp, Wj)}  para 1 ≤ j ≤ N
   ```
3. Actualizar los pesos de las neuronas vecinas según la **regla de Kohonen**.

### 5.3 Regla de Kohonen (Iteración i, paso 3)
Está definido por el radio en la iteración, **R(i)**. Se actualiza el vecindario:
```
Nk̂(i) = {n / ||n - nk̂|| < R(i)}
```
Donde:
- nk̂ es la neurona ganadora
- n es una neurona
- Nk̂(i) es el vecindario

**Notas sobre R(i):**
- R(0) es un dato de entrada.
- R(i) → 1 cuando i → ∞ (decrece con el tiempo).
- También puede permanecer constante durante todo el proceso.

### 5.4 Actualización de Pesos
```
Si j ∈ Nk̂(i):   Wj^(i+1) = Wj^i + η(i) * (Xp - Wj^i)
Si j ∉ Nk̂(i):   Wj^(i+1) = Wj^i
```

Donde **η(i) → 0** cuando i aumenta. Por ejemplo: η(i) = 1/i.

### 5.5 Convergencia
¿Por qué converge?
```
Wk̂^(i+1) - Xp = Wk̂^i + η(i)(Xp - Wk̂^i) - Xp
               = (1 - η(i))(Wk̂^i - Xp)
```

Entonces:
```
||Wk̂^(i+1) - Xp|| ≤ ||Wk̂^i - Xp||
```

Los pesos se parecen progresivamente a los datos de entrada.

---

## 6. Medidas de Similitud (Funciones de Propagación)

### 6.1 Distancia Euclídea
```
Wk̂ = arg min {||Xp - Wj||}  para 1 ≤ j ≤ N
```

### 6.2 Exponencial
```
Wk̂ = arg min {e^(-||Xp - Wj||²)}  para 1 ≤ j ≤ N
```

> **IMPORTANTE:** Estandarizar todos los vectores antes de aplicar la medida.

---

## 7. Inicialización - Consideraciones Prácticas

### 7.1 Valores Iniciales de los Pesos
- **Aleatorios:** Problema → algunas unidades quedan lejos de los valores iniciales y nunca ganan. Se denominan **unidades muertas**.
- **Muestras de datos:** Es mejor inicializar los pesos con muestras del conjunto de entrenamiento para evitar unidades muertas.

### 7.2 Cantidad Total de Iteraciones
En función de la cantidad de neuronas de entrada (N).  
**Ejemplo:** 500 * N.

### 7.3 Radio del Vecindario
- **Decreciente:** R(0) puede ser el tamaño total de la grilla y va decreciendo hasta llegar a R = 1, donde solamente se actualizan las neuronas vecinas pegadas.
- **Constante:** También puede mantenerse constante durante todo el proceso.

---

## 8. Visualización de Resultados

### 8.1 Matriz de Coordenadas
Las neuronas de salida forman una matriz. Se puede ver en qué coordenadas se encuentra la neurona asociada a cada ejemplo de entrenamiento.

### 8.2 Conteo de Registros por Neurona
Contar la cantidad de registros que van a cada neurona (heatmap de densidad).

### 8.3 Matriz U (Unified Distance Matrix)
Para cada neurona se calcula el promedio de la distancia euclídea entre:
- El vector de pesos de la neurona.
- El vector de pesos de las neuronas vecinas.

Si el método funciona, deberían observarse **distancias pequeñas** entre vecinos.

### 8.4 Observación de una Variable
Se observa el valor promedio de una sola variable en cada neurona (mapa de calor por componente).

---

## 9. Ventajas y Desventajas

| Ventajas | Desventajas |
|---|---|
| Se aplica en casos donde el conjunto de datos no está etiquetado | Si el conjunto de variables es muy grande puede ser difícil asociarlo con un conjunto bidimensional |
| Pasa de espacio multidimensional a bidimensional | Solo puede realizarse con variables numéricas |
| Puede ser más rápida que el perceptrón multicapa | No hay un criterio demostrado para definir el tamaño de la grilla |

---

## 10. Bibliografía

1. T. Kohonen. Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 1(43):59–69, 1982.
2. T. Kohonen. The self-organizing map. *Neurocomputing*, pages 1–6, 1998.
