# Regla de Oja y Regla de Sanger

> **Contexto teórico para implementación**  
> Fuente: Material de clase - Sistemas de Inteligencia Artificial 2025/2026  
> **ADVERTENCIA PARA EL LLM:** Si durante la implementación te desvías de los conceptos, algoritmos o definiciones aquí expuestos, debes indicarlo explícitamente al usuario y justificar la desviación.

---

## 1. Introducción

Algunos modelos de redes neuronales permiten calcular las **componentes principales** en forma **iterativa**, en lugar de resolver el problema de autovalores y autovectores de manera directa (como en PCA clásico).

**Ventaja:** Reduce el costo computacional (especialmente cuando la cantidad de registros es muy grande, o hay muchas variables).

**Desventaja:** Si el dataset tiene muchas variables, interpretarlo en una sola componente puede hacerse perder información al reducir tan drásticamente la dimensión.

---

## 2. Perceptrón Lineal Simple

Dados n datos de entrada x₁, ..., xₙ, una red neuronal calcula la salida:
```
y = Σ wi * xi = wᵀx
```

En aprendizaje supervisado, actualizaríamos los pesos comparando y con el valor de verdad.

---

## 3. Aprendizaje Hebbiano

Regla de Aprendizaje Hebbiano para Aprendizaje No Supervisado:

Dados m datos de entrada x¹, ..., xᵐ, una red neuronal calcula la salida y.

**Como no se conoce el valor de verdad**, la regla de actualización Hebbiana es:
```
Δw = η * y * x
```

### Problema de Convergencia
Oja demostró que si la red anterior converge, el vector de pesos resultante w_final sería un punto sobre la dirección de máxima variación de los datos (la primer componente principal).

**Pero...** El problema es que **no converge** porque el ||w|| va aumentando en cada paso y se hace tan grande que produce que el algoritmo sea **inestable**.

---

## 4. Regla de Oja

**Autor:** Dr. Erkki Oja, Helsinki University of Technology, Finland.

### 4.1 Fórmula de Actualización
Utilizando una aproximación por el polinomio de Taylor, se deriva la regla de Oja:
```
Δw = η * y * (x - y * w)
```

O equivalentemente:
```
w^(t+1) = w^t + η * y * (x - y * w^t)
```

Donde:
- **y = wᵀx** es la salida de la neurona.
- **η** es la tasa de aprendizaje.
- El término **(x - y*w)** introduce una **normalización implícita** que evita que los pesos crezcan indefinidamente.

### 4.2 Convergencia
- **||w|| se mantiene acotado**. Tiende a 1.
- Luego de varias iteraciones, el método converge al **autovector correspondiente al mayor autovalor** de la matriz de correlaciones de los datos de entrada.
- Con este vector w_final se construye la **primera componente principal**:
  ```
  y1 = a1*x1 + ... + an*xn
  ```
  Los ai forman el primer autovector, que sería el w_final.

### 4.3 Tasa de Aprendizaje
Para asegurar la convergencia, se debe cumplir:
```
η < 1 / (2 * λ_max)
```

Como no conocemos el autovalor máximo, es mejor:
1. **Estandarizar todas las variables de entrada**.
2. Comenzar con **η = 0.5** y disminuirla, o usar **η = 10⁻³**.

---

## 5. Implementación de la Regla de Oja

### 5.1 Arquitectura
- Es un **perceptrón simple** con una capa de salida (una sola neurona).

### 5.2 Inicialización de Pesos
- Distribución uniforme entre 0 y 1.

### 5.3 Tasa de Aprendizaje
- **η(0) = 0.5** y disminuye con el tiempo, o
- **η = 10⁻³** constante pequeña.

### 5.4 Algoritmo
```
input: X datos de dimensión N con media 0; η, w inicial con distribución uniforme entre 0 y 1

for epoch in #epochs:
    for i = 1 to N:
        y = inner(xi, w)          # calcular el output
        w += η * y * (xi - y * w)  # actualizar según regla de Oja

return(w)
```

> **Importante:** Los datos deben tener **media 0** (centrados).

---

## 6. Regla de Sanger

La **Regla de Sanger** es una extensión de la regla de Oja.

### 6.1 Características Principales
- Converge a la **matriz de autovectores** de la matriz de covarianzas de los datos.
- Permite encontrar **todas las componentes principales** (k autovectores), no solo la primera.

### 6.2 Arquitectura
- Es un perceptrón lineal con **múltiples neuronas de salida** (una por cada componente principal que se desea extraer).

### 6.3 Fórmula de Actualización
Para la neurona j (la j-ésima componente principal):
```
Δwj = η * yj * (x - Σ_{l=1 to j} yl * wl)
```

O equivalentemente:
```
wj^(t+1) = wj^t + η * yj * (x - Σ_{l=1 to j} yl * wl^t)
```

Donde:
- **yj = wjᵀx** es la salida de la neurona j.
- La sumatoria Σ_{l=1 to j} yl * wl introduce una **deflación** que fuerza a cada neurona a capturar la siguiente dirección de máxima varianza, ortogonal a las anteriores.

### 6.4 Interpretación
- La primera neurona (j=1) aplica la regla de Oja estándar.
- La segunda neurona (j=2) resta la proyección sobre la primera componente, extrayendo la siguiente dirección de máxima varianza ortogonal.
- Y así sucesivamente para las k neuronas.

---

## 7. Comparación entre Oja y Sanger

| Característica | Regla de Oja | Regla de Sanger |
|---|---|---|
| Componentes principales | Solo la primera (PC1) | Las k primeras (PC1, PC2, ..., PCk) |
| Neuronas de salida | 1 | k |
| Convergencia | Autovector del mayor autovalor | Matriz completa de autovectores |
| Base del algoritmo | Normalización implícita | Deflación progresiva |

---

## 8. Bibliografía

1. E. Oja. A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15:267–273, 1982.
2. T. D. Sanger. Optimal unsupervised learning in a single-layer linear feedforward neural network. *Neural Networks*, 2(6):459–473, 1989.
3. T. Kohonen. The self-organizing map. *Neurocomputing*, pages 1–6, 1998.
