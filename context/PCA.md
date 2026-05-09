# Análisis de Componentes Principales (PCA)

> **Contexto teórico para implementación**  
> Fuente: Material de clase - Sistemas de Inteligencia Artificial 2026  
> **ADVERTENCIA PARA EL LLM:** Si durante la implementación te desvías de los conceptos, algoritmos o definiciones aquí expuestos, debes indicarlo explícitamente al usuario y justificar la desviación.

---

## 1. Introducción

El **Análisis de Componentes Principales (PCA)** es una técnica de **reducción de dimensionalidad** que busca eliminar la redundancia en un conjunto de datos.

Si las variables de un conjunto de datos están **muy correlacionadas**, entonces poseen información redundante. El objetivo de PCA es transformar el conjunto de variables original en otro conjunto de variables que:
- Sean **combinaciones lineales** de las anteriores.
- **No estén correlacionadas** entre sí.

Este nuevo conjunto se denomina **Componentes Principales**.

### Historia
- Publicada por **H. Hotelling** en 1933.
- Las primeras versiones se encuentran en los ajustes ortogonales por cuadrados mínimos introducidos por **K. Pearson** en 1901.

---

## 2. Medidas Descriptivas Previas

### 2.1 Media Muestral
Dado un conjunto de datos de una variable:
```
x̄ = (1/m) Σ xi
```

### 2.2 Varianza Muestral
```
s² = (1/(m-1)) Σ (xi - x̄)²
```

### 2.3 Covarianza Muestral
Dado un conjunto de datos con **n variables** y **m observaciones**, la covarianza muestral mide la asociación lineal entre las variables Xi y Xk:
```
sik = (1/(m-1)) Σ (xji - x̄i)(xjk - x̄k)
```

### 2.4 Matriz de Covarianzas
Las covarianzas forman una **matriz simétrica definida positiva** S de dimensión n × n.

### 2.5 Interpretación de la Covarianza Muestral
- **sik > 0**: indica asociación lineal positiva entre los datos de las variables.
- **sik < 0**: indica asociación lineal negativa.
- **sik = 0**: indica que las variables son independientes (en términos lineales).

> La varianza muestral es la covarianza muestral de la variable Xi con ella misma: sii.

### 2.6 Correlación Muestral
Es otra medida de asociación lineal. Es la covarianza muestral con las variables estandarizadas:
```
rik = sik / (si * sk)
```

---

## 3. Objetivo de PCA

Dadas **p variables originales**, se desean encontrar **q < p** variables que sean combinaciones lineales de las p originales, recogiendo la mayor parte de la información o variabilidad de los datos.

> **Si las variables originales no están correlacionadas, entonces no tiene sentido realizar un PCA.**

Se hallan **p componentes principales** en total, pero se seleccionan las q más importantes.

---

## 4. Criterio de Variabilidad

¿Cómo sabemos que la variable "es principal"?
- Se toma la característica que **maximiza la variabilidad**.
- Es un buen factor para diferenciar objetos en un conjunto de datos.

**Ejemplo:**
- Característica 1: Cantidad de ruedas (baja variabilidad).
- Característica 2: Longitud del vehículo (mayor variabilidad) → Mejor para diferenciar.

---

## 5. Transformación

Sea **Xnxp** una matriz con:
- n elementos de la población (observaciones).
- p variables.

Se realiza una transformación de X de forma tal que la **varianza del nuevo conjunto de variables sea máxima**.

### 5.1 Primera Componente Principal
Sea Y₁ la primera componente principal, y **a₁** un vector de cargas (loadings):
```
Y₁ = a₁₁X₁ + a₁₂X₂ + ... + a₁pXp = a₁ᵀX
```

> Nota: si las variables están estandarizadas no es necesario restar la media.

### 5.2 Cargas (Loadings)
El conjunto de componentes principales es una combinación lineal de las variables originales. Se desea encontrar la carga **a₁** tal que la varianza de Y₁ resulte máxima.

Los coeficientes **aij** se denominan **cargas (loadings)**.

**¿Cómo hallar las cargas?**
Son los **autovectores** de la matriz de covarianzas (o correlaciones).

---

## 6. Cálculo de Componentes Principales

Para hallar las componentes principales se deben calcular los **autovectores vi** correspondientes a las cargas.

Entonces las componentes se calculan como:
```
Yi = viᵀX
```

### 6.1 Autovalores
El autovalor **λi** se corresponde con la **varianza de la componente i**.

Ordenando los autovalores de mayor a menor se logra reducir la dimensionalidad tomando los autovectores correspondientes a los primeros q autovalores, que son los que proveen mayor información (en términos de variabilidad).

---

## 7. Covarianza vs Correlación en PCA

### 7.1 Problema de las Escalas
Si alguna de las variables toma valores mayores a las demás, entonces tendrá mayor varianza, pero **no quiere decir que tenga mayor variabilidad**.

Cuando las escalas de medida de las variables son muy distintas, la maximización de la varianza depende decisivamente de estas escalas, y las variables con valores más grandes tendrán más peso en el análisis.

### 7.2 Solución: Estandarización
Para evitar el problema se deben **estandarizar las variables** cuando calculamos las componentes principales. De esta manera las magnitudes de los valores numéricos de las variables originales serán comparables.

### 7.3 Equivalencia
Esto equivale a aplicar el análisis de componentes principales utilizando la **matriz de correlaciones** en lugar de la matriz de covarianzas.

> **Cuando las variables tienen las mismas unidades, ambas alternativas son posibles.**

---

## 8. Procedimiento Completo

1. Construir la matriz X a partir del dataset, poniendo las variables en columnas.
2. **Estandarizar** las variables X (especialmente si tienen escalas distintas).
3. Calcular la matriz de correlaciones Sx.
4. Calcular los **autovalores y autovectores** de Sx (diagonalización).
5. Ordenar los autovalores de mayor a menor.
6. Construir la matriz V con los autovectores correspondientes a los mayores autovalores.
7. Calcular las nuevas variables Y como combinación lineal de las originales: **Y = XV**.

---

## 9. Interpretación de PC1

- Si la carga (coeficiente o loading) de una variable en la componente principal es **positiva**, significa que la variable y la componente tienen una correlación positiva.
- Si la carga es **negativa**, la variable se correlaciona en forma negativa con la primera componente.
- La primera componente representa un **índice** (o una característica) por el cual se pueden ordenar los registros.

**Ejemplo:** Si PC1 es una suma ponderada de gastos familiares con mayor carga en alimentación, salud y educación, entonces ordenar por PC1 equivale a ordenar por capacidad de gasto.

---

## 10. Proporción de Varianza Explicada

La proporción de la varianza explicada por cada componente i es:
```
Proporción_i = λi / (λ₁ + λ₂ + ... + λp)
```

Esto permite decidir cuántas componentes retener (ej: las que sumen al menos el 80-90% de la varianza total).

---

## 11. Bibliografía

- H. Hotelling. Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 1933.
- K. Pearson. On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 1901.
