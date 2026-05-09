# Aprendizaje No Supervisado

> **Contexto teórico para implementación**  
> Fuente: Material de clase - Sistemas de Inteligencia Artificial 2026  
> **ADVERTENCIA PARA EL LLM:** Si durante la implementación te desvías de los conceptos, algoritmos o definiciones aquí expuestos, debes indicarlo explícitamente al usuario y justificar la desviación.

---

## 1. ¿Qué es el Aprendizaje No Supervisado?

En el **aprendizaje no supervisado** la variable de respuesta **no es información disponible**. El agente no tiene acceso a etiquetas. Esto lo hace apto para resolver problemas menos definidos que el aprendizaje supervisado, donde las tareas están bien definidas y no cambian mucho en el tiempo (clasificación y regresión).

**Tareas principales del aprendizaje no supervisado:**
- **Clustering** (agrupamiento)
- **Asociación**
- **Reducción de dimensionalidad**

> Dado que no se conoce el valor de verdad, una pregunta clave es: *¿cómo sabemos si los resultados que obtenemos son significativos?* El modelo construye predicciones y estrategias para obtener características o patrones y sacar conclusiones.

---

## 2. Supervisado vs No Supervisado

| Aprendizaje Supervisado | Aprendizaje No Supervisado |
|---|---|
| El agente tiene acceso a etiquetas | La variable de respuesta no está disponible |
| Conocimiento de la variable de respuesta | Puede resolver problemas menos definidos |
| Tareas bien definidas | Clustering, Asociación y Reducción de Dimensionalidad |

---

## 3. Problemas a Resolver

### 3.1 Clustering
Agrupar observaciones de forma tal que el grado de similitud entre miembros de un mismo grupo sea lo más fuerte posible.
- Identificar similitudes entre los datos y asignarlos a un grupo (cluster).
- Implica **definir similitud**.

**Ejemplos de aplicación:**
- **Medicina:** Datos de edad, sexo, peso, colesterol. ¿Hay patrones que indiquen si un paciente responderá bien ante cierto tratamiento?
- **Detección de anomalías:** Identificar outliers dentro de los clusters.
- **Estrategias de marketing:** Segmentación de usuarios para mejorar su experiencia.

### 3.2 Asociación
Encontrar relaciones entre los atributos del conjunto de datos.
- **Memorias Asociativas:** El almacenamiento y recuperación de información por asociación con otros datos.
- **Ejemplo:** Modelo de Hopfield.

**Aplicaciones:**
- Sistemas de recomendación: identificar qué artículos los clientes compran juntos con frecuencia.
- Identificar relaciones entre diferentes síntomas y enfermedades.

### 3.3 Reducción de Dimensionalidad
La reducción de la dimensionalidad proyecta el conjunto de datos en un espacio menor, dejando de lado las características menos relevantes.
- **Ejemplos:** PCA, Autoencoders.
- Pasa de N variables a 2 variables (o menos).

**Aplicaciones:**
- Identificar qué variables afectan más a cada país para luego invertir (educación, agricultura, inflación, etc.).
- Comprimir una imagen y reducir el costo computacional.

---

## 4. Problemas del Aprendizaje Supervisado que el No Supervisado Resuelve

1. **Etiquetas:** Es costoso generar/obtener un gran conjunto de datos etiquetados. El aprendizaje no supervisado **no necesita etiquetas**.
2. **Dimensionalidad:** Cuando hay muchas características (features), es costoso encontrar una buena aproximación. El aprendizaje no supervisado permite la **reducción de dimensionalidad**.
3. **Outliers:** Si se ignoran los outliers, el modelo aprende de ellos y comete más errores. El aprendizaje no supervisado **agrupa los outliers por un lado y el resto de los datos por el otro**.

---

## 5. Modelo General

```
INPUT: Datos no etiquetados
       ↓
MODELO: Aprendizaje No Supervisado
       ↓
OUTPUT: Datos similares agrupados
```

---

## 6. Redes Neuronales Estudiadas en la Materia

En esta materia se estudiarán métodos de aprendizaje no supervisado usando modelos de redes neuronales:

1. **Red de Kohonen** (SOM: Mapas Auto-Organizados)
2. **Red de Hopfield**
3. **Modelo de Oja**
4. **Modelo de Sanger**

> **NOTA IMPORTANTE:** Cada uno de estos modelos tiene su propio archivo de contexto teórico detallado en este mismo directorio.

---

## 7. Bibliografía

1. McKay D.J.C. Hopfield Networks. *Information Theory, Inference and Learning Algorithms*, Cambridge, 2003.
2. Anders Krogh, John Hertz, Richard Palmer. *Introduction to the Theory of Neural Computation*. Addison-Wesley, 1991.
3. T. Kohonen. Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 1(43):59–69, 1982.
4. T. Kohonen. The self-organizing map. *Neurocomputing*, pages 1–6, 1998.
5. Hiran, K. K., Jain, R. K., Lakhwani, K., & Doshi, R. (2021). *Machine Learning: Master Supervised and Unsupervised Learning Algorithms with Real Examples*. BPB Publications.
