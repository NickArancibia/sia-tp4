# SIA-TP4 - Aprendizaje No Supervisado

Trabajo Practico 4 de Sistemas de Inteligencia Artificial (ITBA, 2026).

Implementacion completa del **Ejercicio 1 (Europa)** con tres tecnicas de aprendizaje no supervisado aplicadas sobre `europe.csv` (28 paises x 7 variables):

- **PCA con libreria** (sklearn + numpy.linalg.eigh) - reduccion de dimensionalidad por autovalores.
- **Regla de Oja** - extraccion iterativa de PC1 sin resolver el problema de autovalores.
- **Red de Kohonen / SOM** - mapeo no lineal a una grilla 2D conservando topologia.

El **Ejercicio 2 (Hopfield con patrones de letras)** queda para una entrega posterior.

---

## Estructura del repositorio

```
sia-tp4/
├── AGENTS.md                       # convenciones del proyecto (para agentes IA)
├── README.md                       # este archivo
├── requirements.txt                # dependencias Python
├── context/                        # apuntes teoricos de referencia
├── material/                       # enunciado y clases (gitignored)
├── shared/                         # utilidades comunes a los 3 bloques
│   ├── preprocessing.py            # ZScoreScaler
│   ├── io_utils.py                 # load_europe()
│   ├── plotting.py                 # save_fig, paleta
│   └── config_loader.py            # load_yaml()
├── ej1/
│   ├── config.yaml                 # hiperparametros PCA + Oja + Kohonen
│   ├── data/europe.csv             # dataset
│   ├── models/                     # implementacion (numpy puro)
│   │   ├── pca.py
│   │   ├── oja.py
│   │   └── kohonen.py
│   ├── analysis/                   # graficos e interpretacion
│   │   ├── pca.py
│   │   ├── oja.py
│   │   └── kohonen.py
│   ├── main_pca.py                 # entrypoint PCA
│   ├── main_oja.py                 # entrypoint Oja
│   ├── main_kohonen.py             # entrypoint Kohonen
│   └── results/{pca,oja,kohonen}/  # figuras generadas
└── presentacion/
    ├── main.tex                    # slides Beamer (tema Metropolis)
    ├── main.pdf                    # presentacion compilada
    └── README.md                   # como compilar
```

La separacion `models/` vs `analysis/` es estricta: el codigo de `models/` no
importa matplotlib y solo expone estructuras de datos; `analysis/` produce
figuras consumiendo esos resultados. Cada `main_*.py` orquesta ambas capas.

---

## Setup

Requiere Python 3.10+ y LaTeX (TeX Live 2023+) si se quiere compilar la
presentacion.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencias: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyyaml`.

---

## Como correr

Los tres entrypoints son independientes. Cada uno lee la seccion correspondiente
de `ej1/config.yaml`.

```bash
# PCA con libreria (sub-entrega prioritaria)
python3 ej1/main_pca.py
# -> 8 figuras en ej1/results/pca/

# Regla de Oja: PC1 iterativa + comparacion con sklearn
python3 ej1/main_oja.py
# -> 5 figuras en ej1/results/oja/

# Red de Kohonen / SOM
python3 ej1/main_kohonen.py
# -> 5 figuras en ej1/results/kohonen/

# Presentacion (despues de correr los anteriores)
cd presentacion && pdflatex main.tex && pdflatex main.tex
```

Cada main imprime un resumen con los hiperparametros, metricas y sanity checks.

---

## Hallazgos principales

### PCA
- **PC1 (46% de la varianza)** captura un eje de **desarrollo / bienestar**:
  $+$GDP, $+$Life.expect, $+$Pop.growth, $-$Inflation, $-$Unemployment.
- Ranking de paises: Luxembourg, Switzerland, Norway al tope; Ukraine,
  Bulgaria, Latvia al fondo.
- **PC2 (17%)** captura **crisis fiscal**: $+$Military, $+$Unemployment
  vs $-$Inflation. Separa a Grecia / Spain del resto.
- Validacion cruzada sklearn vs `np.linalg.eigh(corrcoef)`: cosine similarity
  $= 1.0$ en las 7 componentes.

### Oja
- Hiperparametros: $\eta = 10^{-3}$, 200 epocas, $w_0 \sim \mathcal{U}(0,1)$ normalizado.
- Convergencia: `cos_sim(w_oja, PC1_sklearn) = 0.99999943`.
- `Spearman ranking = 1.0` --- mismo orden exacto que la libreria.
- Max abs error en cargas: $0.00126$. RMSE: $0.00066$.

### Kohonen
- Grilla $4 \times 4$, 500 epocas, $\eta: 0.5 \to 0.01$, $R: 2 \to 1$.
- 14 de 16 neuronas reciben paises; 2 vacias en la zona-frontera.
- Mapa topologico coherente con PCA: paises nordicos/centrales abajo,
  este/mediterraneo arriba, Ucrania en una esquina aislada (distancia
  $2.93$ en la Matriz U).

---

## Configuracion

Toda la configuracion (paths, hiperparametros) esta en `ej1/config.yaml`.
Se eligio YAML por consistencia con el TP3 y porque soporta comentarios.

Modificar valores en config.yaml y volver a correr `main_*.py` ajusta los
resultados sin tocar codigo.

---

## Hash del commit

Para la entrega: ejecutar `git log --oneline -1` despues del commit final.
