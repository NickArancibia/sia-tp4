# Presentacion - Ejercicio 1 (Europa)

Slides Beamer (tema Metropolis) que cubren los tres bloques del EJ1: PCA con libreria, Regla de Oja y Red de Kohonen.

## Compilacion

Las figuras se leen directamente de `../ej1/results/{pca,oja,kohonen}/` via `\graphicspath`. Asegurate de haber corrido los tres `main_*.py` antes de compilar.

```bash
cd presentacion
pdflatex main.tex && pdflatex main.tex
```

La doble pasada es por las refs internas (TOC / progress bar).

## Salida

`main.pdf` (incluye 19 frames + portada y cierre).

## Dependencias LaTeX

- `beamer` y tema `metropolis`.
- `inputenc`, `fontenc`, `babel` (espaniol).
- `graphicx`, `amsmath`, `booktabs`, `siunitx`, `xcolor`.

En TeX Live 2023+ todo eso esta incluido por defecto.
