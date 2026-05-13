# Presentacion PCA

Slides Beamer cortas para defender la parte de PCA del `Ejercicio 1`.

## Compilacion

Las figuras se leen desde `../ej1/results/pca/`, asi que conviene tener corrido antes:

```bash
python3 ej1/main_pca.py
```

Luego compilar con doble pasada:

```bash
cd presentacion_pca
pdflatex main.tex
pdflatex main.tex
```

## Contenido

- Estandarizacion vs datos crudos.
- Estructura del dataset via correlaciones.
- Varianza explicada por componente.
- Biplot PC1 vs PC2 con lectura de paises.
