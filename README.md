# `fkdc`: Distancia de Fermat en Clasificadores de Densidad Nuclear
Lic. Gonzalo Barrera Borla

Dir.: Dr. Pablo Groisman

Tesis en progreso para la Maestría en Estadística Matemática, IC, FCEN-UBA
2022, ningún derecho reservado


## Instalación

### Requisitos previos
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes Python)
- [Typst](https://typst.app/) (para compilar la tesis)

### Configuración del entorno

```bash
# Clonar el repositorio
git clone https://github.com/gonzalobb/fkdc.git
cd fkdc

# Crear entorno virtual e instalar dependencias
uv sync --all-extras

# Instalar hooks de pre-commit
uv run pre-commit install
```

## Compilar la tesis

```bash
typst compile docs/tesis.typ docs/tesis.pdf
```

Para compilación continua durante la edición:

```bash
typst watch docs/tesis.typ docs/tesis.pdf
```

## Bibliografía

### Zotero

Este proyecto usa [Zotero](https://www.zotero.org/) para gestionar las referencias bibliográficas.

### Better BibTeX

Para exportar las referencias en formato compatible con Typst, se requiere la extensión [Better BibTeX for Zotero](https://retorque.re/zotero-better-bibtex/).

Descarga e instalación: https://retorque.re/zotero-better-bibtex/installation/

### Exportar referencias

Para generar o actualizar `bib/references.bib`:

1. En Zotero, seleccionar la colección correspondiente a la tesis
2. Clic derecho → "Export Collection..."
3. Formato: "Better BibLaTeX" o "Better BibTeX"
4. Guardar como `bib/references.bib`

Alternativamente, configurar una exportación automática:
1. Clic derecho en la colección → "Export Collection..."
2. Marcar "Keep updated"
3. Seleccionar la ruta `bib/references.bib`

## Recursos
- [Classification with Fermat](https://dl.dropboxusercontent.com/s/ogzs6olgberc3lx/Classification%20with%20Fermat.ipynb)
- [Density estimation with Fermat](./Density estimation with Fermat.zip)
- [Fermat](https://www.aristas.com.ar/fermat/fermat.html), librería en python para el cómputo de la distancia homónima
