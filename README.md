# `fkdc`: Distancia de Fermat en Clasificadores de Densidad Nuclear
Lic. Gonzalo Barrera Borla

Dir.: Dr. Pablo Groisman

Tesis en progreso para la Maestría en Estadística Matemática, IC, FCEN-UBA 2022, ningún derecho reservado


## Biblioteca

El núcleo de la tesis, `fkdc.fermat`, se publica como biblioteca compatible con scikit-learn: la distancia de Fermat muestral (`SampleFermatDistance`), el estimador de densidad por núcleos con esa distancia (`FermatKDE`), su par euclídeo exacto (`EuclideanKDE`) y los clasificadores `KDClassifier` y `FermatKNeighborsClassifier`.

```bash
pip install fkdc
```

```python
from sklearn.datasets import make_moons
from fkdc import KDClassifier

X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
clf = KDClassifier(metric="fermat", alpha=2.0, bandwidth=0.3).fit(X, y)
probas = clf.predict_proba(X[:5])
```

`metric` acepta `"fermat"` (con su `alpha`), `"euclidea_exacta"` y las métricas de `sklearn.neighbors.KernelDensity` que no requieren parámetros propios, ya que `metric_params` no se transmite. Para densidades euclídeas exactas conviene `metric="euclidea_exacta"`: el valor por defecto, `"euclidean"`, pasa por `KernelDensity`, cuyo recorrido del árbol devuelve log-densidades erróneas en la cola profunda.

Instalada así, la biblioteca no escribe caché en disco. Para memorizar las distancias entre corridas, basta definir la variable de entorno `FKDC_CACHE` con un directorio; se lee al importar `fkdc`, así que debe definirse antes del primer `import fkdc`.


## Reproducir la tesis

### Instalación

#### Requisitos previos
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes Python)
- [Typst](https://typst.app/) (para compilar los documentos)
- GNU Make

#### Instalar Typst

```bash
# macOS
brew install typst

# o con cargo
cargo install typst-cli
```

Alternativamente, instalar la extensión [Tinymist](https://marketplace.visualstudio.com/items?itemName=myriad-dreamin.tinymist) en VS Code para compilación integrada y vista previa.

#### Configuración del entorno

```bash
# Clonar el repositorio
git clone https://github.com/capitantoto/fermat.git
cd fermat

# Crear entorno virtual e instalar dependencias
uv sync

# Instalar hooks de pre-commit y filtro de notebooks
uv run pre-commit install
uv run nbstripout --install
```

### Compilar documentos

```bash
# Compilar todos los .typ → .pdf (tesis, plan, poster)
make docs

# Compilar solo la tesis
typst compile docs/tesis.typ
```

Para compilación continua durante la edición:

```bash
typst watch docs/tesis.typ
```

Ver `make help` para la lista completa de targets disponibles.

### Bibliografía

#### Zotero

Este proyecto usa [Zotero](https://www.zotero.org/) para gestionar las referencias bibliográficas.

#### Better BibTeX

Para exportar las referencias en formato compatible con Typst, se requiere la extensión [Better BibTeX for Zotero](https://retorque.re/zotero-better-bibtex/).

Descarga e instalación: https://retorque.re/zotero-better-bibtex/installation/

De ser necesario, una copia de la extensión está disponible en [src/zotero-better-bibtex-6.7.240.xpi](https://github.com/capitantoto/fermat/blob/main/src/zotero-better-bibtex-6.7.240.xpi).

#### Exportar referencias

Para generar o actualizar `docs/references.bib`:

1. En Zotero, seleccionar la colección correspondiente a la tesis
2. Clic derecho → "Export Collection..."
3. Formato: "Better BibLaTeX" o "Better BibTeX"
4. Guardar como `docs/references.bib`

Alternativamente, configurar una exportación automática:
1. Clic derecho en la colección → "Export Collection..."
2. Marcar "Keep updated"
3. Seleccionar la ruta `docs/references.bib`

## Uso de IA

Este proyecto utiliza herramientas de IA (Claude, de Anthropic) para tareas de revisión y formato. El contenido intelectual es enteramente del autor. Ver [CLAUDE.md](https://github.com/capitantoto/fermat/blob/main/CLAUDE.md) para más detalles sobre los principios y límites de este uso.
