# ========================
# Directorios
# ========================
CONFIGS_DIR  ?= configs
TARGETS_DIR  ?= infos
DOCS_DIR     ?= docs
DOCS_DATA    ?= $(DOCS_DIR)/data
DOCS_IMG     ?= $(DOCS_DIR)/img

# ========================
# Fuentes
# ========================
CONFIGS := $(wildcard $(CONFIGS_DIR)/*.yaml)
TARGETS := $(patsubst $(CONFIGS_DIR)/%, $(TARGETS_DIR)/%, $(CONFIGS:.yaml=.pkl))

# Documentos .typ compilables (poster y seminario excluidos: dependencias faltantes)
TYP_DOCS     := $(DOCS_DIR)/tesis.typ $(DOCS_DIR)/plan.typ
TYP_PDFS     := $(TYP_DOCS:.typ=.pdf)

# Archivos de datos e imágenes que usa la tesis
DATA_FILES   := $(wildcard $(DOCS_DATA)/*.csv) $(wildcard $(DOCS_DATA)/*.json)
IMG_FILES    := $(wildcard $(DOCS_IMG)/*.svg) $(wildcard $(DOCS_IMG)/*.png)

# ========================
# Targets principales
# ========================
.PHONY: all docs data clean clean-docs clean-data clean-all help lint fmt check strip-notebooks

all: $(TARGETS)

## Compilar todos los documentos Typst
docs: $(TYP_PDFS)

## Procesar datos (configs → pkl)
data: $(TARGETS)

# ========================
# Reglas de compilación
# ========================

# Cada PDF depende de su .typ, de los datos, imágenes y bibliografía
$(DOCS_DIR)/tesis.pdf: $(DOCS_DIR)/tesis.typ $(DATA_FILES) $(IMG_FILES) $(DOCS_DIR)/references.bib
	typst compile $<

$(DOCS_DIR)/plan.pdf: $(DOCS_DIR)/plan.typ
	typst compile $<

$(DOCS_DIR)/poster.pdf: $(DOCS_DIR)/poster.typ $(DOCS_DIR)/poster-template.typ $(IMG_FILES)
	typst compile $<

$(DOCS_DIR)/seminario-modesto.pdf: $(DOCS_DIR)/seminario-modesto.typ $(IMG_FILES)
	typst compile $<

# Datos: configs → pkl
$(TARGETS_DIR)/%.pkl: $(CONFIGS_DIR)/%.yaml
	uv run python fkdc/process.py --config-file $^ --workdir $(TARGETS_DIR)

datasets: fkdc/datasets.py
	uv run python $^

configs: fkdc/config.py
	uv run python $^

# ========================
# Linting y formato
# ========================

## Correr ruff check + format
lint:
	uv run ruff check --fix .
	uv run ruff format .

## Solo formatear
fmt:
	uv run ruff format .

## Limpiar metadata y outputs de notebooks
strip-notebooks:
	uv run nbstripout docs/*.ipynb

## Verificar sin modificar (para CI / pre-commit)
check:
	uv run ruff check .
	uv run ruff format --check .
	@tmp=$$(mktemp /tmp/typst-check-XXXXXX.pdf); \
	trap "rm -f $$tmp" EXIT; \
	for f in $(TYP_DOCS); do \
		echo "typst compile $$f"; \
		typst compile $$f $$tmp 2>&1 || exit 1; \
	done
	@echo "All checks passed."

# ========================
# Limpieza
# ========================

## Limpiar PDFs generados
clean-docs:
	rm -f $(TYP_PDFS)

## Limpiar datos procesados (con confirmación: son costosos de regenerar)
clean-data:
	@echo "Se borrarán $(words $(TARGETS)) archivos .pkl en $(TARGETS_DIR)/."
	@echo "Regenerarlos puede tomar horas. ¿Continuar? [y/N]" && read ans && [ "$$ans" = "y" ]
	rm -f $(TARGETS)

## Limpiar todo
clean-all: clean-docs clean-data

## Alias retrocompatible
clean: clean-data

# ========================
# Ayuda
# ========================
help:
	@echo "Targets disponibles:"
	@echo "  make docs          Compilar todos los .typ → .pdf"
	@echo "  make data          Procesar configs → pkl"
	@echo "  make all           Solo data (retrocompatible)"
	@echo "  make lint          Ruff check + format (Python/Jupyter)"
	@echo "  make fmt           Solo ruff format"
	@echo "  make strip-notebooks  Limpiar metadata y outputs de notebooks"
	@echo "  make check         Verificar lint + compilación Typst (no modifica)"
	@echo "  make clean-docs    Borrar PDFs generados"
	@echo "  make clean-data    Borrar .pkl procesados"
	@echo "  make clean-all     Borrar todo lo generado"
	@echo "  make help          Mostrar esta ayuda"
	@echo ""
	@echo "Directorios:"
	@echo "  DOCS_DATA = $(DOCS_DATA)  (CSV, JSON)"
	@echo "  DOCS_IMG  = $(DOCS_IMG)   (SVG, PNG)"
	@echo "  BIB       = $(DOCS_DIR)/references.bib"
