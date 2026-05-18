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

# Stamp file for viz.py output
DOCS_STAMP   := $(DOCS_DIR)/.viz-stamp

# ========================
# Entrega final al sistema de la maestría
# ========================
# El archivo TESIS_APELLIDO_NOMBRE.pdf concatena carátula + resúmenes + cuerpo
# + bibliografía y lo comprime con ghostscript para entrar al límite de 15 MB.
APELLIDO ?= BARRERA
NOMBRE   ?= GONZALO
# FIRMA = 0 → líneas vacías (firma manual posterior); 1 → imágenes embebidas
FIRMA    ?= 0
# VERSION = breve (default) | corta | larga
VERSION  ?= breve

ENTREGA_PDF   := TESIS_$(APELLIDO)_$(NOMBRE).pdf
TYPST_FIRMAS  := $(if $(filter 1,$(FIRMA)),--input firmas=true,)
TYPST_VERSION := --input version=$(VERSION)

# ========================
# Targets principales
# ========================
.PHONY: all docs data viz clean clean-docs clean-data clean-all clean-entrega \
        help lint fmt check entrega tesis-firmada

all: $(TARGETS)

## Compilar todos los documentos Typst
docs: $(TYP_PDFS)

## Procesar datos (configs → pkl)
data: $(TARGETS)

## Armar el PDF final para subir al sistema (sin firmas embebidas)
entrega: $(ENTREGA_PDF)

## Atajo: PDF final con firmas escaneadas embebidas
# Borra el bundle antes de rebuildear porque Make no rastrea el valor de FIRMA.
tesis-firmada:
	rm -f $(ENTREGA_PDF)
	$(MAKE) $(ENTREGA_PDF) FIRMA=1

# ========================
# Reglas de compilación
# ========================

# Visualizaciones: regenerar si viz.py o sus dependencias cambian
$(DOCS_STAMP): fkdc/viz.py fkdc/datasets.py fkdc/config.py
	uv run python fkdc/viz.py
	@touch $@

## Generar figuras y datos para la tesis
viz: $(DOCS_STAMP)

# Cada PDF depende de su .typ, de las visualizaciones generadas y la bibliografía
$(DOCS_DIR)/tesis.pdf: $(DOCS_DIR)/tesis.typ $(DOCS_STAMP) $(DOCS_DIR)/references.bib
	typst compile $<

$(DOCS_DIR)/plan.pdf: $(DOCS_DIR)/plan.typ
	typst compile $<

$(DOCS_DIR)/poster.pdf: $(DOCS_DIR)/poster.typ $(DOCS_DIR)/poster-template.typ $(DOCS_STAMP)
	typst compile $<

$(DOCS_DIR)/seminario-modesto.pdf: $(DOCS_DIR)/seminario-modesto.typ $(DOCS_STAMP)
	typst compile $<

# Bundle final: caratula + resumen + tesis, concatenados y comprimidos.
# El nombre del archivo respeta el formato exigido por la maestría
# (TESIS_APELLIDO_NOMBRE.pdf). Toggles: FIRMA=1 para firmas escaneadas,
# VERSION=breve|corta|larga para elegir longitud del resumen.
$(ENTREGA_PDF): $(DOCS_DIR)/caratula.typ $(DOCS_DIR)/resumen.typ \
                $(DOCS_DIR)/tesis.typ $(DOCS_DIR)/references.bib \
                $(DOCS_DIR)/img/Logo-fcenuba.png $(DOCS_STAMP) \
                $(if $(filter 1,$(FIRMA)),$(DOCS_DIR)/img/firma-gonzalo.png $(DOCS_DIR)/img/firma-pablo.png,)
	@echo "==> Compilando carátula (FIRMA=$(FIRMA))"
	typst compile $(TYPST_FIRMAS) $(DOCS_DIR)/caratula.typ
	@echo "==> Compilando resúmenes (VERSION=$(VERSION))"
	typst compile $(TYPST_VERSION) $(DOCS_DIR)/resumen.typ
	@echo "==> Compilando cuerpo de la tesis (FIRMA=$(FIRMA))"
	typst compile $(TYPST_FIRMAS) $(DOCS_DIR)/tesis.typ
	@echo "==> Concatenando con pdfunite"
	@TMP_RAW=$$(mktemp -t TESIS_RAW_XXXXXX) && mv $$TMP_RAW $$TMP_RAW.pdf && TMP_RAW=$$TMP_RAW.pdf; \
	pdfunite $(DOCS_DIR)/caratula.pdf $(DOCS_DIR)/resumen.pdf $(DOCS_DIR)/tesis.pdf $$TMP_RAW 2>/dev/null; \
	echo "==> Comprimiendo con ghostscript (PDFSETTINGS=/ebook)"; \
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
	   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(ENTREGA_PDF) $$TMP_RAW; \
	rm -f $$TMP_RAW
	@echo "==> Listo: $(ENTREGA_PDF) ($$(du -h $(ENTREGA_PDF) | awk '{print $$1}'))"
	@pdfinfo $(ENTREGA_PDF) 2>/dev/null | grep -E "^Pages|^File size" | sed 's/^/    /'

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

## Borrar el bundle de entrega
clean-entrega:
	rm -f $(ENTREGA_PDF)

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
	@echo "  make viz           Generar figuras y datos (fkdc/viz.py)"
	@echo "  make data          Procesar configs → pkl"
	@echo "  make all           Solo data (retrocompatible)"
	@echo "  make lint          Ruff check + format"
	@echo "  make fmt           Solo ruff format"
	@echo "  make check         Verificar lint + compilación Typst (no modifica)"
	@echo "  make clean-docs    Borrar PDFs generados"
	@echo "  make clean-data    Borrar .pkl procesados"
	@echo "  make clean-all     Borrar todo lo generado"
	@echo "  make help          Mostrar esta ayuda"
	@echo ""
	@echo "Entrega al sistema de la maestría:"
	@echo "  make entrega                  Armar $(ENTREGA_PDF) (sin firmas)"
	@echo "  make tesis-firmada            Armar con firmas embebidas (FIRMA=1)"
	@echo "  make $(ENTREGA_PDF)   Directamente"
	@echo "  make clean-entrega            Borrar el bundle"
	@echo ""
	@echo "Variables ajustables:"
	@echo "  FIRMA=0|1            (default: 0 / líneas vacías)"
	@echo "  VERSION=breve|corta|larga  (default: breve)"
	@echo "  APELLIDO=...         (default: BARRERA)"
	@echo "  NOMBRE=...           (default: GONZALO)"
