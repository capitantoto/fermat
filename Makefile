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
# Fuentes tipográficas locales (Excalifont para los SVG de Excalidraw)
TYPST        := typst compile --font-path $(DOCS_DIR)/fonts

# Stamp file for viz.py output
DOCS_STAMP   := $(DOCS_DIR)/.viz-stamp

# ========================
# Entrega final al sistema de la maestría
# ========================
# El archivo TESIS_APELLIDO_NOMBRE.pdf concatena carátula + resúmenes + cuerpo
# + bibliografía y lo comprime con ghostscript para entrar al límite de 15 MB.
APELLIDO ?= BARRERA
NOMBRE   ?= GONZALO
# FIRMA = 0 → sin firmas ni aclaraciones; 1 → imágenes escaneadas + Lic./Dr.
FIRMA    ?= 0

ENTREGA_PDF   := TESIS_$(APELLIDO)_$(NOMBRE).pdf
TYPST_FIRMAS  := $(if $(filter 1,$(FIRMA)),--input firmas=true,)

# ========================
# Targets principales
# ========================
.PHONY: all docs data viz clean clean-docs clean-data clean-all clean-entrega \
        help lint fmt check entrega tesis-firmada preparar-campania correr-cpu \
        correr-torch progreso-campania tablas-campania tablero-campania

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
	$(TYPST) $<

$(DOCS_DIR)/plan.pdf: $(DOCS_DIR)/plan.typ
	$(TYPST) $<

$(DOCS_DIR)/poster.pdf: $(DOCS_DIR)/poster.typ $(DOCS_DIR)/poster-template.typ $(DOCS_STAMP)
	$(TYPST) $<

$(DOCS_DIR)/seminario-modesto.pdf: $(DOCS_DIR)/seminario-modesto.typ $(DOCS_STAMP)
	$(TYPST) $<

# Bundle final: tesis.typ ya integra carátula + resúmenes + cuerpo + firmas.
# El nombre del archivo respeta el formato exigido por la maestría
# (TESIS_APELLIDO_NOMBRE.pdf). Toggle: FIRMA=1 para firmas escaneadas.
$(ENTREGA_PDF): $(DOCS_DIR)/tesis.typ $(DOCS_DIR)/references.bib \
                $(DOCS_DIR)/img/Logo-fcenuba.png $(DOCS_STAMP) \
                $(if $(filter 1,$(FIRMA)),$(DOCS_DIR)/img/firma-gonzalo.png $(DOCS_DIR)/img/firma-pablo.png,)
	@echo "==> Compilando tesis completa (FIRMA=$(FIRMA))"
	@TMP_RAW=$$(mktemp -t TESIS_RAW_XXXXXX) && mv $$TMP_RAW $$TMP_RAW.pdf && TMP_RAW=$$TMP_RAW.pdf; \
	$(TYPST) $(TYPST_FIRMAS) $(DOCS_DIR)/tesis.typ $$TMP_RAW; \
	echo "==> Comprimiendo con ghostscript (/printer + sRGB)"; \
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer \
	   -dColorConversionStrategy=/sRGB \
	   -dColorImageResolution=200 -dGrayImageResolution=200 \
	   -dColorImageDownsampleThreshold=1.5 \
	   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(ENTREGA_PDF) $$TMP_RAW; \
	rm -f $$TMP_RAW
	@echo "==> Listo: $(ENTREGA_PDF) ($$(du -h $(ENTREGA_PDF) | awk '{print $$1}'))"
	@pdfinfo $(ENTREGA_PDF) 2>/dev/null | grep -E "^Pages|^File size" | sed 's/^/    /'

# Datos: configs → pkl
$(TARGETS_DIR)/%.pkl: $(CONFIGS_DIR)/%.yaml
	uv run python fkdc/process.py --archivo-config $^ --dir-trabajo $(TARGETS_DIR)

datasets: fkdc/datasets.py
	uv run python $^

configs: fkdc/config.py
	uv run python $^

# ========================
# Campañas (fkdc/campanias.py y fkdc/correr.py)
# ========================
# Cadena de marcas: datasets → configs → lotes.mk → un lote por marca `.hecho`. Toda
# escritura es atómica (temporal + renombre) y cada lote saltea las configs que ya
# tienen info, así que tras un corte (incluso kill -9 o un apagón) basta con repetir
# el mismo comando: `make -k` sigue ante lotes fallidos y la próxima corrida los
# reintenta. Ejemplo (los dos carriles a la vez):
#   caffeinate -i make -k -j10 correr-cpu    y    caffeinate -i make -k correr-torch
CAMPANIA     ?= heroica
CONJUNTOS    ?=
CAMPANIA_DIR := sandbox/$(CAMPANIA)
LOTES_MK     := $(CAMPANIA_DIR)/lotes.mk
CORRER       := nice -n 10 .venv/bin/python -m fkdc.correr lote $(CAMPANIA)

ifneq ($(filter correr-cpu correr-torch,$(MAKECMDGOALS)),)
-include $(LOTES_MK)
endif

$(CAMPANIA_DIR)/datasets.hecho:
	.venv/bin/python -m fkdc.campanias datasets $(CAMPANIA) --escribir
	@touch $@

$(CAMPANIA_DIR)/configs.hecho: $(CAMPANIA_DIR)/datasets.hecho
	.venv/bin/python -m fkdc.campanias configs $(CAMPANIA) $(CONJUNTOS)
	@touch $@

$(LOTES_MK): $(CAMPANIA_DIR)/configs.hecho
	.venv/bin/python -m fkdc.campanias lotes $(CAMPANIA) $(CONJUNTOS)

$(CAMPANIA_DIR)/lotes/cpu/%.hecho:
	@mkdir -p $(CAMPANIA_DIR)/logs/cpu
	$(CORRER) cpu $@ $^ >> $(CAMPANIA_DIR)/logs/cpu/$*.log 2>&1

$(CAMPANIA_DIR)/lotes/torch/%.hecho:
	@mkdir -p $(CAMPANIA_DIR)/logs/torch
	$(CORRER) torch $@ $^ >> $(CAMPANIA_DIR)/logs/torch/$*.log 2>&1

## Preparar datasets, configs y lotes de la campaña (también lo hace correr-*)
preparar-campania: $(CAMPANIA_DIR)/configs.hecho
	.venv/bin/python -m fkdc.campanias lotes $(CAMPANIA) $(CONJUNTOS)

## Carril de CPU (con -j N, N procesos de un hilo) y carril de torch (sin -j)
correr-cpu: $(LOTES_CPU)
correr-torch: $(LOTES_TORCH)

## Avance: infos por clasificador, lotes completos y lotes con fallas
progreso-campania:
	@.venv/bin/python -m fkdc.campanias progreso $(CAMPANIA) $(CONJUNTOS)

## Tablero HTML de seguimiento (sandbox/$(CAMPANIA)/tablero/index.html)
tablero-campania:
	@.venv/bin/python scripts/tablero.py $(CAMPANIA)

## Tablas resumen de la campaña ({dir}/tablas)
tablas-campania:
	.venv/bin/python -m fkdc.campanias tablas $(CAMPANIA) $(CONJUNTOS)

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
		echo "$(TYPST) $$f"; \
		$(TYPST) $$f $$tmp 2>&1 || exit 1; \
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
	@echo "Campañas (CAMPANIA=heroica por omisión; repetir el mismo comando reanuda):"
	@echo "  make preparar-campania     Datasets, configs y lotes"
	@echo "  make -k -j10 correr-cpu    Carril de CPU, 10 procesos"
	@echo "  make -k correr-torch       Carril de torch (fundacionales y profundos)"
	@echo "  make progreso-campania     Avance"
	@echo "  make tablas-campania       Tablas resumen"
	@echo ""
	@echo "Entrega al sistema de la maestría:"
	@echo "  make entrega                  Armar $(ENTREGA_PDF) (sin firmas)"
	@echo "  make tesis-firmada            Armar con firmas embebidas (FIRMA=1)"
	@echo "  make $(ENTREGA_PDF)   Directamente"
	@echo "  make clean-entrega            Borrar el bundle"
	@echo ""
	@echo "Variables ajustables:"
	@echo "  FIRMA=0|1            (default: 0 / sin firmas)"
	@echo "  APELLIDO=...         (default: BARRERA)"
	@echo "  NOMBRE=...           (default: GONZALO)"
