"""
Tablero de seguimiento de una campaña (por omisión, la heroica).

Escribe una página HTML autocontenida (sin red) con el avance por carril y por
clasificador, el ritmo por hora y una vista descriptiva parcial de los resultados:
medianas de R² crudo, R² templado y exactitud por variante y clasificador, y
comparaciones apareadas del mismo tipo (crudo con crudo, templado con templado).
No reemplaza el análisis prerregistrado (`scripts/replicar.py`), que se corre al
terminar la campaña.

Las métricas de cada info se leen una sola vez y se guardan en un caché
(`{dir}/tablero/corridas.parquet`): cada actualización lee solo las infos nuevas.

Uso, desde la raíz del repositorio:
    .venv/bin/python scripts/tablero.py heroica            # una vez
    .venv/bin/python scripts/tablero.py heroica --cada 5   # cada 5 minutos
y abrir `sandbox/heroica/tablero/index.html` en el navegador (se recarga sola).
"""

import html
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from fkdc import campanias, registro

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("tablero")

recarga_segundos = 120
tolerancia_empate = 1e-9  # como en el prerregistro: |Δ| < 1e-9 es empate
columnas_cache = [
    "archivo",
    "mtime",
    "variante",
    "semilla",
    "clf",
    "r2",
    "r2_temp",
    "accuracy",
    "t_entrenar",
]
metricas = {"r2": "R² crudo", "r2_temp": "R² templado", "accuracy": "Exactitud"}
comparaciones = [
    ("fkdc", "kdc_exacto"),
    ("fkn", "kn"),
    ("fkdc", "kdc"),
    ("tabpfn", "fkdc"),
]
nombres_grupos = {
    "tesis": "Tesis",
    "compendio": "Compendio",
    "fundacionales": "Fundacionales",
    "profundos": "Profundos",
}
# En la heroica, el papel «kdc» lo cumple kdc_exacto; `kdc` es el de sklearn
etiquetas_clf = {"kdc_exacto": "kdc", "kdc": "kdc·sk"}
# Rampa secuencial azul (100 → 700) de la paleta de referencia
rampa = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


# =====================================================================
# Datos
# =====================================================================


def actualizar_cache(c: campanias.Campania) -> pd.DataFrame:
    """Métricas de todas las infos, leyendo del disco solo las que faltan en caché."""
    import __main__
    from fkdc.datasets import Dataset

    __main__.Dataset = Dataset  # algunas infos pickleadas lo referencian
    ruta_cache = c.dir / "tablero" / "corridas.parquet"
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    previas = (
        pd.read_parquet(ruta_cache)
        if ruta_cache.exists()
        else pd.DataFrame(columns=columnas_cache)
    )
    conocidas = set(previas.archivo)
    nuevas = [
        ruta
        for ruta in sorted(c.dir_infos.glob("*.pkl"))
        if ruta.name not in conocidas and ruta.stem.split("-")[2] in c.todos
    ]
    filas = []
    for ruta in nuevas:
        try:
            fila = campanias._fila_corrida(ruta)
        except Exception as exc:  # no debería pasar: las infos se escriben atómicas
            logger.warning("No se pudo leer %s: %s", ruta.name, exc)
            continue
        fila["archivo"] = ruta.name
        fila["mtime"] = ruta.stat().st_mtime
        filas.append({k: fila.get(k, np.nan) for k in columnas_cache})
    if not filas:
        return previas
    nuevas_filas = pd.DataFrame(filas)
    todas = (
        nuevas_filas
        if previas.empty
        else pd.concat([previas, nuevas_filas], ignore_index=True)
    )
    for columna in ("mtime", "r2", "r2_temp", "accuracy", "t_entrenar"):
        todas[columna] = pd.to_numeric(todas[columna], errors="coerce")
    temporal = ruta_cache.with_name(f".{ruta_cache.name}.tmp")
    todas.to_parquet(temporal, index=False)
    temporal.rename(ruta_cache)
    logger.info("Caché: %d infos (%d nuevas)", len(todas), len(filas))
    return todas


def carriles_corriendo() -> dict[str, bool]:
    """Si hay un `make … correr-{carril}` vivo, por carril."""
    salida = subprocess.run(
        ["ps", "-Ao", "command"], capture_output=True, text=True, check=False
    ).stdout
    return {
        carril: any(
            "make" in linea and f"correr-{carril}" in linea
            for linea in salida.splitlines()
        )
        for carril in ("cpu", "torch")
    }


def estado_carriles(c, avance, corridas, ahora) -> pd.DataFrame:
    """Hechas, totales, ritmo de la última hora, última info y fin estimado."""
    corriendo = carriles_corriendo()
    carril_de = {clf: registro.entrada(clf).carril for clf in c.todos}
    corridas = corridas.assign(carril=corridas.clf.map(carril_de))
    filas = []
    for carril, grupo in avance.groupby("carril"):
        mias = corridas[corridas.carril.eq(carril)]
        ritmo = int((mias.mtime >= ahora - 3600).sum())
        faltan = int(grupo.faltan.sum())
        fin = ahora + 3600 * faltan / ritmo if ritmo and faltan else None
        filas.append(
            {
                "carril": carril,
                "hechas": int(grupo.infos.sum()),
                "total": int(grupo.configs.sum()),
                "ritmo": ritmo,
                "ultima": mias.mtime.max() if len(mias) else None,
                "fin": fin,
                "corre": corriendo.get(carril, False),
            }
        )
    return pd.DataFrame(filas)


def medianas(corridas: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Mediana, IQR y n por (variante, clasificador) de una métrica."""
    datos = corridas.dropna(subset=[metrica])
    agrupado = datos.groupby(["variante", "clf"])[metrica]
    return pd.DataFrame(
        {
            "mediana": agrupado.median(),
            "iqr": agrupado.quantile(0.75) - agrupado.quantile(0.25),
            "n": agrupado.size(),
        }
    )


def apareadas(corridas: pd.DataFrame, a: str, b: str, metrica: str) -> pd.DataFrame:
    """Por variante: semillas en que gana `a`, gana `b`, empatan, y mediana de Δ."""
    tabla = corridas.pivot_table(
        index=["variante", "semilla"], columns="clf", values=metrica, aggfunc="first"
    )
    if a not in tabla or b not in tabla:
        return pd.DataFrame()
    delta = (tabla[a] - tabla[b]).dropna().rename("delta").reset_index()
    return delta.groupby("variante").delta.agg(
        gana_a=lambda d: int((d > tolerancia_empate).sum()),
        gana_b=lambda d: int((d < -tolerancia_empate).sum()),
        empates=lambda d: int((d.abs() <= tolerancia_empate).sum()),
        mediana="median",
    )


# =====================================================================
# HTML
# =====================================================================


dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def _fecha(momento: datetime, formato: str = "%d %H:%M") -> str:
    """Fecha con el día de la semana en castellano, sin depender del locale."""
    return f"{dias[momento.weekday()]} {momento:{formato}}"


def _hora(marca: float | None) -> str:
    if marca is None or pd.isna(marca):
        return "—"
    return _fecha(datetime.fromtimestamp(marca))


def _hace(marca: float | None, ahora: float) -> str:
    if marca is None or pd.isna(marca):
        return "sin infos"
    minutos = (ahora - marca) / 60
    return f"hace {minutos:.0f} min" if minutos < 120 else f"hace {minutos / 60:.1f} h"


def _celda_color(valor: float, minimo: float, maximo: float) -> tuple[str, str]:
    """(fondo, tinta) de la rampa secuencial; los negativos van en gris neutro."""
    if valor < 0:
        return "var(--neutro)", "var(--tinta-1)"
    t = 0.0 if maximo <= minimo else (valor - minimo) / (maximo - minimo)
    paso = int(round(np.clip(t, 0, 1) * (len(rampa) - 1)))
    return rampa[paso], "#0b0b0b" if paso <= 5 else "#ffffff"


def tarjetas(estados: pd.DataFrame, fallas: int, ahora: float) -> str:
    hechas, total = int(estados.hechas.sum()), int(estados.total.sum())
    piezas = [
        f"""<div class="tarjeta"><div class="rotulo">Total</div>
        <div class="cifra">{100 * hechas / total:.1f}%</div>
        <div class="detalle">{hechas:,} de {total:,} tareas</div>
        <div class="barra"><span style="width:{100 * hechas / total:.2f}%"></span></div>
        </div>"""
    ]
    for fila in estados.itertuples():
        if not fila.corre and fila.hechas < fila.total:
            estado = '<span class="alerta">⚠ make no corre</span>'
        elif fila.hechas >= fila.total:
            estado = "✓ terminado"
        else:
            estado = "▶ corriendo"
        fin = _hora(fila.fin) if fila.fin else "—"
        piezas.append(
            f"""<div class="tarjeta"><div class="rotulo">Carril {fila.carril}</div>
            <div class="cifra">{100 * fila.hechas / fila.total:.1f}%</div>
            <div class="detalle">{fila.hechas:,} de {fila.total:,} · {estado}</div>
            <div class="barra">
            <span style="width:{100 * fila.hechas / fila.total:.2f}%"></span></div>
            <div class="detalle">{fila.ritmo:,} tareas en la última hora ·
            última {_hace(fila.ultima, ahora)}</div>
            <div class="detalle">Fin estimado: <b>{fin}</b></div></div>"""
        )
    clase = "alerta" if fallas else ""
    piezas.append(
        f"""<div class="tarjeta"><div class="rotulo">Lotes con fallas</div>
        <div class="cifra {clase}">{fallas}</div>
        <div class="detalle">se reintentan al relanzar el mismo <code>make</code></div>
        </div>"""
    )
    return '<div class="tarjetas">' + "".join(piezas) + "</div>"


def grafico_ritmo(corridas: pd.DataFrame, ahora: float) -> str:
    """Barras de infos terminadas por hora, desde la primera info hasta ahora."""
    if corridas.empty:
        return "<p>Todavía no hay infos.</p>"
    horas = pd.to_datetime(corridas.mtime, unit="s").dt.floor("h")
    conteo = horas.value_counts().sort_index()
    rango = pd.date_range(
        conteo.index.min(), pd.to_datetime(ahora, unit="s").floor("h"), freq="h"
    )
    conteo = conteo.reindex(rango, fill_value=0)
    ancho, alto, margen_i, margen_a = 760, 170, 44, 26
    tope = max(int(conteo.max()), 1)
    paso_y = 10 ** int(np.floor(np.log10(tope)))
    paso_y = paso_y * (5 if tope / paso_y > 5 else 2 if tope / paso_y > 2 else 1)
    tope = int(np.ceil(tope / paso_y) * paso_y)
    alto_util = alto - margen_a - 8
    ancho_barra = (ancho - margen_i) / len(conteo)
    partes = [
        f'<svg viewBox="0 0 {ancho} {alto}" class="grafico" role="img" '
        'aria-label="Tareas terminadas por hora">'
    ]
    for y in range(0, tope + 1, paso_y):
        py = 8 + alto_util * (1 - y / tope)
        partes.append(
            f'<line x1="{margen_i}" x2="{ancho}" y1="{py:.1f}" y2="{py:.1f}" '
            'class="grilla"/>'
            f'<text x="{margen_i - 6}" y="{py + 4:.1f}" class="eje" '
            f'text-anchor="end">{y:,}</text>'
        )
    for i, (hora, valor) in enumerate(conteo.items()):
        x = margen_i + i * ancho_barra + 1
        w = max(ancho_barra - 2, 1)
        h = alto_util * valor / tope
        base = 8 + alto_util
        r = min(4, w / 2, h)
        if valor:
            partes.append(
                f'<path class="barra-dato" d="M{x:.1f},{base:.1f} '
                f"V{base - h + r:.1f} Q{x:.1f},{base - h:.1f} "
                f"{x + r:.1f},{base - h:.1f} "
                f"H{x + w - r:.1f} Q{x + w:.1f},{base - h:.1f} {x + w:.1f},"
                f'{base - h + r:.1f} V{base:.1f} Z">'
                f"<title>{_fecha(hora, '%d %H')}h: {valor:,} tareas</title></path>"
            )
        else:
            partes.append(
                f'<rect x="{x:.1f}" y="{base - 2:.1f}" width="{w:.1f}" height="2" '
                f'class="barra-cero"><title>{_fecha(hora, "%d %H")}h: 0 tareas'
                "</title></rect>"
            )
        if hora.hour % 3 == 0:
            partes.append(
                f'<text x="{x + w / 2:.1f}" y="{alto - 6}" class="eje" '
                f'text-anchor="middle">{hora:%H}h</text>'
            )
    partes.append("</svg>")
    return "".join(partes)


def tabla_avance(avance: pd.DataFrame) -> str:
    filas = []
    for fila in avance.sort_values(["carril", "clf"]).itertuples():
        porcentaje = 100 * fila.infos / fila.configs
        filas.append(
            f"<tr><td>{html.escape(etiquetas_clf.get(fila.clf, fila.clf))}</td>"
            f"<td>{fila.carril}</td><td class='num'>{fila.infos:,} / {fila.configs:,}"
            f"</td><td><div class='barra fina'><span style='width:{porcentaje:.1f}%'>"
            f"</span></div></td><td class='num'>{porcentaje:.0f}%</td></tr>"
        )
    return (
        "<table class='avance'><thead><tr><th>Clasificador</th><th>Carril</th>"
        "<th>Hechas</th><th></th><th></th></tr></thead><tbody>"
        + "".join(filas)
        + "</tbody></table>"
    )


def tabla_medianas(c, corridas: pd.DataFrame, metrica: str) -> str:
    """Mapa de calor: mediana por variante (filas) y clasificador (columnas)."""
    datos = medianas(corridas, metrica)
    if datos.empty:
        return "<p>Sin datos todavía.</p>"
    columnas = [clf for clf in c.clasificadores if clf in set(corridas.clf)]
    grupos = [registro.entrada(clf).grupo for clf in columnas]
    minimo = 0.0 if metrica != "accuracy" else float(datos.mediana.min())
    encabezado_grupos = "".join(
        f"<th colspan='{grupos.count(g)}' class='grupo'>{nombres_grupos.get(g, g)}</th>"
        for g in dict.fromkeys(grupos)
    )
    encabezado = "".join(
        f"<th class='clf'><span>{html.escape(etiquetas_clf.get(clf, clf))}</span></th>"
        for clf in columnas
    )
    nuevas = set(campanias._variantes_nuevas())
    filas, separador = [], False
    for variante in c.variantes:
        if variante in nuevas and not separador:
            filas.append(
                f"<tr class='separador'><td colspan='{len(columnas) + 2}'>"
                "Datasets nuevos (de investigacion-v6)</td></tr>"
            )
            separador = True
        if variante not in datos.index.get_level_values(0):
            continue
        fila_datos = datos.loc[variante]
        mejor = fila_datos.mediana.round(4).max()
        ns = fila_datos.n
        celdas = []
        for clf in columnas:
            if clf not in fila_datos.index:
                celdas.append("<td class='vacia'>·</td>")
                continue
            d = fila_datos.loc[clf]
            fondo, tinta = _celda_color(d.mediana, minimo, 1.0)
            negrita = " mejor" if round(d.mediana, 4) == mejor else ""
            titulo = (
                f"{etiquetas_clf.get(clf, clf)} · {variante}: mediana {d.mediana:.3f}, "
                f"IQR {d.iqr:.3f}, n = {int(d.n)}"
            )
            celdas.append(
                f"<td class='calor{negrita}' style='background:{fondo};color:{tinta}' "
                f"title='{html.escape(titulo)}'>{d.mediana:.2f}</td>"
            )
        filas.append(
            f"<tr><th class='variante'>{variante}</th>"
            f"<td class='num n'>{int(ns.min())}–{int(ns.max())}</td>"
            f"{''.join(celdas)}</tr>"
        )
    return (
        "<div class='desborde'><table class='calor-tabla'><thead>"
        f"<tr><th></th><th></th>{encabezado_grupos}</tr>"
        f"<tr><th>Variante</th><th>semillas</th>{encabezado}</tr></thead>"
        f"<tbody>{''.join(filas)}</tbody></table></div>"
    )


def tabla_apareadas(c, corridas: pd.DataFrame) -> str:
    bloques = []
    for a, b in comparaciones:
        if a not in set(corridas.clf) or b not in set(corridas.clf):
            continue
        por_metrica = {m: apareadas(corridas, a, b, m) for m in ("r2", "r2_temp")}
        filas = []
        for variante in c.variantes:
            celdas = []
            for metrica in ("r2", "r2_temp"):
                t = por_metrica[metrica]
                if t.empty or variante not in t.index:
                    celdas.append("<td class='vacia'>·</td><td></td>")
                    continue
                f = t.loc[variante]
                celdas.append(
                    f"<td class='num'>{f.gana_a:.0f}–{f.gana_b:.0f}–"
                    f"{f.empates:.0f}</td>"
                    f"<td class='num'>{f.mediana:+.3f}</td>"
                )
            filas.append(
                f"<tr><th class='variante'>{variante}</th>{''.join(celdas)}</tr>"
            )
        ea, eb = etiquetas_clf.get(a, a), etiquetas_clf.get(b, b)
        bloques.append(
            f"<details><summary>{ea} contra {eb}</summary>"
            "<table class='apareadas'><thead><tr><th></th>"
            "<th colspan='2'>R² crudo vs crudo</th>"
            "<th colspan='2'>R² templado vs templado</th></tr>"
            f"<tr><th>Variante</th><th>{ea}–{eb}–empates</th><th>mediana Δ</th>"
            f"<th>{ea}–{eb}–empates</th><th>mediana Δ</th></tr></thead>"
            f"<tbody>{''.join(filas)}</tbody></table></details>"
        )
    return "".join(bloques)


estilos = """
:root {
  color-scheme: light;
  --fondo: #f7f7f5; --superficie: #fcfcfb; --borde: #e3e2de;
  --tinta-1: #0b0b0b; --tinta-2: #52514e; --tinta-3: #7a7974;
  --dato: #2a78d6; --grilla: #e9e8e4; --neutro: #f0efec; --alerta: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --fondo: #121211; --superficie: #1a1a19; --borde: #33332f;
    --tinta-1: #ffffff; --tinta-2: #c3c2b7; --tinta-3: #8f8e86;
    --dato: #3987e5; --grilla: #2a2a27; --neutro: #383835; --alerta: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --fondo: #121211; --superficie: #1a1a19; --borde: #33332f;
  --tinta-1: #ffffff; --tinta-2: #c3c2b7; --tinta-3: #8f8e86;
  --dato: #3987e5; --grilla: #2a2a27; --neutro: #383835; --alerta: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--fondo); color: var(--tinta-1);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; }
p, .nota { color: var(--tinta-2); }
.nota { font-size: 12.5px; }
code { font-size: 12px; }
.aviso { background: var(--superficie); border: 1px solid var(--borde);
  border-left: 4px solid var(--dato); padding: 8px 12px; border-radius: 6px;
  color: var(--tinta-2); font-size: 13px; }
.tarjetas { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px; margin-top: 14px; }
.tarjeta { background: var(--superficie); border: 1px solid var(--borde);
  border-radius: 8px; padding: 12px 14px; }
.rotulo { color: var(--tinta-2); font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; }
.cifra { font-size: 28px; font-weight: 600; font-variant-numeric: tabular-nums; }
.detalle { color: var(--tinta-2); font-size: 12.5px; margin-top: 2px; }
.alerta { color: var(--alerta); font-weight: 600; }
.barra { height: 8px; background: var(--grilla); border-radius: 4px; overflow: hidden;
  margin: 6px 0; }
.barra span { display: block; height: 100%; background: var(--dato);
  border-radius: 4px; }
.barra.fina { height: 6px; width: 160px; margin: 0; }
.panel { background: var(--superficie); border: 1px solid var(--borde);
  border-radius: 8px; padding: 12px 14px; }
.grafico { width: 100%; height: auto; display: block; }
.grilla { stroke: var(--grilla); stroke-width: 1; }
.eje { fill: var(--tinta-3); font-size: 11px; }
.barra-dato { fill: var(--dato); }
.barra-dato:hover { opacity: .8; }
.barra-cero { fill: var(--tinta-3); opacity: .35; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 3px 6px; text-align: left; }
.num { text-align: right; }
.avance td, .avance th { border-bottom: 1px solid var(--grilla); }
.desborde { overflow-x: auto; }
.calor-tabla { font-size: 12px; }
.calor-tabla th.clf { vertical-align: bottom; height: 86px; padding: 0 1px; }
.calor-tabla th.clf span { writing-mode: vertical-rl; transform: rotate(180deg);
  white-space: nowrap; color: var(--tinta-2); font-weight: 500; }
.calor-tabla th.grupo { text-align: center; color: var(--tinta-2); font-weight: 600;
  border-bottom: 2px solid var(--borde); }
.calor-tabla td.calor { text-align: center; min-width: 38px; border: 2px solid
  var(--superficie); border-radius: 3px; }
.calor-tabla td.mejor { font-weight: 700; outline: 2px solid var(--tinta-1);
  outline-offset: -3px; }
.calor-tabla td.vacia, .apareadas td.vacia { color: var(--tinta-3);
  text-align: center; }
.calor-tabla td.n { color: var(--tinta-3); }
th.variante { font-weight: 500; white-space: nowrap; }
tr.separador td { padding-top: 12px; color: var(--tinta-2); font-weight: 600;
  border-bottom: 1px solid var(--borde); }
.pestanas { display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
.pestanas button { font: inherit; padding: 4px 10px; border-radius: 6px;
  border: 1px solid var(--borde); background: var(--fondo); color: var(--tinta-1);
  cursor: pointer; }
.pestanas button[aria-selected="true"] { background: var(--dato); color: #fff;
  border-color: var(--dato); }
.vista[hidden] { display: none; }
details { margin: 6px 0; }
summary { cursor: pointer; font-weight: 600; }
.apareadas { font-size: 12.5px; margin: 6px 0 12px; }
.apareadas td, .apareadas th { border-bottom: 1px solid var(--grilla); }
.escala { display: flex; align-items: center; gap: 6px; font-size: 12px;
  color: var(--tinta-2); margin-top: 6px; }
.escala span.paso { width: 18px; height: 10px; display: inline-block; }
"""

guion = """
const botones = document.querySelectorAll('.pestanas button');
botones.forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.pestanas button').forEach(o =>
    o.setAttribute('aria-selected', o === b));
  document.querySelectorAll('.vista').forEach(v => v.hidden = v.id !== b.dataset.vista);
  try { localStorage.setItem('tablero-vista', b.dataset.vista); } catch (e) {}
}));
try {
  const v = localStorage.getItem('tablero-vista');
  const b = v && document.querySelector(`.pestanas button[data-vista="${v}"]`);
  if (b) b.click();
} catch (e) {}
"""


def pagina(c: campanias.Campania) -> str:
    ahora = time.time()
    avance = campanias.progreso(c)
    corridas = actualizar_cache(c)
    estados = estado_carriles(c, avance, corridas, ahora)
    dir_fallas = c.dir / "fallas"
    fallas = (
        sum(1 for r in dir_fallas.rglob("*") if r.is_file())
        if dir_fallas.exists()
        else 0
    )
    botones = "".join(
        f"<button data-vista='vista-{m}' aria-selected='{str(i == 0).lower()}'>"
        f"{nombre}</button>"
        for i, (m, nombre) in enumerate(metricas.items())
    )
    vistas = "".join(
        f"<div class='vista' id='vista-{m}' {'hidden' if i else ''}>"
        f"{tabla_medianas(c, corridas, m)}</div>"
        for i, m in enumerate(metricas)
    )
    escala = "".join(
        f"<span class='paso' style='background:{p}'></span>" for p in rampa
    )
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{recarga_segundos}">
<title>Corrida {c.nombre}</title><style>{estilos}</style></head>
<body><main>
<h1>Corrida {c.nombre}</h1>
<p class="nota">Actualizado {_fecha(datetime.fromtimestamp(ahora), "%d/%m %H:%M:%S")}
· la página se recarga cada {recarga_segundos} s · {c.repeticiones} semillas,
{len(c.variantes)} variantes, {len(c.clasificadores)} clasificadores</p>
<div class="aviso">Vista descriptiva parcial: medianas sobre las semillas ya
terminadas, que no son las mismas en todas las celdas. No reemplaza el análisis
prerregistrado (<code>scripts/replicar.py</code>) que se corre al terminar.</div>
{tarjetas(estados, fallas, ahora)}
<h2>Tareas terminadas por hora</h2>
<div class="panel">{grafico_ritmo(corridas, ahora)}</div>
<h2>Medianas por variante y clasificador</h2>
<div class="pestanas" role="tablist">{botones}</div>
{vistas}
<div class="escala">0 {escala} 1 · gris: R² negativo · recuadro: mejor mediana de la
fila (empates incluidos) · «semillas»: mínimo–máximo de semillas terminadas por
clasificador · kdc = exacto, kdc·sk = KernelDensity de sklearn</div>
<h2>Comparaciones apareadas por semilla</h2>
<p class="nota">Cada celda cuenta semillas en que gana el primero, gana el segundo o
empatan (|Δ| ≤ 1e-9), y la mediana de Δ = primero − segundo. Siempre del mismo
tipo: crudo contra crudo, templado contra templado.</p>
{tabla_apareadas(c, corridas)}
<h2>Avance por clasificador</h2>
{tabla_avance(avance)}
</main><script>{guion}</script></body></html>"""


def generar(c: campanias.Campania) -> Path:
    """Escribe `{dir}/tablero/index.html` de forma atómica."""
    destino = c.dir / "tablero" / "index.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(f".{destino.name}.tmp")
    temporal.write_text(pagina(c), encoding="utf-8")
    temporal.rename(destino)
    logger.info("Tablero en %s", destino)
    return destino


def main(
    nombre: Annotated[str, typer.Argument(help="Campaña")] = "heroica",
    cada: Annotated[
        float | None, typer.Option(help="Regenerar cada N minutos (sin fin)")
    ] = None,
):
    """Genera el tablero de seguimiento de una campaña."""
    c = campanias.campania(nombre)
    generar(c)
    while cada:
        time.sleep(cada * 60)
        try:
            generar(c)
        except Exception:  # un error puntual no debe matar el bucle
            logger.exception("Falló la actualización; reintento en %s min", cada)


if __name__ == "__main__":
    typer.run(main)
