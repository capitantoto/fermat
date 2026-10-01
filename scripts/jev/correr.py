# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27,<1"]
# ///
"""
Envía a Jev (TypeSafe, `POST /v1/systemone`) las solicitudes de
`solicitudes.jsonl.gz` y guarda las respuestas crudas en `respuestas.jsonl`.

Sin opciones solo estima costo y tiempo; no envía nada. Ver LEEME.md.

    export JEV_API_KEY=...
    uv run correr.py              # estimación de costo y tiempo
    uv run correr.py --dry-run    # envía 3 solicitudes de prueba
    uv run correr.py --confirmar  # envía todo lo pendiente

Es reanudable: al volver a correrlo saltea las solicitudes que ya tienen respuesta
exitosa en `respuestas.jsonl` y reintenta las que fallaron. Respeta los límites
documentados (40 solicitudes/s, 100 000 tokens/s) con márgenes, y reintenta con
espera exponencial los errores transitorios (408, 429, 5xx, 529), honrando
`Retry-After`.
"""

import argparse
import gzip
import json
import os
import queue
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

dir_script = Path(__file__).resolve().parent
url_base_jev = "https://api.typesafe.ai"
ruta_api = "/v1/systemone"
# USD por millón de tokens de entrada; los de salida no se cobran (docs.typesafe.ai)
precio_por_millon = 0.042
reintentables = {408, 409, 425, 429, 500, 502, 503, 504, 529}
fatales = {401, 403}


def abrir(ruta: Path):
    """Abre un JSONL, comprimido o no."""
    if ruta.suffix == ".gz":
        return gzip.open(ruta, "rt", encoding="utf-8")
    return open(ruta, encoding="utf-8")


def leer_jsonl(ruta: Path):
    """Líneas de un JSONL; ignora una última línea cortada por una interrupción."""
    if not ruta.exists():
        return
    with abrir(ruta) as archivo:
        for linea in archivo:
            try:
                yield json.loads(linea)
            except json.JSONDecodeError:
                continue


def respondidas(ruta: Path) -> set[str]:
    """Ids con respuesta exitosa en `ruta`."""
    return {r["id"] for r in leer_jsonl(ruta) if r.get("estado") == 200}


def pendientes(args, hechas: set[str]):
    """Solicitudes aún sin respuesta exitosa, filtradas por régimen."""
    for linea in leer_jsonl(args.solicitudes):
        if linea["id"] in hechas:
            continue
        if args.regimen and linea["regimen"] not in args.regimen:
            continue
        yield linea


def factor_tokens(*rutas: Path) -> float | None:
    """Cociente entre tokens cobrados y estimados en respuestas previas."""
    cobrados = estimados = 0
    for ruta in rutas:
        for r in leer_jsonl(ruta):
            uso = (r.get("respuesta") or {}).get("usage") or {}
            if r.get("estado") == 200 and uso.get("input_tokens"):
                cobrados += uso["input_tokens"]
                estimados += r["tokens_estimados"]
    return cobrados / estimados if estimados else None


def estimar(args, hechas: set[str]) -> int:
    """Imprime solicitudes, tokens, costo y horas pendientes por régimen."""
    por_regimen = defaultdict(lambda: [0, 0])
    for linea in pendientes(args, hechas):
        por_regimen[linea["regimen"]][0] += 1
        por_regimen[linea["regimen"]][1] += linea["tokens_estimados"]
    factor = factor_tokens(args.respuestas, args.respuestas_prueba)
    print(f"Ya respondidas: {len(hechas)}")
    columnas = ["solicitudes", "Mtokens", "USD", "horas"]
    anchos = [12, 9, 8, 7]
    print(
        f"{'régimen':16s}",
        *(f"{c:>{a}s}" for c, a in zip(columnas, anchos, strict=True)),
    )
    total_n = total_tokens = 0
    for regimen, (n, tokens) in sorted(por_regimen.items()):
        tokens *= factor or 1
        print(
            f"{regimen:16s} {n:12d} {tokens / 1e6:9.1f}"
            f" {tokens / 1e6 * args.precio:8.2f} {n / args.rps / 3600:7.2f}"
        )
        total_n += n
        total_tokens += tokens
    print(
        f"{'total':16s} {total_n:12d} {total_tokens / 1e6:9.1f}"
        f" {total_tokens / 1e6 * args.precio:8.2f} {total_n / args.rps / 3600:7.2f}"
    )
    origen = (
        f"corregidos por el cociente {factor:.2f} observado en respuestas previas"
        if factor
        else "estimados con el tokenizador de Laya (la prueba los corrige)"
    )
    print(
        f"Tokens {origen}; precio {args.precio} USD por millón de tokens de"
        f" entrada; horas a {args.rps} solicitudes/s."
    )
    return total_n


class Limitador:
    """Cubeta de fichas compartida: solicitudes por segundo y tokens por segundo."""

    def __init__(self, rps: float, tps: float):
        self.rps, self.tps = rps, tps
        self.cerrojo = threading.Lock()
        self.proxima = time.monotonic()
        self.fichas = tps
        self.ultima = time.monotonic()

    def esperar(self, tokens: int):
        """Bloquea hasta que haya lugar para una solicitud de `tokens` tokens."""
        tokens = min(tokens, self.tps)
        while True:
            with self.cerrojo:
                ahora = time.monotonic()
                self.fichas = min(
                    self.tps, self.fichas + (ahora - self.ultima) * self.tps
                )
                self.ultima = ahora
                if ahora >= self.proxima and self.fichas >= tokens:
                    self.proxima = max(self.proxima, ahora) + 1 / self.rps
                    self.fichas -= tokens
                    return
                espera = max(self.proxima - ahora, (tokens - self.fichas) / self.tps)
            time.sleep(max(espera, 0.001))

    def frenar(self, segundos: float):
        """Posterga todas las solicitudes (tras un 429)."""
        with self.cerrojo:
            self.proxima = max(self.proxima, time.monotonic() + segundos)


def segundos_retry_after(encabezados) -> float | None:
    """Espera indicada por `retry-after-ms` o `Retry-After` (segundos o fecha)."""
    if valor := encabezados.get("retry-after-ms"):
        try:
            return float(valor) / 1000
        except ValueError:
            pass
    if valor := encabezados.get("retry-after"):
        try:
            return float(valor)
        except ValueError:
            try:
                fecha = parsedate_to_datetime(valor)
                return max(0.0, (fecha - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError):
                pass
    return None


class Fatal(Exception):
    """Error que no se arregla reintentando (clave inválida o sin permiso)."""


def enviar(cliente: httpx.Client, linea: dict, args, limitador: Limitador) -> dict:
    """Envía una solicitud con reintentos; devuelve el registro a guardar."""
    cuerpo = linea["cuerpo"]
    if args.modelo:
        cuerpo = {**cuerpo, "model": args.modelo}
    registro = {"id": linea["id"], "tokens_estimados": linea["tokens_estimados"]}
    for intento in range(1, args.reintentos + 2):
        limitador.esperar(linea["tokens_estimados"])
        t0 = time.monotonic()
        espera = None
        try:
            r = cliente.post(ruta_api, json=cuerpo)
        except httpx.TransportError as exc:
            estado, error = None, repr(exc)
        else:
            estado, error = r.status_code, r.text[:500]
            if estado == 200:
                try:
                    respuesta = r.json()
                except ValueError:
                    respuesta = None
                if isinstance(respuesta, dict) and "answers" in respuesta:
                    return {
                        **registro,
                        "estado": 200,
                        "respuesta": respuesta,
                        "id_solicitud": r.headers.get("x-typesafe-request-id"),
                        "segundos": round(time.monotonic() - t0, 4),
                        "intentos": intento,
                        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
                    }
                estado, error = "invalida", r.text[:500]
            elif estado in fatales:
                raise Fatal(f"HTTP {estado}: {error}")
            elif estado not in reintentables:
                break  # 400, 413, 422...: reintentar no cambia nada
            else:
                espera = segundos_retry_after(r.headers)
        if intento > args.reintentos:
            break
        # Espera exponencial con azar; si el servidor indica `Retry-After`, se toma
        # como mínimo y se le suma el azar, para que los hilos no reintenten juntos
        exponencial = min(60.0, 0.5 * 2 ** (intento - 1)) * (0.5 + random.random())
        espera = (
            exponencial if espera is None else espera + random.random() * exponencial
        )
        if estado == 429:
            limitador.frenar(espera)
        time.sleep(espera)
    return {**registro, "estado": estado, "error": error, "intentos": intento}


def cliente_http(args) -> httpx.Client:
    """Cliente con la clave en `Authorization: Bearer` (si la hay)."""
    encabezados = {"Content-Type": "application/json"}
    if clave := os.environ.get("JEV_API_KEY"):
        encabezados["Authorization"] = f"Bearer {clave}"
    limites = httpx.Limits(max_connections=args.concurrencia * 2)
    return httpx.Client(
        base_url=args.url_base,
        headers=encabezados,
        timeout=args.timeout,
        limits=limites,
    )


def prueba(args, hechas: set[str]) -> int:
    """Envía 3 solicitudes (de regímenes distintos, si hay) y muestra el resultado."""
    primeras, por_regimen = [], {}
    for linea in pendientes(args, hechas):
        if len(primeras) < 3:
            primeras.append(linea)
        por_regimen.setdefault(linea["regimen"], linea)
        if len(por_regimen) >= 3:
            break
    elegidas = list(por_regimen.values())[:3]
    elegidas += [linea for linea in primeras if linea not in elegidas]
    elegidas = elegidas[:3]
    limitador = Limitador(args.rps, args.tps)
    with cliente_http(args) as cliente, open(args.respuestas_prueba, "a") as salida:
        for linea in elegidas:
            try:
                registro = enviar(cliente, linea, args, limitador)
            except Fatal as exc:
                print(f"Error fatal, revisá la clave: {exc}")
                return 1
            salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            print(f"\n{linea['id']} -> estado {registro['estado']}")
            if registro["estado"] == 200:
                respuesta = registro["respuesta"]
                print(json.dumps(respuesta, ensure_ascii=False, indent=2)[:1500])
                uso = respuesta.get("usage", {}).get("input_tokens")
                print(
                    f"tokens estimados {linea['tokens_estimados']}, cobrados {uso};"
                    f" {registro['segundos']} s"
                )
            else:
                print(registro.get("error"))
    print("\nEstimación corregida con lo observado:")
    estimar(args, hechas)
    return 0


def correr(args, hechas: set[str], total: int):
    """Envía todo lo pendiente con `concurrencia` hilos; guarda cada respuesta."""
    limitador = Limitador(args.rps, args.tps)
    cola = queue.Queue(maxsize=args.concurrencia * 4)
    alto = threading.Event()
    cerrojo = threading.Lock()
    cuenta = Counter()
    tokens_cobrados = [0]
    fatal = []

    def trabajar(cliente, salida):
        while True:
            linea = cola.get()
            if linea is None:
                return
            if alto.is_set():
                continue
            try:
                registro = enviar(cliente, linea, args, limitador)
            except Fatal as exc:
                fatal.append(str(exc))
                alto.set()
                continue
            with cerrojo:
                salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                salida.flush()
                cuenta["ok" if registro["estado"] == 200 else "error"] += 1
                cuenta["seguidos"] = (
                    0 if registro["estado"] == 200 else cuenta["seguidos"] + 1
                )
                uso = (registro.get("respuesta") or {}).get("usage") or {}
                tokens_cobrados[0] += uso.get("input_tokens", 0)
                if cuenta["seguidos"] >= args.max_errores_seguidos:
                    print(f"\n{cuenta['seguidos']} errores seguidos; me detengo.")
                    alto.set()

    inicio = ultimo_aviso = time.monotonic()
    with cliente_http(args) as cliente, open(args.respuestas, "a") as salida:
        hilos = [
            threading.Thread(target=trabajar, args=(cliente, salida), daemon=True)
            for _ in range(args.concurrencia)
        ]
        for hilo in hilos:
            hilo.start()
        try:
            for linea in pendientes(args, hechas):
                while not alto.is_set():
                    try:
                        cola.put(linea, timeout=1)
                        break
                    except queue.Full:
                        pass
                if alto.is_set():
                    break
                if time.monotonic() - ultimo_aviso > 30:
                    ultimo_aviso = time.monotonic()
                    hechos = cuenta["ok"] + cuenta["error"]
                    ritmo = hechos / (ultimo_aviso - inicio)
                    restante = (
                        (total - hechos) / ritmo / 3600 if ritmo else float("nan")
                    )
                    print(
                        f"{hechos}/{total} ({cuenta['error']} errores),"
                        f" {ritmo:.1f}/s, {tokens_cobrados[0] / 1e6:.1f} Mtokens"
                        f" ({tokens_cobrados[0] / 1e6 * args.precio:.2f} USD),"
                        f" faltan ~{restante:.1f} h",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nInterrumpido: termino las solicitudes en curso y salgo.")
            alto.set()
        for _ in hilos:
            cola.put(None)
        for hilo in hilos:
            hilo.join()
    if fatal:
        print(f"\nError fatal, revisá la clave: {fatal[0]}")
    print(
        f"\nListo: {cuenta['ok']} respuestas, {cuenta['error']} errores,"
        f" {tokens_cobrados[0] / 1e6:.2f} Mtokens"
        f" ({tokens_cobrados[0] / 1e6 * args.precio:.2f} USD)."
    )
    if cuenta["error"] or alto.is_set():
        print("Volvé a correr el mismo comando para completar lo pendiente.")
    return 1 if fatal else 0


def argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--solicitudes", type=Path, default=None)
    p.add_argument("--respuestas", type=Path, default=dir_script / "respuestas.jsonl")
    p.add_argument("--dry-run", action="store_true", help="envía 3 solicitudes")
    p.add_argument("--confirmar", action="store_true", help="envía todo lo pendiente")
    p.add_argument("--regimen", action="append", help="solo este régimen (repetible)")
    p.add_argument("--url-base", default=os.environ.get("JEV_URL_BASE", url_base_jev))
    p.add_argument("--modelo", help="reemplaza el campo `model` de cada solicitud")
    p.add_argument("--concurrencia", type=int, default=8)
    p.add_argument("--rps", type=float, default=30, help="solicitudes por segundo")
    p.add_argument("--tps", type=float, default=80_000, help="tokens por segundo")
    p.add_argument("--reintentos", type=int, default=10)
    p.add_argument("--max-errores-seguidos", type=int, default=50)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--precio", type=float, default=precio_por_millon)
    p.add_argument("--sin-clave", action="store_true", help="servidor local sin clave")
    args = p.parse_args(argv)
    if args.solicitudes is None:
        comprimidas = dir_script / "solicitudes.jsonl.gz"
        args.solicitudes = (
            comprimidas if comprimidas.exists() else comprimidas.with_suffix("")
        )
    args.respuestas_prueba = args.respuestas.with_name("respuestas-prueba.jsonl")
    return args


def main(argv=None) -> int:
    args = argumentos(argv)
    if not args.solicitudes.exists():
        print(f"No encuentro {args.solicitudes}")
        return 1
    if not (args.sin_clave or os.environ.get("JEV_API_KEY")):
        print("Falta la clave: export JEV_API_KEY=... (--sin-clave si no hace falta)")
        return 1
    print(f"Destino: {args.url_base}{ruta_api}")
    hechas = respondidas(args.respuestas)
    if args.dry_run:
        return prueba(args, hechas)
    total = estimar(args, hechas)
    if total == 0:
        print("No queda nada pendiente.")
        return 0
    if not args.confirmar:
        print("\nNada enviado. Para enviar lo pendiente: uv run correr.py --confirmar")
        return 0
    return correr(args, hechas, total)


if __name__ == "__main__":
    sys.exit(main())
