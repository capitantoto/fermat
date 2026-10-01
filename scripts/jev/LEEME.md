# Clasificación con Jev

Este paquete tiene solicitudes para la API de Jev (TypeSafe, `POST https://api.typesafe.ai/v1/systemone`, modelo `jev-latest`). Cada solicitud le pide a Jev que clasifique una observación de un dataset sintético o público (lunas, círculos, iris, pingüinos...), con texto de distinto nivel de contexto: solo la observación, el nombre del dataset, una descripción o el código que lo generó. No hay datos personales. Las respuestas se usan como línea de base en una tesis de maestría.

## Qué hace falta

- [uv](https://docs.astral.sh/uv/getting-started/installation/): `uv run` instala solo Python y la única dependencia (`httpx`).
- Una clave de la API de Jev.

## Pasos

1. Descomprimir y entrar: `unzip jev.zip && cd jev`
2. Cargar la clave en la terminal: `export JEV_API_KEY=...` (el script la lee de ahí y no la guarda en ningún lado).
3. Ver cuánto costaría, sin enviar nada: `uv run correr.py`
4. Probar con 3 solicitudes: `uv run correr.py --dry-run`. Muestra las respuestas y recalcula la estimación con los tokens que Jev efectivamente cobró.
5. Enviar todo: `uv run correr.py --confirmar`. Tarda horas: conviene dejarlo en una terminal que no se cierre (en macOS, `caffeinate -i uv run correr.py --confirmar` evita que la máquina se duerma).
6. Devolver las respuestas: `zip respuestas.zip respuestas.jsonl respuestas-prueba.jsonl`

Se puede cortar con Ctrl-C en cualquier momento: volviendo a correr el mismo comando, saltea lo ya respondido y reintenta lo que falló. Los errores transitorios (límite de tasa 429, 5xx, cortes de red) se reintentan solos con espera creciente.

## Costo y tiempo

`uv run correr.py` imprime solicitudes, millones de tokens, dólares y horas por régimen. Jev cobra 0,042 USD por millón de tokens de entrada; los límites documentados son 40 solicitudes y 100 000 tokens por segundo, y el script va por debajo (30 solicitudes/s). Si aparecen muchos 429, bajar el ritmo con `--rps 15`.

El paquete completo (cuatro regímenes, 600 tareas) son unas 815 000 solicitudes y unos 330 millones de tokens según el tokenizador de Laya; otros tokenizadores cuentan hasta un 40 % más. Del orden de 15 a 20 USD y 7 a 8 horas. La prueba corrige los tokens con lo que Jev efectivamente cobra.

Para enviar solo una parte: `--regimen descripcion` (repetible). Los regímenes son `zero-shot`, `nombre`, `descripcion` y `codigo`.

## Qué se envía

Cada línea de `solicitudes.jsonl.gz` tiene un `id` y, en `cuerpo`, el JSON exacto que se envía. Para ver una: `gzip -dc solicitudes.jsonl.gz | head -1`
