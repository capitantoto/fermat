import os
from pathlib import Path

dir_raiz = Path(__file__).parent.parent.absolute()
# Sobreescribible por entorno (lo heredan también los workers de loky): campañas o
# experimentos con un caché propio no inflan el global. Sin la variable, `.cache` en la
# raíz del repositorio; instalado como biblioteca, sin caché en disco: `None` y `fermat`
# no escribe nada. El repositorio se reconoce por su `pyproject.toml` y por un módulo de
# la investigación que el wheel excluye (copiado o instalado con `--target`, no está).
_en_repositorio = (dir_raiz / "pyproject.toml").is_file() and (
    dir_raiz / "fkdc" / "campanias.py"
).is_file()
dir_cache = (
    Path(os.environ["FKDC_CACHE"])
    if os.environ.get("FKDC_CACHE")
    else (dir_raiz / ".cache" if _en_repositorio else None)
)

# API pública de la biblioteca, cargada a demanda: `import fkdc` (y con él cada módulo
# de la investigación que importa `fkdc.config` o `dir_raiz`) no arrastra numpy ni
# sklearn
__all__ = [
    "EuclideanKDE",
    "FermatKDE",
    "FermatKNeighborsClassifier",
    "KDClassifier",
    "SampleFermatDistance",
]


def __getattr__(nombre):
    if nombre in __all__:
        from fkdc import fermat

        return getattr(fermat, nombre)
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")
