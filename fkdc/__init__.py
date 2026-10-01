import os
from pathlib import Path

dir_raiz = Path(__file__).parent.parent.absolute()
# Sobreescribible por entorno (lo heredan también los workers de loky): campañas o
# experimentos con un caché propio no inflan el global. Sin la variable, `.cache`.
dir_cache = Path(os.environ.get("FKDC_CACHE") or dir_raiz / ".cache")
