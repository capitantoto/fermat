"""
La defensa como PowerPoint, para presentar con la vista del orador.

Cada página de `docs/defensa.pdf` (un paso de cada diapo) es una diapo con la imagen a
pantalla completa; sus notas son el guion de esa diapo, tomado de la mitad derecha de
`docs/defensa-guion.pdf`. Uso, desde la raíz del repositorio, con ambos PDF al día:
    uv run --with python-pptx python scripts/defensa_pptx.py [salida.pptx]
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

dir_docs = Path(__file__).resolve().parent.parent / "docs"
pdf_diapos = dir_docs / "defensa.pdf"
pdf_guion = dir_docs / "defensa-guion.pdf"


def paginas(pdf):
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    return int(
        next(lin for lin in info.splitlines() if lin.startswith("Pages:")).split()[1]
    )


def guion(pagina, ancho_pt=842, alto_pt=474):
    """Texto de la mitad derecha de la página del guion, en párrafos.

    pdftotext corta en renglones: un renglón empieza párrafo si abre con ▸ o [, o si el
    anterior terminó una oración sin llenar el ancho.
    """
    texto = subprocess.run(
        ["pdftotext", "-f", str(pagina), "-l", str(pagina), "-x", str(ancho_pt),
         "-y", "0", "-W", str(ancho_pt), "-H", str(alto_pt), str(pdf_guion), "-"],
        capture_output=True, text=True, check=True,
    ).stdout.replace("\f", "")  # fmt: skip
    renglones = [r.strip() for r in texto.splitlines() if r.strip()]
    ancho = max((len(r) for r in renglones), default=0)
    parrafos, ultimo = [], ""
    for renglon in renglones:
        anterior = parrafos[-1] if parrafos else None
        corte = (
            anterior is None
            or renglon.startswith(("▸", "[", "—"))
            or anterior.startswith("—")
            or (len(ultimo) < 0.8 * ancho and ultimo.endswith((".", "?", ":", "!")))
        )
        if corte:
            parrafos.append(renglon)
        else:
            parrafos[-1] += " " + renglon
        ultimo = renglon
    return "\n\n".join(parrafos)


def main():
    salida = Path(sys.argv[1]) if len(sys.argv) > 1 else dir_docs / "defensa.pptx"
    total = paginas(pdf_diapos)
    assert total == paginas(pdf_guion), "defensa.pdf y defensa-guion.pdf no coinciden"
    presentacion = Presentation()
    presentacion.slide_width, presentacion.slide_height = Emu(12192000), Emu(6858000)
    vacia = presentacion.slide_layouts[6]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdftoppm", "-png", "-scale-to-x", "1920", "-scale-to-y", "1080",
             str(pdf_diapos), f"{tmp}/p"],
            check=True,
        )  # fmt: skip
        imagenes = sorted(Path(tmp).glob("p-*.png"))
        assert len(imagenes) == total
        for pagina, imagen in enumerate(imagenes, 1):
            diapo = presentacion.slides.add_slide(vacia)
            diapo.shapes.add_picture(
                str(imagen), 0, 0, presentacion.slide_width, presentacion.slide_height
            )
            diapo.notes_slide.notes_text_frame.text = guion(pagina)
        presentacion.save(salida)
    print(f"{salida}: {total} diapos")


if __name__ == "__main__":
    main()
