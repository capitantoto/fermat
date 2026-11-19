#!/bin/zsh
# Genera la invitación a la defensa en salida/ (PNG y JPG).
# Requiere uv, typst (con Comic Sans MS instalada), ImageMagick y curl.
# Uso: docs/invitacion/hacer.sh
set -e
cd "${0:A:h}"
repo_dir=${0:A:h:h:h}
mkdir -p salida

# Plantilla del meme (600 × 429), ampliada 3× para que el texto no se pixele
[[ -f plantilla.jpg ]] || curl -sfo plantilla.jpg https://i.imgflip.com/65pv3s.jpg
magick plantilla.jpg -filter Lanczos -resize 300% fondo.png

# Máscara: los píxeles casi blancos del recuadro, sin el payaso ni los globos
magick fondo.png -fuzz 7% -fill white -opaque white -fill black +opaque white \
  -morphology Erode Disk:2 \
  \( +clone -fill black -colorize 100 -fill white -draw "rectangle 150,165 1650,1119" \) \
  -compose multiply -composite mascara.png

(cd "$repo_dir" && uv run python docs/invitacion/datasets.py docs/invitacion)

# Lunas centradas en el recuadro blanco, ensanchadas, y enmascaradas detrás del payaso
cat > puntos.typ <<'TYP'
#set page(width: 600pt, height: 429pt, margin: 0pt, fill: none)
#place(top + left, dx: 50pt, dy: 55pt, box(width: 500pt, height: 318pt,
  align(center + horizon, image("lunas.svg", width: 94%, height: 92%, fit: "stretch"))))
TYP
typst compile --format png --ppi 216 puntos.typ puntos.png
magick puntos.png \( +clone -alpha extract mascara.png -compose multiply -composite \) \
  -alpha off -compose copy_opacity -composite puntos-enmascarados.png
magick fondo.png puntos-enmascarados.png -compose over -composite fondo-con-puntos.png

# El texto encima
typst compile --format png --ppi 216 invitacion.typ salida/invitacion-defensa.png
magick salida/invitacion-defensa.png -quality 90 salida/invitacion-defensa.jpg
ls salida
