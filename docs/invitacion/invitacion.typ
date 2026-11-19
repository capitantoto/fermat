// Invitación a la defensa sobre la plantilla «te invito a» (ver hacer.sh)
#set page(width: 600pt, height: 429pt, margin: 0pt, background: image("fondo-con-puntos.png", width: 100%, height: 100%))
#set text(font: "Comic Sans MS", size: 14pt, fill: rgb("#2B3A55"), lang: "es")
#set par(leading: 0.55em)

#let rojo = rgb("#E5304A")
#let azul = rgb("#3A7CC9")

// El dataset de fondo va en fondo-con-puntos.png, enmascarado detrás del payaso
#place(top + left, dx: 48pt, dy: 56pt, block(width: 350pt, height: 318pt, align(center + horizon)[
  #text(size: 26pt, fill: rojo, weight: "bold")[¡Te invito a mi defensa!]
  #v(2pt)
  #text(size: 16.5pt, fill: azul, weight: "bold")[«Distancia de Fermat en clasificadores\ de densidad por núcleos» 🤯]
  #v(10pt)
  #grid(columns: (auto, auto), column-gutter: 8pt, row-gutter: 7pt, align: left,
    [🗓️], [*Viernes 20/11, 12 h*],
    [📍], [Pabellón 0+∞, Ciudad Universitaria],
    [🚪], [Aula 2119 (a confirmar)],
    [🥂], [Si apruebo, brindamos con sanguchitos],
  )
  #v(10pt)
  No hace falta entender nada para aplaudir.\
  #text(size: 11pt)[*Confirmá respondiendo este mensaje* 🙋]
]))
