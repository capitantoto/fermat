// Carátula de tesis de Maestría en Estadística Matemática (UBA).
// Estructurada según el Anexo II de la Resolución 2265/18 del Consejo
// Directivo de la FCEN, tomando como referencia visual la portada de la
// tesis de Daniela L. Parada (2023, dirigida por Graciela Boente).
//
// Compilación:
//   typst compile docs/caratula.typ                    # sin firmas (default)
//   typst compile --input firmas=true docs/caratula.typ  # con firmas escaneadas
//
// Si `firmas=true` y existen las imágenes
// `docs/img/firma-gonzalo.png` y `docs/img/firma-pablo.png`,
// se insertan sobre las líneas de firma. Si no existen, igual se imprimen
// las líneas vacías (no rompe la compilación).

#let firmas = sys.inputs.at("firmas", default: "false") == "true"

#set page(paper: "a4", margin: (top: 1in, bottom: 1in, left: 1.5in, right: 1.5in), numbering: none)
#set text(font: "New Computer Modern", lang: "es", size: 11pt)
#set par(leading: 0.55em, justify: false)

// Helper para insertar una firma escaneada sobre la línea de firma.
// Si `firmas=true`, espera encontrar la imagen en la ruta indicada
// (Typst falla con error claro si no existe). Si `firmas=false`, no
// imprime nada y la línea queda en blanco para firma manual.
#let firma-img(path) = if firmas { image(path, height: 1.2cm) }

#align(center)[
  // Logo de la Facultad
  #image("img/Logo-fcenuba.png", height: 4cm)

  #v(0.5em)

  // Universidad y Facultad (en versalitas, como pide la tradición de la FCEN)
  #text(size: 12pt, smallcaps[Universidad de Buenos Aires])

  #text(size: 12pt, smallcaps[Facultad de Ciencias Exactas y Naturales])

  #v(0.6em)

  // Maestría (campo obligatorio del Anexo II)
  Maestría en Estadística Matemática

  #v(4em)

  // Título
  // Anexo II: "DESTACADO, centrado y deberá comenzar en mayúscula y continuar
  //  en minúscula, salvo que el título contenga nombres propios o vocabulario
  //  o tipografía específica".
  // Conservamos la capitalización del título tal como lo usa el autor en el
  // plan de tesis y a lo largo de la monografía.
  #text(size: 17pt, weight: "bold")[
    Distancia de Fermat en Clasificadores de Densidad por Núcleos
  ]

  #v(3.5em)

  // Tipo de trabajo (campo del Anexo II)
  Tesis presentada para optar al título de Magíster de la Universidad de Buenos Aires
  en Estadística Matemática

  #v(2.5em)

  // Maestrando
  #text(weight: "bold", size: 12pt)[Lic. Gonzalo Barrera Borla]
]

#v(1fr)

// Datos al pie (campos obligatorios del Anexo II)
#text(weight: "bold")[Director:] Dr. Pablo Groisman

#v(0.3em)

#text(weight: "bold")[Lugar de Trabajo:] Departamento de Matemática
e Instituto de Cálculo, FCEN, UBA

#v(0.6em)

#text(weight: "bold")[Fecha de presentación del ejemplar:] mayo de 2026

#text(weight: "bold")[Fecha de Defensa:] #h(0.4em) #box(width: 5cm, stroke: (bottom: 0.5pt))
#h(0.4em) #text(size: 9pt)[(se completará el día de la defensa)]

#v(3em)

#grid(
  columns: (1fr, 1fr),
  align: (center, center),
  gutter: 1em,
  [
    #box(width: 6cm, height: 1.5cm)[
      #align(center + bottom, firma-img("img/firma-gonzalo.png"))
    ]
    #box(width: 6cm, stroke: (top: 0.5pt), inset: (top: 0.3em))[
      Firma del maestrando
    ]
  ],
  [
    #box(width: 6cm, height: 1.5cm)[
      #align(center + bottom, firma-img("img/firma-pablo.png"))
    ]
    #box(width: 6cm, stroke: (top: 0.5pt), inset: (top: 0.3em))[
      Firma del director
    ]
  ],
)
