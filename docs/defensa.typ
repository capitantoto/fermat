#import "@preview/touying:0.6.2": *

#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: [Defensa de tesis · Maestría en Estadística Matemática · FCEN-UBA],
)
#show link: set text(blue)
#set text(font: "Fira Sans", weight: "light", size: 22pt, lang: "es")
#set strong(delta: 100)
#set par(justify: false)

#let bu(x) = $bold(upright(#x))$
#let fkdc = `fKDC`
#let kdc = `KDC`
#let fkn = `fkNN`
#let kn = `kNN`
#let chico(cuerpo) = text(size: 16pt, cuerpo)
#let fuente(cuerpo) = text(size: 13pt, fill: gray.darken(30%), cuerpo)

// La matemática va al pie, chica y al grano: el jurado la reconoce, el resto sigue de largo
#let silbato(cuerpo) = {
  v(1fr)
  block(width: 100%, fill: luma(245), inset: 10pt, radius: 4pt, text(size: 15pt, cuerpo))
}

// Lugar reservado para una figura que todavía no existe
#let pendiente(alto: 60%, cuerpo) = block(
  width: 100%,
  height: alto,
  stroke: (paint: gray, dash: "dashed"),
  radius: 4pt,
  inset: 12pt,
  align(center + horizon, text(size: 16pt, fill: gray.darken(20%))[*Pendiente:* #cuerpo]),
)

// Animación cuadro a cuadro: cada clic avanza un SVG de scripts/animaciones_defensa.py
#let animacion(prefijo, cuadros, alto: 85%) = alternatives(
  ..range(cuadros).map(i => {
    let n = if i < 10 { "0" + str(i) } else { str(i) }
    align(center, image(prefijo + "-" + n + ".svg", height: alto))
  }),
)

#let cifra(numero, rotulo) = align(center)[
  #text(size: 44pt, weight: "bold", numero) \
  #text(size: 16pt, rotulo)
]

#title-slide(
  author: [Lic. Gonzalo Barrera Borla · Director: Dr. Pablo Groisman],
  title: "Distancia de Fermat en Clasificadores de Densidad por Núcleos",
  subtitle: "Defensa de tesis de Maestría en Estadística Matemática",
  institution: [Departamento de Matemática, FCEN, UBA],
)

= Los personajes

== En alta dimensión, todo queda lejos

#grid(columns: (60%, 40%), column-gutter: 1em,
  image("img/curse-dim.svg"),
  align(horizon)[
    Una cajita de la mitad de ancho, en 20 dimensiones, guarda menos de una millonésima del volumen.

    #v(0.5em)
    Los vecinos dejan de ser vecinos: la _maldición de la dimensionalidad_.
  ],
)

#silbato[$X tilde.op "Unif"([-1, 1]^d)$: la fracción esperada dentro de $[-h, h]^d$ es $h^d$.]

== Y sin embargo, funciona

#grid(columns: (50%, 50%), column-gutter: 1em,
  image("img/digitos-mnist-ejemplos.png"),
  align(horizon)[
    Clasificamos imágenes, audio, video... y ahora también texto.

    #v(0.5em)
    *¿Cómo?*
  ],
)

== La hipótesis de la variedad

#grid(columns: (45%, 55%), column-gutter: 1em,
  image("img/hormiga-petalo.jpg"),
  align(horizon)[
    Los datos se registran en $RR^D$, pero viven cerca de una superficie de dimensión mucho menor.

    #v(0.5em)
    La hormiga camina en dos dimensiones, aunque el pétalo esté en tres.
  ],
)

#silbato[Es una afirmación sobre el espacio de probabilidad: $X : (Omega, cal(A), P) -> cal(M) subset RR^D$, con $dim cal(M) = d << D$.]

== Y hay teoría

#grid(columns: (40%, 60%), column-gutter: 1em,
  image("img/von-mises-s1.png"),
  align(horizon)[
    Se puede hacer estadística en variedades de Riemann compactas y sin frontera, adaptando los métodos euclídeos.
  ],
)

#silbato[
  $f_(N, K)(p) = 1 / N sum_(i=1)^N 1 / h^d 1 / (theta_(X_i)(p)) K((d_g (p, X_i)) / h)$ #h(1em) #fuente[Pelletier (2005)]
]

== Pero hay que conocer la variedad, o aprenderla

#grid(columns: 3, column-gutter: 1em, row-gutter: 0.6em,
  align(center, image("img/pca.png", height: 52%)),
  align(center, image("img/isomap-2.png", height: 52%)),
  align(center, image("img/distancia-basada-en-densidad.svg", height: 52%)),
  align(center)[*PCA* \ #chico[variedades lineales]],
  align(center)[*Isomap* \ #chico[el grafo de vecinos aproxima la variedad]],
  align(center)[*Fermat* \ #chico[caminos que prefieren la densidad]],
)

#chico[El aprendizaje de variedades es un caso del _aprendizaje de representaciones_, que es lo que hacen las arquitecturas profundas de hoy.]

== La distancia de Fermat

#grid(columns: (45%, 55%), column-gutter: 1em,
  image("img/distancia-cuadrada.svg"),
  align(horizon)[
    Elevar las distancias a $alpha > 1$ abarata los saltos cortos: el camino más barato pasa por donde hay datos.

    #v(0.5em)
    Y $alpha$ se puede elegir por validación cruzada, sin conocer la dimensión de la variedad.
  ],
)

#silbato[
  $D_(bu(X), alpha)(x, y) = inf sum_j norm(q_(j+1) - q_j)^alpha$ #h(0.5em) y #h(0.5em) $n^beta D_(bu(X)_n, alpha) -> mu integral_gamma f^(-beta)$, con $beta = (alpha - 1) slash d$
  #h(1em) #fuente[Groisman, Jonckheere y Sapienza (2022); Bijral et al. (2011)]
]

== Fermat en imágenes: densidades

// TODO(autor): con espirales_lo bien muestreado, Fermat no reduce el solapamiento (0.19 contra 0.08); ¿mover a «Dónde no» o quitar?
#align(center, image("img/defensa/densidades-espirales.svg", height: 80%))
#chico[Densidad de cada clase sobre una muestra nueva de `espirales_lo`, con el ancho de banda elegido por validación cruzada.]

== Fermat en imágenes: vecindarios

#animacion("img/defensa/anim/vecindario-espirales", 12)

#focus-slide[¿Servirá para clasificar?]

= El plan

== Del estimador al clasificador

#grid(columns: 4, column-gutter: 0.8em,
  [*2005* \ #chico[Pelletier: densidad por núcleos en variedades]],
  [*2008* \ #chico[Loubes y Pelletier: clasificador binario]],
  [*2009* \ #chico[Henry y Rodríguez: mejores cotas, y la esfera]],
  [*Esta tesis* \ #chico[multiclase, sobre una variedad *desconocida*]],
)

#v(1em)
#align(center, image("img/henry-rodriguez-bolas.png", height: 45%))

== La propuesta: #fkdc y #fkn

- La distancia de Fermat _muestral_ reemplaza a la de la variedad, que no conocemos.
- #fkdc: clasificador de densidad por núcleos con esa distancia. #fkn: $k$ vecinos más cercanos con la misma.
- Con $alpha = 1$, son los euclídeos de siempre.

#silbato[
  $hat(Pr)(G = g_i | X = x) prop hat(pi)_i hat(f)_i (x)$, con $hat(f)_i$ evaluada sobre $D_(Q_i, alpha)(x, dot)$ y $Q_i = {x} union {x_j : g_j = g_i}$
]

== El código

#grid(columns: (50%, 50%), column-gutter: 1em,
  [
    - Estimadores compatibles con `scikit-learn`: `KDClassifier`, `FermatKNeighborsClassifier`.
    - Cada experimento es una _tarea_: dataset, clasificador, semilla, grilla.
    - Todo reproducible desde un `Makefile`.
  ],
  pendiente(alto: 70%)[publicar como biblioteca en PyPI, después de pulirla],
)

== El banco de pruebas

#grid(columns: 4, column-gutter: 0.5em,
  image("img/lunas_lo-scatter.svg"),
  image("img/espirales_lo-scatter.svg"),
  image("img/helices_0-scatter.svg"),
  image("img/hueveras_0-scatter.svg"),
)

#align(center)[
  #grid(columns: 3, column-gutter: 3em,
    cifra("20", "datasets: sintéticos y reales chicos"),
    cifra("8", "clasificadores"),
    cifra("25", "repeticiones cada uno"),
  )
]

= Resultados

== Sin escalar, nada anda

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/pinguinos-crudo-vs-std-r2.svg"),
  align(horizon)[
    En `pinguinos`, un atributo en gramos tapa a los demás, en milímetros.

    #v(0.5em)
    Fermat es tan sensible a la escala como la euclídea. Estandarizar lo resuelve.
  ],
)

== Fermat agranda el vecindario sin cruzar

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/r2-fkn-kn-helices_0.svg"),
  align(horizon)[
    En `helices_0`, con el mismo $k$, #fkn le gana a #kn en la gran mayoría de las semillas.

    #v(0.5em)
    Es lo que mostraba la animación: el vecindario sigue la hélice.
  ],
)

== Dónde gana: pocos puntos, mucha curva

#grid(columns: (45%, 55%), column-gutter: 1em,
  image("img/espirales_lo-fkdc-decision_boundary.svg"),
  align(horizon)[
    Con 100 semillas, la ventaja se sostiene:

    #chico[
    - #fkdc sobre #kdc: `lunas_lo` (73 a 0) y `espirales_lo` (53 a 11).
    - #fkn sobre #kn: `helices_0` (83 a 8) y `hueveras_0` (61 a 15).
    ]

    #v(0.4em)
    Ganancias marginales, pero reales.
  ],
)

== Dónde no: bien muestreado, $alpha$ y $h$ se confunden

#grid(columns: (50%, 50%), column-gutter: 1em,
  image("img/espirales_lo-1434-fkdc-bandwidth-alpha-loss_contour.svg"),
  align(horizon)[
    De cerca, la distancia de Fermat es la euclídea con otra escala.

    #v(0.5em)
    Aparece un «risco» de combinaciones equivalentes, y la regla de parsimonia elige $alpha = 1$.
  ],
)

= El futuro

== Después de la entrega

#grid(columns: 4, column-gutter: 1em,
  cifra("100", "semillas, prerregistradas"),
  cifra("72 600", "tareas, 0 fallas"),
  cifra("22", "clasificadores"),
  cifra("33", "datasets"),
)

#v(1em)
#align(center)[Una réplica más grande, y la comparación con el estado del arte.]

// TODO(autor): elegir entre una y tres observaciones; candidatas: calibración, TabPFN, atención

== Parte de la ventaja era calibración

#grid(columns: (58%, 42%), column-gutter: 1em,
  image("img/defensa/fkdc-kdc-crudo-templado.svg"),
  align(horizon)[
    #kdc es poco confiado; $alpha$ agudiza las probabilidades.

    #v(0.5em)
    Recalibrando a ambos, la ventaja sobrevive en `lunas_lo` y `espirales_lo`, no en el resto.
  ],
)

== TabPFN: el estado del arte

#grid(columns: (58%, 42%), column-gutter: 1em,
  image("img/defensa/mapa-tabpfn.svg"),
  align(horizon)[
    Una red preentrenada con millones de tablas simuladas: queda entre los mejores en 19 de 33 datasets.

    #v(0.5em)
    Los métodos de la tesis aprenden de 400 puntos.
  ],
)

// Sin fuga ni contaminación: ver «¿El resultado de TabPFN es legítimo?» en preguntas-defensa.typ

== El #kdc es una cabeza de atención

#align(center + horizon)[
  Una cabeza de atención de un _transformer_ hace, cuenta por cuenta, lo mismo que un #kdc gaussiano.

  #v(0.5em)
  #fkdc fija el núcleo con una regla; TabPFN lo _aprende_.
]

#silbato[
  $hat(Pr)(dot | x) = sum_i "softmax"_i (- norm(x - x_i)^2 / (2 h^2)) e_(g_i)$ #h(0.5em) frente a #h(0.5em) $sum_i "softmax"_i (q^top k_i slash sqrt(d)) v_i$
]

== ¿Y la IA?

#align(center + horizon)[
  Las redes profundas aprenden representaciones de dominios enormes.

  #v(0.8em)
  *¿Puede la geometría de Riemann ayudarnos a entenderlas mejor?*
]

#focus-slide[¡Gracias!]

= Respaldo

== Lo que da conocer la variedad

#let filas-oraculo = csv("data/defensa/oraculo.csv")
#align(center)[
  #set text(size: 16pt)
  #table(
    columns: filas-oraculo.first().len(),
    align: (left, right, right, right, right),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + gray.lighten(40%)) },
    table.header(..filas-oraculo.first().map(c => strong(c))),
    ..filas-oraculo.slice(1).flatten(),
  )
]

#chico[El oráculo invierte el proceso generador de cada dataset sintético y ajusta una regresión logística sobre las coordenadas intrínsecas. Mediana de $R^2$, 25 semillas.]

== Modelos de texto: al azar

#chico[
- Laya, GLiNER2.5-Decide y una réplica abierta de Jev reciben cada fila como texto.
- En todos los regímenes y datasets quedaron en el nivel del azar.
- Son modelos de un solo paso: decidir de qué círculo viene $(x_1, x_2)$ exige calcular $sqrt(x_1^2 + x_2^2)$ en una pasada.
]

== Fronteras de decisión en `espirales_lo`

#grid(columns: 4, column-gutter: 0.5em,
  image("img/espirales_lo-kn-decision_boundary.svg"),
  image("img/espirales_lo-fkn-decision_boundary.svg"),
  image("img/espirales_lo-kdc-decision_boundary.svg"),
  image("img/espirales_lo-fkdc-decision_boundary.svg"),
)
