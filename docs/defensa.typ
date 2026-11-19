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
#let gnb = `GNB`
#let logr = `LR`
#let gbt = `GBT`
#let svc = `SVC`
#let chico(cuerpo) = text(size: 16pt, cuerpo)
#let fuente(cuerpo) = text(size: 13pt, fill: gray.darken(30%), cuerpo)

// Doble público: el cuerpo de la diapo es técnico sin abrumar; el recuadro al pie da
// la intuición para quien no lee matemática
#let intuicion(cuerpo) = {
  v(1fr)
  block(width: 100%, fill: luma(245), inset: 10pt, radius: 4pt, text(size: 16pt, cuerpo))
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

// Línea de tiempo al tope de la diapo: los hitos en una recta, los actuales resaltados
// (`actual` es un índice o un arreglo de índices)
#let cronologia(hitos, actual) = block(width: 100%, below: 1em, {
  let n = hitos.len()
  let actuales = if type(actual) == array { actual } else { (actual,) }
  place(top + left, dx: 50% / n, dy: 4.25pt, line(length: 100% - 100% / n, stroke: 1.5pt + luma(200)))
  grid(columns: (1fr,) * n, ..hitos.enumerate().map(((i, (fecha, nombre))) => {
    let resaltado = i in actuales
    let color = if resaltado { rgb("#eb811b") } else { luma(160) }
    align(center, stack(spacing: 5pt,
      circle(radius: 5pt, fill: color, stroke: none),
      text(size: 15pt, fill: color, weight: if resaltado { "bold" } else { "light" }, fecha),
      text(size: 13pt, fill: color, nombre),
    ))
  }))
})

#title-slide(
  author: [Lic. Gonzalo Barrera Borla · Director: Dr. Pablo Groisman],
  title: "Distancia de Fermat en Clasificadores de Densidad por Núcleos",
  subtitle: "Defensa de tesis de Maestría en Estadística Matemática",
  institution: [Departamento de Matemática, FCEN, UBA],
)

= Enunciación

== En alta dimensión, todo queda lejos

#grid(columns: (55%, 45%), column-gutter: 1em,
  align(horizon, image("img/curse-dim.svg", height: 58%)),
  align(horizon)[
    Con $X tilde.op "Unif"([-1, 1]^D)$, la fracción dentro de $[-h, h]^D$ es $h^D -> 0$: $0.9^15 approx 0.2$, $0.9^130 approx 10^(-6)$.

    #pause
    #v(0.3em)
    #chico[Las distancias se _concentran_: en `mnist`, ya reducido a 96 dimensiones, la distancia mediana al vecino más cercano ronda $1300$, y entre dos puntos cualesquiera, $2500$.]
  ],
)

#intuicion[Una cajita del 90 % del ancho guarda, en 15 dimensiones, un quinto del volumen; en 130, una millonésima. Los vecinos dejan de ser vecinos: es la _maldición de la dimensionalidad_.]

== Y sin embargo, funciona: la hipótesis de la variedad

#grid(columns: (45%, 55%), column-gutter: 1em,
  box(radius: 0.5em, clip: true, image("img/bandera-argentina.png")),
  align(horizon)[
    Un dígito de `mnist` tiene $784$ dimensiones; una foto de teléfono, $36$ millones. Y se clasifican.

    #pause
    #v(0.4em)
    Los datos se registran en $RR^D$, pero viven cerca de una _variedad_ de dimensión mucho menor:
    $ X : (Omega, cal(A), P) -> cal(M) subset RR^D, quad dim cal(M) = d << D. $
  ],
)

#intuicion[Una _variedad_ es un espacio potencialmente de lo más rebuscado, pero que localmente se comporta como uno euclídeo: de cerca, la bandera al viento es una hoja plana, de dimensión $d = 2$ en el espacio físico, con $D = 3$.]

#let hitos-teoria = (
  ([1918], [von Mises: $S^1$]),
  ([1953], [Fisher: $S^2$]),
  ([1975], [Mardia: $S^n$]),
  ([2005], [Pelletier: KDE]),
  ([2008], [Loubes, Pelletier]),
  ([2009], [Henry, Rodríguez]),
)

== Y hay teoría: densidad

#cronologia(hitos-teoria, 3)
#grid(columns: (1fr, 1fr), column-gutter: 1.5em,
  [
    *En $RR^d$*
    $ hat(f)(x) = 1 / N sum_(i=1)^N 1 / h^d K((norm(x - X_i)) / h) $
  ],
  [
    #pause
    *En $(cal(M), g)$, compacta y sin frontera*
    $ hat(f)(p) = 1 / N sum_(i=1)^N 1 / h^d 1 / (theta_(X_i)(p)) K((d_g (p, X_i)) / h) $
  ],
)

#pause
#align(center)[La norma pasa a ser la distancia geodésica $d_g$; lo nuevo es la _densidad de volumen_ $theta$. Y converge a la velocidad de la dimensión _intrínseca_: $O(n^(-4 slash (d + 4)))$.]

#intuicion[La misma receta: una campana sobre cada dato, y se suman. Cambia la regla con que medimos, que ahora sigue la variedad, y una corrección por cuánto se estira el espacio.]

== Y hay teoría: clasificación

#cronologia(hitos-teoria, (4, 5))
#align(center + horizon)[
  Clasificador por la regla de Bayes:
  $ hat(G)(p) = arg max_(k) sum_(i=1)^N bb(1)(G_i = k) K_h (p, X_i) $
  #chico[Fuertemente consistente con dos clases (Loubes y Pelletier, 2008).]

  #pause
  #v(0.4em)
  Consistencia fuerte uniforme, $sup_(p in cal(M)) abs(hat(f)_n (p) - f(p)) -> 0$ c.s., y la densidad de volumen explícita en $S^2$.
  #chico[Henry y Rodríguez (2009)]
]

#intuicion[Cada clase vota con la suma de sus campanas en el punto $p$, y gana la que más pesa.]

#focus-slide[¿Y si no conocemos la variedad? \ ¡Hay que aprenderla!]

#let hitos-aprendizaje = (
  ([1901], [PCA]),
  ([2000], [Isomap]),
  ([2011--2022], [Fermat]),
)

== Aprender la variedad

#only(1, cronologia(hitos-aprendizaje, 0))#only(2, cronologia(hitos-aprendizaje, 1))#only(3, cronologia(hitos-aprendizaje, 2))
#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1em,
  align(center)[
    #image("img/pca.png", height: 30%)
    #chico[Variedades _lineales_: las direcciones de mayor variabilidad.]
  ],
  uncover("2-", align(center)[
    #image("img/isomap-2.png", height: 30%)
    #chico[Geodésicas por caminos mínimos en el grafo de $k$ vecinos.]
  ]),
  uncover("3-", align(center)[
    #image("img/distancia-cuadrada.svg", height: 30%)
    #chico[Caminos en el grafo completo, con aristas elevadas a $alpha >= 1$.]
  ]),
)
#fuente[Pearson (1901); Tenenbaum, de Silva y Langford (2000); Bijral, Ratliff y Srebro (2011); Groisman, Jonckheere y Sapienza (2022)]

#uncover("3-", intuicion[Elevar las distancias a una potencia abarata los saltos cortos: el camino más barato pasa por donde hay muchos datos.])

== Paseos al azar

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1em,
  align(center)[#image("img/defensa/paseos-espirales-euclidea.svg", height: 62%) Euclídea],
  align(center)[#image("img/defensa/paseos-espirales-fermat.svg", height: 62%) Fermat, $alpha = 4$],
  align(horizon)[
    #pause
    100 paseos de 15 saltos desde #text(fill: rgb("#d4a017"))[★], sobre el grafo completo:
    $ Pr(x_i -> x_j) prop 1 / "dist"(x_i, x_j) $

    Cambios de brazo por paseo: $6.3$ contra $1.1$.
  ],
)
// Applet para jugar con los parámetros: docs/paseos-fermat.html

#intuicion[Con la distancia de siempre, el paseante salta de una espiral a la otra a cada rato; con la de Fermat, sigue el brazo en el que está.]

== Fermat macroscópica y microscópica

#grid(columns: (1fr, 1fr), column-gutter: 1.5em,
  [
    *Macroscópica*, sobre curvas rectificables $gamma$ de $x$ a $y$:
    $ cal(D)_(f, beta)(x, y) = inf_gamma integral_gamma f^(-beta) dif s $
  ],
  [
    *Microscópica*, sobre caminos en la muestra $Q$:
    $ D_(Q, alpha)(x, y) = inf sum_j norm(q_(j+1) - q_j)^alpha $
  ],
)

#pause
#align(center)[
  Si $cal(M)$ es isométrica embebida y $Q_n tilde.op^"iid" f$, con $beta = (alpha - 1) slash d$:
  $ n^beta D_(Q_n, alpha) -> mu thin cal(D)_(f, beta) "c.s." $
  #fuente[Groisman, Jonckheere y Sapienza (2022), Teorema 2.7]
]

// Observación 2 del jurado (Jonckheere): el desbalance de clases con distancias por clase
#pause
#chico[*Ojo con la escala:* $D_(Q_n, alpha) approx n^(-beta)$. Si cada clase aprende su propia distancia y $alpha > 1$, la clase mayor queda «más cerca» de todo. En `pinguinos` y `vino`, desbalanceados, la validación cruzada eligió $alpha = 1$.]

#intuicion[Un camino cuesta más donde hay pocos datos. Sumar las distancias infladas a lo largo del grafo alcanza para aproximar ese costo, y con más datos, la aproximación mejora.]

#focus-slide[¿Servirá para clasificar?]

= El plan

== El clasificador #fkdc

El de Loubes y Pelletier, _multiclase_ y sobre una variedad _desconocida_. Para cada clase $i$, con $Q_i = {x} union {x_j : g_j = i}$ y $N_i = abs(Q_i) - 1$:
$ hat(f)_i (x) = 1 / N_i sum_(j : g_j = i) 1 / h^D K((D_(Q_i, alpha)(x, x_j)) / h), quad K(u) = (2 pi)^(-D slash 2) e^(-u^2 slash 2) $
#pause
$ hat(Pr)(G = i | X = x) = (hat(pi)_i hat(f)_i (x)) / (sum_k hat(pi)_k hat(f)_k (x)) $

#pause
#chico[La distancia de Fermat muestral reemplaza a la geodésica, y $theta equiv 1$. $(h, alpha)$ por validación cruzada con la regla de un desvío; con $alpha = 1$, es el #kdc euclídeo de siempre. *#fkn*: los $k$ más cercanos según la misma $D_(Q_i, alpha)$ votan por su clase.]

#intuicion[Para clasificar un punto nuevo, lo sumamos a cada clase y medimos con Fermat cuánto lo «abriga» cada una. Gana la que más lo abriga.]

== El código

#text(size: 17pt)[
- `SampleFermatDistance(Q, alpha=1, grupos=None)`: distancias de Fermat muestrales a través del grafo completo de `Q`, con un algoritmo incremental para puntos fuera de la muestra.
  - `FermatKNeighborsClassifier(n_neighbors=5, alpha=1)` reutiliza `KNeighborsClassifier(metric="precomputed")` de `scikit-learn`.
- `KDClassifier(bandwidth=1.0, metric="euclidean", alpha=1.0)`: densidad por núcleos con distancia configurable.
  - #fkdc es `KDClassifier(metric="fermat")`.
- Un marco de evaluación con `Pipeline` y `GridSearchCV` de `scikit-learn`, que compara sistemáticamente los clasificadores en repetidas `Tarea`s sobre datasets definidos de antemano.
]

// TODO(autor): reemplazar por el comando y el enlace reales cuando la biblioteca esté en PyPI
#align(center, text(size: 17pt, fill: gray.darken(20%))[*Pendiente:* `pip install fkdc` y enlace a PyPI])

#intuicion[Todo se enchufa a `scikit-learn`, la caja de herramientas estándar de aprendizaje automático en Python: los clasificadores nuevos y los de siempre se evalúan exactamente con el mismo procedimiento.]

== 8 clasificadores, 25 repeticiones

#grid(columns: (1fr, 1.2fr), column-gutter: 2em,
  [
    *La familia de los núcleos*
    - #fkdc y #kdc: densidad por núcleos
    - #fkn y #kn: $k$ vecinos

    *De referencia*
    - #gnb, #logr, #gbt
    - #svc, solo exactitud
    #pause
  ],
  [
    25 semillas por tarea; comparamos la _mediana_ del $R^2$ de McFadden:
    $ R^2 = 1 - (cal(l)(hat(G))) / (cal(l)(hat(G)_0)) $
    #chico[$cal(l)$: log-verosimilitud promedio; $hat(G)_0$ asigna las frecuencias de clase. En los sintéticos, muestras nuevas; en los reales, particiones nuevas.]
  ],
)

#intuicion[Cuatro métodos que deciden por cercanía, con y sin Fermat, contra cuatro que no. Miramos el resultado _del medio_ de 25. $R^2 = 0$: no mejor que adivinar con las proporciones de cada clase; $R^2 = 1$: acierta siempre, y sin dudar.]

== 20 datasets

#let muestra(ruta) = align(center + horizon, image(ruta, height: 30%))
#grid(columns: 4, column-gutter: 0.5em, row-gutter: 0.4em,
  muestra("img/lunas_lo-scatter.svg"),
  muestra("img/circulos_lo-scatter.svg"),
  muestra("img/espirales_lo-scatter.svg"),
  muestra("img/anteojos-scatter.svg"),
  muestra("img/helices-scatter-3d.svg"),
  muestra("img/hueveras-scatter-3d.svg"),
  muestra("img/pinguinos-scatter.svg"),
  muestra("img/digitos-mnist-ejemplos.png"),
)

#chico[Sintéticos en $D = 2$, con poco y mucho ruido (`_lo`, `_hi`), más `anteojos`, con tres clases; y en $D = 3$, con y sin 12 dimensiones de ruido (`_0`, `_12`). Reales y chicos: `iris`, `vino`, `pinguinos`, `digitos` y `mnist`.]

= Resultados

#let leyenda-resumen = chico[#text(fill: rgb("#1a9641"))[▲ mejor] #h(0.6em) #text(fill: rgb("#9a9ea4"))[● igual] #h(0.6em) #text(fill: rgb("#d7191c"))[▼ peor] #h(1em) A es peor que B si la mediana de su $R^2$ queda bajo el primer cuartil de B, como en las tablas resumen de la tesis.]

== En resumen: #fkdc

#align(center, image("img/defensa/resumen-fkdc.svg", width: 100%))
#leyenda-resumen

#intuicion[Contra su versión euclídea, #fkdc gana en 6 datasets y pierde en 2. Contra el mejor de los demás, gana en 3 y pierde en 7.]

== En resumen: #fkn

#align(center, image("img/defensa/resumen-fkn.svg", width: 100%))
#leyenda-resumen

#intuicion[Contra su versión euclídea, #fkn gana en 4 datasets y no pierde en ninguno. Contra el mejor de los demás, nunca gana.]

== Las fronteras en `espirales_lo`

#grid(columns: 4, column-gutter: 0.4em, row-gutter: 0.3em,
  ..("kn", "fkn", "kdc", "fkdc", "gnb", "lr", "svc", "gbt").map(c => image("img/espirales_lo-" + c + "-decision_boundary.svg")),
)

#intuicion[Arriba, #kn, #fkn, #kdc y #fkdc siguen la espiral. Abajo, #gnb y #logr no pueden, #svc la sigue con una frontera dura, sin probabilidades, y #gbt la aproxima con rectángulos.]

== Ablación: con y sin Fermat

#let ablacion(par, c) = image("img/" + c + "_lo-" + par + "-r2-scatter.svg", height: 25%)
#grid(columns: (auto, 1fr, 1fr, 1fr), column-gutter: 0.4em, row-gutter: 0.2em, align: center + horizon,
  [], raw("lunas_lo"), raw("circulos_lo"), raw("espirales_lo"),
  [#kdc \ vs. \ #fkdc], ..("lunas", "circulos", "espirales").map(c => ablacion("kdc-fkdc", c)),
  uncover("2-")[#kn \ vs. \ #fkn], ..("lunas", "circulos", "espirales").map(c => uncover("2-", ablacion("kn-fkn", c))),
)

#chico[$R^2$ por semilla, sin Fermat ($x$) y con Fermat ($y$). #fkdc parece ganar en `lunas_lo` y `espirales_lo`. #uncover("2-")[#fkn y #kn casi coinciden, salvo en `espirales_lo`: #fkn gana en algunas semillas, pero en dos cae a $R^2 approx 0$.]]

#intuicion[¿Qué se pierde al _quitar_ la distancia de Fermat? Cada punto es una repetición: por encima de la diagonal, Fermat ayudó.]

== Bien muestreado, $alpha$ y $h$ se confunden

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/circulos_lo-8527-fkdc-bandwidth-alpha-loss_contour.svg"),
  align(horizon)[
    Superficie de pérdida de #fkdc en `circulos_lo`: un «risco» de parametrizaciones equivalentes, $log h prop alpha$.

    #pause
    #v(0.5em)
    Localmente, $D_(Q, alpha) approx c norm(dot)$, y lo que $alpha$ cambia en la distancia lo compensa $h$. La regla de parsimonia elige $alpha = 1$, y #fkdc se reduce a #kdc.
  ],
)

#intuicion[Donde los datos son densos, el camino más barato entre vecinos es el salto directo: Fermat no logra nada que no se logre ajustando el ancho de la campana.]

== ¿De dónde sale la ventaja en `lunas_lo`?

#grid(columns: (50%, 50%), column-gutter: 1em,
  align(center + horizon, image("img/lunas_lo-[f]kdc-delta_r2-vs-delta_h.svg", height: 62%)),
  align(horizon)[
    #fkdc elige $alpha = 1$ en _todas_ las semillas de `lunas_lo` (y en 24 de 25 en `espirales_lo`).

    #pause
    #v(0.4em)
    Su $h$ es apenas menor que el de #kdc, y cuanto mayor la diferencia de $h$, mayor la de $R^2$.

    #pause
    #v(0.4em)
    La ventaja no viene de $alpha$, sino de la regla de parsimonia sobre una grilla ampliada.
  ],
)

#intuicion[Desarmamos nuestro propio resultado favorable: #fkdc gana, pero no por Fermat, sino por elegir una campana un poco más angosta.]

== Donde Fermat sí aporta: `helices_0`

#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center + horizon, image("img/helices_0-kdc-fkdc-r2-scatter.svg", height: 52%)),
  uncover("2-", align(center + horizon, image("img/helices_0-1188-fkdc-bandwidth-alpha-loss_contour.svg", height: 52%))),
)

#chico[#fkdc elige $alpha > 1$ en 13 de las 25 semillas y supera a #kdc en 19. #uncover("2-")[Der.: con $s = 1188$, una «isla» en $alpha approx 3$, $h approx 10^(-3)$; todo lo que está a menos de un desvío tiene $alpha >= 2.5$, y le saca $0.23$ de $R^2$ a #kdc.]]

// TODO(autor): con un KDC exacto (el KernelDensity de scikit-learn falla en las colas), la ventaja de 5 pp deja de ser significativa; ver preguntas-defensa.typ. Las comparaciones de fkn contra kn no cambian.
#intuicion[Cuando la curva de la variedad es comparable a la distancia entre observaciones, aparece un $alpha$ que ningún ancho de campana sustituye.]

== Donde Fermat sí aporta: #fkn en `helices_0`

#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center + horizon, image("img/r2-fkn-kn-helices_0.svg", height: 48%)),
  uncover("2-", align(center + horizon, image("img/helices_0-fkn_kn-mean_test_score.svg", width: 100%))),
)

#chico[Izq.: $R^2$ por semilla; #fkn supera a #kn en 19 de las 25. #uncover("2-")[Der.: mejor log-verosimilitud en validación cruzada para cada $k$; la de #fkn casi no cae, la de #kn se desploma.]]

#intuicion[Como en los paseos: el vecindario de Fermat sigue la hélice, y agrandarlo no trae vecinos de la otra clase. Con $k = 1$ coinciden; Fermat aporta cuando el vecindario crece.]

== Donde Fermat sí aporta: `hueveras_0`

#grid(columns: (40%, 60%), column-gutter: 1em,
  align(center + horizon, image("img/hueveras-scatter-3d.svg", height: 55%)),
  align(horizon)[
    $z = plus.minus sin(x) sin(y)$: dos maples de huevos encastrados. Solo la familia de los núcleos logra $R^2 > 0$ ($0.25$--$0.30$).

    #pause
    #v(0.4em)
    En las 12 repeticiones en que #fkn y #kn eligen el _mismo_ $k$:
    - con $alpha = 1$, empatan en las 4;
    #pause
    - con $alpha > 1$, #fkn gana en 7 de 8, hasta por $0.096$.
  ],
)

#intuicion[Mismo vecindario, misma cantidad de vecinos: lo único que cambia es la regla con que se mide, y eso alcanza.]

== Lo feo: sin escalar, la cercanía no anda

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/pinguinos-crudo-vs-std-r2.svg"),
  align(horizon)[
    `pinguinos`: picos y aletas en mm, de 13 a 230; masa en g, de 2700 a 6300. Casi linealmente separable: #logr llega a $R^2 approx 0.96$; la familia de los núcleos, a $0.40$--$0.50$.

    #pause
    #v(0.5em)
    La distancia euclídea es, en la práctica, la diferencia de masa, y Fermat la hereda. Estandarizando, #fkdc pasa de $0.42$ a $0.96$.
  ],
)

#intuicion[Comparar pingüinos sumando milímetros y gramos es como comparar personas por su altura en kilómetros y su peso en gramos: solo cuenta el peso.]

== Lo feo: dimensiones de ruido

#grid(columns: (55%, 45%), column-gutter: 1em,
  grid(columns: 2, gutter: 2pt,
    ..("pionono", "eslabones", "helices", "hueveras").map(f => image("img/" + f + "-caida_r2-15d.svg")),
  ),
  align(horizon)[
    $ norm(x - y)^2 = sum_(k=1)^D (x_k - y_k)^2 $
    suma _todas_ las coordenadas: con 12 dimensiones de ruido, el ruido domina las aristas del grafo, y Fermat lo hereda.

    #pause
    #v(0.4em)
    #chico[Lleno, en 3D; vacío, en 15D. En `eslabones_12`, #fkdc cae a $R^2 approx 0.24$; #gbt queda en $0.92$.]
  ],
)

#intuicion[Si casi todas las coordenadas son ruido, toda distancia, euclídea o de Fermat, mide sobre todo ruido. #gnb y #gbt pueden ignorar las coordenadas que no informan; #kdc y #fkdc, no.]

== Lo feo: reducir antes

#grid(columns: (50%, 50%), column-gutter: 1em,
  align(horizon)[
    En `mnist`, PCA primero: de $784$ a $96$ dimensiones, con el 90 % de la variación.

    #pause
    #v(0.5em)
    Después, #fkdc llega a $R^2 approx 0.74$: a la par de #logr ($0.76$), por debajo de #kdc ($0.77$) y por encima de #gbt ($0.67$).
  ],
  align(center + horizon, image("img/mnist-r2-boxplot.svg", height: 62%)),
)

#intuicion[Hay que limpiar antes: quedarse con las direcciones en que los datos más varían, y recién ahí medir.]

= El futuro

== Después de la entrega

#let antes-despues(antes, despues, rotulo) = align(center)[
  #text(size: 22pt, fill: luma(140), antes) #h(0.2em) #text(size: 22pt, fill: luma(140))[→] \
  #text(size: 40pt, weight: "bold", despues) \
  #text(size: 16pt, rotulo)
]
#v(1fr)
#grid(columns: (1fr,) * 4, column-gutter: 1em,
  antes-despues("25", "100", "semillas, prerregistradas"),
  antes-despues("8", "22", "clasificadores"),
  antes-despues("24", "33", "datasets"),
  antes-despues("4800", "72 600", "tareas, 0 fallas"),
)
#v(1fr)
#align(center)[Una réplica más grande, y la comparación con el estado del arte.]
#chico[Los datasets cuentan las variantes estandarizadas.]

== Parte de la ventaja era calibración

#grid(columns: (58%, 42%), column-gutter: 1em,
  align(center, image("img/defensa/fkdc-kdc-crudo-templado.svg", height: 68%)),
  align(horizon)[
    Con el $R^2$ _templado_, que recalibra a cada clasificador por temperatura, de las ventajas de #fkdc sobre #kdc que reporta la tesis sobreviven las de `lunas_lo` y `espirales_lo`; las de `anteojos` y `helices_0`, no.
  ],
)

#intuicion[#kdc acierta, pero duda; #fkdc se juega más. Parte de la ventaja era esa: confianza, no puntería.]

== TabPFN: el estado del arte

#grid(columns: (58%, 42%), column-gutter: 1em,
  align(center, image("img/defensa/mapa-tabpfn.svg", height: 68%)),
  align(horizon)[
    Una red preentrenada con millones de tablas simuladas: queda entre los mejores en 19 de 33 datasets (en la figura, los 20 de la tesis).

    #v(0.5em)
    Los métodos de la tesis aprenden solo de la muestra: unos cientos de puntos.
  ],
)
// Sin fuga ni contaminación: ver «¿El resultado de TabPFN es legítimo?» en preguntas-defensa.typ
// Para el autor: el superlearner clásico apila con pesos convexos, sobre probabilidades fuera del pliego, 14 clasificadores rápidos (fkdc.clasicos.biblioteca): LR, LDA, QDA, GNB, kNN con k = 5, 15 y 45, tres KDC (isótropo estandarizado, isótropo crudo como el de la tesis, y ARD), GPC, GBT y extra trees. No incluye a fkdc ni a TabPFN.
// Para el autor: en digitos y mnist, fkdc también supera al KDC con un ancho por variable (0.977 contra 0.896, y 0.748 contra 0.664; medianas de R² crudo con 100 semillas, data/defensa/mapa-tabpfn.csv).

#intuicion[Una red entrenada de antemano con millones de problemas inventados resuelve uno nuevo de un vistazo, sin ajustar nada. La escala estira la zona cercana a $R^2 = 1$, donde se pelea el podio.]

== Una pregunta abierta

#align(center + horizon)[
  Lenguaje, imágenes, audio, video: datos de dimensión altísima que, según la hipótesis de la variedad, viven cerca de variedades que no conocemos.

  #pause
  #v(0.5em)
  Las redes profundas aprenden _representaciones_ de esos dominios, y el aprendizaje de variedades es un caso del aprendizaje de representaciones.
  #fuente[Bengio, Courville y Vincent (2013)]

  #pause
  #v(0.8em)
  *¿Pueden herramientas como estas, distancias aprendidas de la muestra y geometría de Riemann, ayudar a entenderlas?*
]

#focus-slide[¡Gracias!]

#show: appendix

= Apéndice

== Clasificadores nuevos

- *Clásicos*: QDA, LDA, mezclas gaussianas, procesos gaussianos, _extra trees_, #kdc de Abramson, Isomap y NCA con #kn, _label spreading_.
- *Redes para datos tabulares*: TabPFN-3.5 y v2, TabICL y TabM.
- *Modelos de lenguaje* que leen cada fila como texto: Laya, GLiNER2.5-Decide y una réplica abierta de Jev. Quedaron al nivel del azar.

#intuicion[Conjeturo que fallan porque deciden en una sola pasada: saber de qué círculo viene $(x_1, x_2)$ exige calcular $sqrt(x_1^2 + x_2^2)$, y eso no se hace «de un vistazo».]

== Datasets nuevos

#let nuevo(nombre, rotulo) = align(center)[
  #image("img/defensa/nuevo-" + nombre + ".svg", height: 42%)
  #raw(nombre) \
  #chico[#rotulo]
]
#grid(columns: 4, column-gutter: 0.8em,
  nuevo("nudos_800", [dos nudos tóricos enlazados]),
  nuevo("paralelas_4", [dos círculos paralelos]),
  nuevo("esfera_2", [dos casquetes en $S^2$]),
  nuevo("lissajous", [dos curvas en $RR^5$]),
)

#intuicion[Variedades conocidas, con su geometría en la mano: sabemos de antemano dónde viven los datos, y podemos medir cuánto de eso aprende cada clasificador.]
