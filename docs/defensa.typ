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

// Línea de tiempo al tope de la diapo: los hitos en una recta, el actual resaltado
#let cronologia(hitos, actual) = block(width: 100%, below: 1em, {
  let n = hitos.len()
  place(top + left, dx: 50% / n, dy: 4.25pt, line(length: 100% - 100% / n, stroke: 1.5pt + luma(200)))
  grid(columns: (1fr,) * n, ..hitos.enumerate().map(((i, (fecha, nombre))) => {
    let color = if i == actual { rgb("#eb811b") } else { luma(160) }
    align(center, stack(spacing: 5pt,
      circle(radius: 5pt, fill: color, stroke: none),
      text(size: 15pt, fill: color, weight: if i == actual { "bold" } else { "light" }, fecha),
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

#let cajita(d, fraccion, rotulo) = align(center)[
  #image("img/defensa/cajita-d" + str(d) + ".svg", height: 45%)
  $d = #d$ \
  #chico[#fraccion del #rotulo]
]
#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1em,
  cajita(1, $h$, "largo"),
  cajita(2, $h^2$, "área"),
  cajita(3, $h^3$, "volumen"),
)

#align(center)[Con $X tilde.op "Unif"([-1, 1]^d)$, la fracción esperada dentro de $[-h, h]^d$ es $h^d$.]

#intuicion[Con $h = 1 slash 2$: la mitad del largo, un cuarto del área, un octavo del volumen. Cada dimensión nueva vuelve a partir la cajita.]

== En alta dimensión, todo queda lejos

#grid(columns: (60%, 40%), column-gutter: 1em,
  image("img/curse-dim.svg"),
  align(horizon)[
    $h^d -> 0$ cuando $d -> oo$, aun con $h$ cerca de $1$:

    - $0.9^15 approx 0.2$
    - $0.9^130 approx 10^(-6)$

    #v(0.5em)
    Las distancias se _concentran_: en `mnist`, la mediana al vecino más cercano es $1300$, y entre dos puntos cualesquiera, $2500$.
  ],
)
// TODO(autor): «lo mismo le pasa a otras distancias (p. ej., coseno)» no está en la tesis; agregar solo si se puede defender

#intuicion[Una cajita del 90 % del ancho guarda, en 15 dimensiones, un quinto del volumen; en 130, una millonésima. Los vecinos dejan de ser vecinos: es la _maldición de la dimensionalidad_.]

== Y sin embargo, funciona

#grid(columns: (50%, 50%), column-gutter: 1em,
  image("img/digitos-mnist-ejemplos.png"),
  align(horizon)[
    - `digitos`: $8 times 8 = 64$ dimensiones, en el borde de lo legible.
    - `mnist`: $28 times 28 = 784$, ¡para un solo dígito!
    - Un segundo de audio a $44.1$ kHz: $44 thin 100$.
    - Una foto de un teléfono, $12$ megapíxeles en color: $36$ millones.

    #v(0.5em)
    *¿Cómo?*
  ],
)

#intuicion[Clasificamos imágenes, audio, video... y ahora también texto. Si en alta dimensión todo queda lejos, algo tiene que estar a nuestro favor.]

== La hipótesis de la variedad

#grid(columns: (45%, 55%), column-gutter: 1em,
  box(radius: 0.5em, clip: true, image("img/bandera-argentina.png")),
  align(horizon)[
    Los datos se registran en $RR^D$, pero viven cerca de una _variedad_ de dimensión mucho menor:
    $ X : (Omega, cal(A), P) -> cal(M) subset RR^D, quad dim cal(M) = d << D. $

    Una bandera al viento es una variedad de dimensión intrínseca $d = 2$ en el espacio físico, con $D = 3$.
  ],
)

#intuicion[Una _variedad_ es un espacio potencialmente de lo más rebuscado, pero que localmente se comporta como uno euclídeo: de cerca, la bandera es una hoja plana.]

#let hitos-teoria = (
  ([1918], [von Mises: $S^1$]),
  ([1953], [Fisher: $S^2$]),
  ([1975], [Mardia: $S^n$]),
  ([2005], [Pelletier: $(cal(M), g)$]),
)

== Y hay teoría

#cronologia(hitos-teoria, 0)
#grid(columns: (40%, 60%), column-gutter: 1em,
  align(center, image("img/von-mises-s1.png", height: 55%)),
  align(horizon)[
    Von Mises se pregunta si los pesos atómicos son enteros con error de medición.

    #v(0.5em)
    Mira la parte fraccionaria: un dato en el círculo $S^1$, donde el error gaussiano no sirve.
  ],
)

#intuicion[$S^1$ es el círculo: un ángulo, una hora del reloj, la dirección del viento. Ahí, $0.99$ y $0.01$ están cerca.]

== Y hay teoría

#cronologia(hitos-teoria, 1)
#grid(columns: (45%, 55%), column-gutter: 1em,
  align(center)[
    #image("img/henry-rodriguez-bolas.png", height: 50%)
    #fuente[Sitios volcánicos en la Tierra; Henry y Rodríguez (2009)]
  ],
  align(horizon)[
    Fisher desarrolla una teoría para posiciones en la esfera $S^2$, con su test de significancia.

    #v(0.5em)
    Los datos: la dirección de magnetización de coladas de lava en Islandia.
  ],
)

#intuicion[$S^2$ es la superficie de una pelota, o de la Tierra: una posición se da con dos ángulos, latitud y longitud.]

== Y hay teoría

#cronologia(hitos-teoria, 2)
#align(center + horizon)[
  Los casos de von Mises y Fisher se integran en la esfera $S^n$: la _estadística direccional_.

  #v(0.5em)
  Teoría de la distribución von Mises--Fisher, familias exponenciales, teorema central del límite.
  #fuente[Mardia (1975); Jupp y Mardia (1989)]
]

#intuicion[$S^n$ contiene todas las _direcciones_ posibles de un vector en $RR^(n+1)$.]

== Pelletier: densidad por núcleos en variedades

#cronologia(hitos-teoria, 3)
#grid(columns: (1fr, 1fr), column-gutter: 1.5em,
  [
    *En $RR^d$*
    $ hat(f)(x) = 1 / N sum_(i=1)^N 1 / h^d K((norm(x - X_i)) / h) $
  ],
  [
    *En $(cal(M), g)$, compacta y sin frontera*
    $ hat(f)(p) = 1 / N sum_(i=1)^N 1 / h^d 1 / (theta_(X_i)(p)) K((d_g (p, X_i)) / h) $
  ],
)

#align(center)[La norma pasa a ser la distancia geodésica $d_g$. Lo nuevo es la _densidad de volumen_ $theta$.]

#intuicion[La misma receta: una campana sobre cada dato, y se suman. Cambia la regla con que medimos, que ahora sigue la variedad, y una corrección por cuánto se estira el espacio.]

#focus-slide[¿Y si no conocemos la variedad? \ ¡Hay que aprenderla!]

#let hitos-aprendizaje = (
  ([1901], [PCA]),
  ([2000], [Isomap]),
  ([2011--2022], [Fermat]),
)

== Aprender la variedad

#cronologia(hitos-aprendizaje, 0)
#grid(columns: (45%, 55%), column-gutter: 1em,
  align(center, image("img/pca.png", height: 55%)),
  align(horizon)[
    Las direcciones de mayor variabilidad, $bu(U)_m$, y la proyección $hat(XX) = XX bu(U)_m$.

    #v(0.5em)
    Aprende variedades _lineales_. #fuente[Pearson (1901)]
  ],
)

#intuicion[La mejor sombra plana de la nube de puntos.]

== Aprender la variedad

#cronologia(hitos-aprendizaje, 1)
#grid(columns: (45%, 55%), column-gutter: 1em,
  align(center, image("img/isomap-2.png", height: 55%)),
  align(horizon)[
    El grafo de $k$ vecinos aproxima la variedad; la distancia geodésica, por caminos mínimos en el grafo.

    #v(0.5em)
    #fuente[Tenenbaum, de Silva y Langford (2000)]
  ],
)

#intuicion[Para ir de un punto a otro, solo se salta entre vecinos cercanos: el camino sigue la forma de los datos.]

== Aprender la variedad: la distancia de Fermat

#cronologia(hitos-aprendizaje, 2)
#grid(columns: (40%, 60%), column-gutter: 1em,
  align(center, image("img/distancia-cuadrada.svg", height: 50%)),
  align(horizon)[
    $ D_(Q, alpha)(x, y) = inf sum_j norm(q_(j+1) - q_j)^alpha, quad q_j in Q $

    Con $alpha > 1$, $alpha$ se elige por validación cruzada, sin conocer $d$.
    #fuente[Bijral, Ratliff y Srebro (2011); Groisman, Jonckheere y Sapienza (2022)]
  ],
)

#intuicion[Elevar las distancias a una potencia abarata los saltos cortos: el camino más barato pasa por donde hay muchos datos.]

== Paseos al azar

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1em,
  align(center)[#image("img/defensa/paseos-espirales-euclidea.svg", height: 62%) Euclídea],
  align(center)[#image("img/defensa/paseos-espirales-fermat.svg", height: 62%) Fermat, $alpha = 4$],
  align(horizon)[
    100 paseos de 15 saltos desde #text(fill: rgb("#d4a017"))[★], sobre el grafo completo:
    $ Pr(x_i -> x_j) prop 1 / d(x_i, x_j) $

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

#align(center)[
  Si $cal(M)$ es una variedad isométrica embebida y $Q_n tilde.op^"iid" f$,
  $ n^beta D_(Q_n, alpha)(x, y) -> mu thin cal(D)_(f, beta)(x, y) "c.s.", quad beta = (alpha - 1) / d. $
  #fuente[Groisman, Jonckheere y Sapienza (2022), Teorema 2.7]
]

#intuicion[Hay una idea más general y poderosa detrás: un camino cuesta más donde hay pocos datos. Sumar las distancias infladas a lo largo del grafo alcanza para aproximarla, y con más datos, la aproximación mejora.]

#focus-slide[¿Servirá para clasificar?]

= El plan

#let hitos-plan = (
  ([2005], [Pelletier: KDE]),
  ([2008], [Loubes y Pelletier: KDC]),
  ([2009], [Henry y Rodríguez]),
  ([2026], [Esta tesis]),
)

== Del estimador al clasificador

#cronologia(hitos-plan, 0)
#align(center + horizon)[
  Densidad por núcleos en variedades de Riemann compactas sin frontera, con $h < "iny" cal(M)$:
  $ EE norm(hat(f)_n - f)_(L^2(cal(M)))^2 = O(n^(-4 slash (d + 4))) $
]

#intuicion[La receta de siempre, midiendo sobre la variedad: converge, y a la velocidad que dicta la dimensión _intrínseca_ $d$, no la del espacio ambiente.]

== Del estimador al clasificador

#cronologia(hitos-plan, 1)
#align(center + horizon)[
  Clasificador por la regla de Bayes:
  $ hat(G)(p) = arg max_(k) sum_(i=1)^N bb(1)(G_i = k) K_h (p, X_i) $
  Fuertemente consistente con dos clases.
]

#intuicion[Cada clase vota con la suma de sus campanas en el punto $p$, y gana la que más pesa.]

== Del estimador al clasificador

#cronologia(hitos-plan, 2)
#align(center + horizon)[
  Consistencia fuerte uniforme:
  $ sup_(p in cal(M)) abs(hat(f)_n (p) - f(p)) -> 0 "c.s." $
  Y en $S^2$, la densidad de volumen $theta_p (q)$ explícita.
]

#intuicion[El estimador no solo se acerca a la densidad verdadera en promedio: se acerca en todos los puntos a la vez.]

== Del estimador al clasificador

#cronologia(hitos-plan, 3)
#align(center + horizon)[
  Clasificador _multiclase_ sobre una variedad _desconocida_:

  #v(0.5em)
  la distancia de Fermat _muestral_ reemplaza a la geodésica, y $theta equiv 1$.
]

#intuicion[No conocemos la variedad, así que no podemos medir sobre ella: medimos con la distancia que aprendemos de los datos.]

== El clasificador #fkdc

Para cada clase $i$, con $Q_i = {x} union {x_j : g_j = i}$ y $N_i = abs(Q_i) - 1$:
$ hat(f)_i (x) = 1 / N_i sum_(j : g_j = i) 1 / h^D K((D_(Q_i, alpha)(x, x_j)) / h), quad K(u) = (2 pi)^(-D slash 2) e^(-u^2 slash 2) $
$ hat(Pr)(G = i | X = x) = (hat(pi)_i hat(f)_i (x)) / (sum_k hat(pi)_k hat(f)_k (x)) $

#chico[$(h, alpha)$ por validación cruzada con la regla de un desvío. Con $alpha = 1$, es el #kdc euclídeo de siempre.]

#v(0.3em)
#text(size: 15pt)[*#fkn*: los $k$ más cercanos según la misma $D_(Q_i, alpha)$ votan por su clase.]

#intuicion[Para clasificar un punto nuevo, lo sumamos a cada clase y medimos con Fermat cuánto lo «abriga» cada una. Gana la que más lo abriga.]

== El código

#text(size: 19pt)[
- `SampleFermatDistance(Q, alpha=1, grupos=None)`: distancias de Fermat muestrales a través del grafo completo de `Q`, con un algoritmo incremental para puntos fuera de la muestra.
  - `FermatKNeighborsClassifier(n_neighbors=5, alpha=1)` reutiliza `KNeighborsClassifier(metric="precomputed")` de `scikit-learn`.
- `KDClassifier(bandwidth=1.0, metric="euclidean", alpha=1.0)`: densidad por núcleos con distancia configurable.
  - #fkdc es `KDClassifier(metric="fermat")`.
- Un marco de evaluación con `Pipeline` y `GridSearchCV` de `scikit-learn`, que compara sistemáticamente los clasificadores en repetidas `Tarea`s sobre datasets definidos de antemano.
]

#intuicion[Todo se enchufa a `scikit-learn`, la caja de herramientas estándar de aprendizaje automático en Python: los clasificadores nuevos y los de siempre se evalúan exactamente con el mismo procedimiento.]

== ¿Una biblioteca?

#pendiente(alto: 70%)[publicar en PyPI y poner el enlace acá]

== 8 clasificadores

#grid(columns: (1fr, 1fr), column-gutter: 2em,
  [
    *La familia de los núcleos*
    - #fkdc y #kdc: densidad por núcleos
    - #fkn y #kn: $k$ vecinos más cercanos
  ],
  [
    *De referencia*
    - #gnb: Naive Bayes gaussiano
    - #logr: regresión logística
    - #gbt: _gradient boosting_
    - #svc: soporte vectorial, solo exactitud
  ],
)

#intuicion[Cuatro métodos que deciden por cercanía, con y sin Fermat, contra cuatro que no: del más simple al más cercano al estado del arte.]

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

== 25 repeticiones

Cada tarea se repite con 25 semillas: en los sintéticos, 25 muestras nuevas; en los reales, 25 particiones de entrenamiento y evaluación.

#v(0.5em)
Comparamos la _mediana_ del $R^2$ de McFadden sobre las repeticiones:
$ R^2 = 1 - (cal(l)(hat(G))) / (cal(l)(hat(G)_0)), $
con $cal(l)$ la log-verosimilitud promedio y $hat(G)_0$ el clasificador que asigna las frecuencias de clase.

#intuicion[Buscamos resultados repetibles, así que cada experimento se corre 25 veces y miramos el resultado _del medio_, al que no lo mueve una semilla con suerte. $R^2 = 0$: no mejor que adivinar con las proporciones de cada clase; $R^2 = 1$: acierta siempre, y sin dudar.]

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

== Lo bueno: #fkdc en `espirales_lo`

#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center, image("img/espirales_lo-fkdc-decision_boundary.svg", height: 55%)),
  align(center, image("img/espirales_lo-r2-boxplot.svg", height: 55%)),
)

#align(center)[Pocos puntos, mucha curva: la mediana de $R^2$ pasa de $0.83$ con #kdc a $0.87$ con #fkdc.]

#intuicion[Cuando las clases se enroscan una en la otra y hay pocos datos, medir a lo largo de la espiral ayuda.]

== Lo bueno: todas las fronteras en `espirales_lo`

#grid(columns: 4, column-gutter: 0.4em, row-gutter: 0.3em,
  ..("kn", "fkn", "kdc", "fkdc", "gnb", "lr", "svc", "gbt").map(c => image("img/espirales_lo-" + c + "-decision_boundary.svg")),
)

== Lo bueno: #fkn en `helices_0`

#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center + horizon, image("img/r2-fkn-kn-helices_0.svg", height: 48%)),
  align(center + horizon, image("img/helices_0-fkn_kn-mean_test_score.svg", width: 100%)),
)

#chico[Izq.: $R^2$ por semilla; #fkn supera a #kn en 19 de las 25. Der.: mejor log-verosimilitud en validación cruzada para cada $k$; la de #fkn casi no cae, la de #kn se desploma.]

#intuicion[Como en los paseos: el vecindario de Fermat sigue la hélice, y agrandarlo no trae vecinos de la otra clase.]

== Lo malo: bien muestreado, $alpha$ y $h$ se confunden

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/circulos_lo-8527-fkdc-bandwidth-alpha-loss_contour.svg"),
  align(horizon)[
    Superficie de pérdida de #fkdc en `circulos_lo`: un «risco» de parametrizaciones equivalentes, $log h prop alpha$.

    #v(0.5em)
    La regla de parsimonia elige $alpha = 1$, y #fkdc se reduce a #kdc.
  ],
)

#intuicion[Donde los datos son densos, el camino más barato entre vecinos es el salto directo: Fermat no logra nada que no se logre ajustando el ancho de la campana.]

== Lo feo: `pinguinos`

#grid(columns: (45%, 55%), column-gutter: 1em,
  align(horizon)[
    Pingüinos de Palmer: $K = 3$ especies, $D = 4$ atributos.
    - largo y ancho del pico, largo de la aleta, en mm: de 13 a 230;
    - masa corporal, en g: de 2700 a 6300.

    #v(0.5em)
    Casi linealmente separable: #logr llega a $R^2 approx 0.96$; la familia de los núcleos, a $0.40$--$0.50$.
  ],
  image("img/pinguinos-r2-boxplot.svg"),
)

#intuicion[Un problema fácil, en el que los métodos por cercanía fracasan.]

== Lo feo: sin escalar, la cercanía no anda

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/pinguinos-crudo-vs-std-r2.svg"),
  align(horizon)[
    Mediana de $R^2$ por clasificador: punto lleno, atributos crudos; vacío, estandarizados.

    #v(0.5em)
    La distancia euclídea es, en la práctica, la diferencia de masa, y Fermat la hereda. Estandarizando, #fkdc pasa de $0.42$ a $0.96$.
  ],
)

#intuicion[Comparar pingüinos sumando milímetros y gramos es como comparar personas por su altura en kilómetros y su peso en gramos: solo cuenta el peso.]

== Lo feo: ruido en alta dimensión

#grid(columns: (55%, 45%), column-gutter: 1em,
  align(horizon)[
    $ norm(x - y)^2 = sum_(k=1)^D (x_k - y_k)^2 $
    suma _todas_ las coordenadas: con 12 dimensiones de ruido, el ruido domina las aristas del grafo, y Fermat lo hereda.

    #v(0.4em)
    En `eslabones_12`, #fkdc cae a $R^2 approx 0.24$; #gbt, que mira cada coordenada por separado, queda en $0.92$.
  ],
  align(center + horizon, image("img/eslabones_12-r2-boxplot.svg", height: 62%)),
)

#intuicion[Si casi todas las coordenadas son ruido, toda distancia, euclídea o de Fermat, mide sobre todo ruido. #gnb y #gbt pueden ignorar las coordenadas que no informan; #kdc y #fkdc, no.]

== Lo feo: reducir antes

#grid(columns: (50%, 50%), column-gutter: 1em,
  align(horizon)[
    En `mnist`, PCA primero: de $784$ a $96$ dimensiones, con el 90 % de la variación.

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

== Clasificadores nuevos

- *Clásicos*: QDA, LDA, mezclas gaussianas, procesos gaussianos, _extra trees_, #kdc de Abramson, Isomap y NCA con #kn, _label spreading_.
- *Redes para datos tabulares*: TabPFN-3.5 y v2, TabICL y TabM.
- *Modelos de lenguaje* que leen cada fila como texto: Laya, GLiNER2.5-Decide y una réplica abierta de Jev. Quedaron al nivel del azar.

#intuicion[Los modelos de lenguaje deciden en una sola pasada: saber de qué círculo viene $(x_1, x_2)$ exige calcular $sqrt(x_1^2 + x_2^2)$, y eso no lo hacen «de un vistazo».]

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
    Una red preentrenada con millones de tablas simuladas: queda entre los mejores en 19 de 33 datasets.

    #v(0.5em)
    Los métodos de la tesis aprenden de 400 puntos.
  ],
)
// Sin fuga ni contaminación: ver «¿El resultado de TabPFN es legítimo?» en preguntas-defensa.typ
// Para el autor: el superlearner clásico apila con pesos convexos, sobre probabilidades fuera del pliego, 14 clasificadores rápidos (fkdc.clasicos.biblioteca): LR, LDA, QDA, GNB, kNN con k = 5, 15 y 45, tres KDC (isótropo estandarizado, isótropo crudo como el de la tesis, y ARD), GPC, GBT y extra trees. No incluye a fkdc ni a TabPFN.
// Para el autor: en digitos y mnist, fkdc también supera al KDC con un ancho por variable (0.977 contra 0.896, y 0.748 contra 0.664; medianas de R² crudo con 100 semillas, data/defensa/mapa-tabpfn.csv).

#intuicion[Una red entrenada de antemano con millones de problemas inventados resuelve uno nuevo de un vistazo, sin ajustar nada. La escala estira la zona cercana a $R^2 = 1$, donde se pelea el podio.]

== El #kdc es una cabeza de atención

#align(center + horizon)[
  $ hat(Pr)(dot | x) = sum_i "softmax"_i (- norm(x - x_i)^2 / (2 h^2)) e_(g_i) $
  frente a
  $ sum_i "softmax"_i (q^top k_i slash sqrt(d_k)) v_i $

  Con claves de igual norma, son la misma cuenta. #fkdc fija el núcleo con una regla; TabPFN lo _aprende_.
]

#intuicion[Una cabeza de atención pondera cada ejemplo por su parecido con el punto a clasificar y promedia sus etiquetas: lo mismo que un #kdc, pero aprendiendo cómo medir el parecido.]

== ¿Y la IA?

#align(center + horizon)[
  El aprendizaje de variedades es un caso del _aprendizaje de representaciones_, y los _transformers_ detrás de los modelos de lenguaje aprenden representaciones de dominios enormes.
  #fuente[Bengio, Courville y Vincent (2013)]

  #v(0.8em)
  *¿Puede la geometría de Riemann ayudarnos a entenderlas mejor?*
]

#focus-slide[¡Gracias!]
