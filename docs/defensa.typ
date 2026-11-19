#import "@preview/touying:0.6.2": *

#import themes.metropolis: *

// Con `--input guion=si`, cada diapo lleva al costado el texto para decir (las notas `#speaker-note`)
#let con-guion = sys.inputs.at("guion", default: "no") == "si"
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: [Defensa de tesis · Maestría en Estadística Matemática · FCEN-UBA],
  config-common(show-notes-on-second-screen: if con-guion { right } else { none }),
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
  extra: text(size: 15pt, fill: rgb("#eb811b"))[*395* días sin aplicarse la Ley de Financiamiento Universitario (Ley 27.795)],
)

= Enunciación

== En alta dimensión, todo queda lejos

// guion
#speaker-note[
  #set text(size: 17pt)
  [En la portada:] Bienvenidos a esta defensa de tesis. Antes de empezar: este número es la cantidad de días que lleva sin aplicarse la Ley de Financiamiento Universitario. Que se cumpla. Pasado ese parroquial, ahora sí, arranquemos.

  [Clic a esta diapo.] Hay un problema en la clasificación en general: en alta dimensión, y casi todos los objetos de interés están en alta dimensión, todo queda muy lejos.

  Imagínense que de un segmento tomo una fracción $h$ del largo: me quedo con $h$. De un cuadrado, con $h^2$ del área; de un cubo, con $h^3$ del volumen. Con una cajita del 90 % del ancho, en 15 dimensiones me quedo con un quinto del volumen; en 130, con una millonésima.

  ▸ Y eso se traduce en que las distancias se concentran: desde una observación cualquiera, el vecino más cercano y el más lejano quedan casi a la misma distancia. En `mnist`, ya reducido a 96 dimensiones, la distancia mediana al vecino más cercano ronda 1300, y entre dos puntos cualesquiera, 2500. Y casi todos los algoritmos de clasificación dependen de esa noción: lo parecido está cerca, lo distinto, lejos.
]


#grid(columns: (55%, 45%), column-gutter: 1em,
  align(horizon, image("img/curse-dim.svg", height: 70%)),
  align(horizon)[
    Con $X tilde.op "Unif"([-1, 1]^D)$, la fracción dentro de $[-h, h]^D$ es $h^D -> 0$: $0.9^15 approx 0.2$, $0.9^130 approx 10^(-6)$.

    #pause
    #v(0.3em)
    #chico[Las distancias se _concentran_: en `mnist`, ya reducido a 96 dimensiones, la distancia mediana al vecino más cercano ronda $1300$, y entre dos puntos cualesquiera, $2500$.]
  ],
)

#intuicion[Una cajita del 90 % del ancho guarda, en 15 dimensiones, un quinto del volumen; en 130, una millonésima. Los vecinos dejan de ser vecinos: es la _maldición de la dimensionalidad_.]

== Y sin embargo, funciona: la hipótesis de la variedad

// guion
#speaker-note[
  #set text(size: 17pt)
  Y sin embargo, funciona. Un dígito de `mnist` tiene 784 dimensiones; un segundo de voz telefónica, 8000; una foto de teléfono, 36 millones. Y se clasifican. ¿Cómo?

  ▸ Hay una hipótesis bastante informal, pero que está siempre presente en estos problemas: la hipótesis de la variedad. El fenómeno tiene una dimensión intrínseca, unas pocas direcciones en las que tiene sentido, que no observamos directamente: lo registramos en un soporte de dimensión muchísimo más alta, la cámara, el grabador, una secuencia de ADN. Los datos viven cerca de una superficie de menor dimensión, embebida en el espacio en que los registramos.

  El ejemplo más pintoresco es nuestra bandera flameando al viento: una superficie de dimensión 2 que, en el mundo, vive en dimensión 3. Si uno pone una hormiguita en la bandera, de cerca es un planito; a gran escala hay pliegues y curvas.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  ¿Se puede hacer estadística en variedades? Sí, y con bastante rigor. La línea de tiempo arranca en 1918: von Mises se pregunta si los pesos atómicos son enteros con error de medición, y mira la parte fraccionaria, un dato en el círculo. Sigue Fisher en la esfera, y la estadística direccional de Mardia. Pero son variedades muy rígidas.

  ▸ En 2005, Bruno Pelletier da el estimador de densidad por núcleos en variedades de Riemann compactas y sin frontera. A la izquierda, el de siempre; a la derecha, el de Pelletier. El mismo promedio de campanas, el mismo $1/h^d$. Cambian dos cosas: la norma pasa a ser la distancia geodésica, y aparece la densidad de volumen $theta$, bastante compleja de calcular; por ahora le ponemos un pin.

  ▸ Y converge a la velocidad que dicta la dimensión intrínseca $d$, no la del espacio ambiente.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  En 2008, Pelletier y Jean-Michel Loubes hacen _plug-in_ de ese estimador en la regla de Bayes: la clase predicha es la que maximiza la suma de sus campanas en el punto. Mientras más puntos tenga una clase, y más cerca estén, más pesa. Es fuertemente consistente con dos clases.

  ▸ En 2009, acá en casa, Guillermo Henry y Daniela Rodríguez prueban la consistencia fuerte uniforme y calculan explícitamente la densidad de volumen en la esfera, donde es tratable porque la esfera se ve igual desde cualquier punto. [Opcional, mirando a Dani: «¿fue fácil?».] En variedades más plegadas, no.

  [Clic a la diapo de foco.] Una cosa es decir que los dígitos o el canto de un ruiseñor viven en una variedad; otra, saber cuál es. Hay que aprenderla de los datos.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  La forma canónica es PCA: el subespacio que pasa más cerca de los puntos. Pero aprende solo variedades lineales, y la bandera no es lineal.

  ▸ Isomap toma el grafo de vecinos más cercanos y mide la distancia como el camino mínimo en ese grafo: saltando de vecino en vecino, el camino sigue la forma de los datos. El problema es elegir el vecindario: si es grande, la variedad se cortocircuita. Piensen en las puntas de un paraguas cerrado: parecen muy cerca, pero un ácaro que camine por la tela tiene que dar toda la vuelta.

  ▸ La distancia de Fermat, en la que trabajaron Pablo Groisman, Matthieu Jonckheere y Facundo Sapienza, entre otros, no elige vecindario: usa el grafo completo, pero eleva cada arista a una potencia $alpha$. Miren el triángulo: con $alpha = 1$, de $a$ a $c$ conviene ir directo, que cuesta 3, y no por $b$, que suma 4. Con $alpha = 2$, el lado directo cuesta 9 y el desvío, 8: dos saltos cortitos le ganan a uno largo. El camino más barato pasa por donde hay datos.
]


#only(1, cronologia(hitos-aprendizaje, 0))#only(2, cronologia(hitos-aprendizaje, 1))#only(3, cronologia(hitos-aprendizaje, 2))
#grid(columns: (1fr, 1fr, 1fr), column-gutter: 1em,
  align(center)[
    #image("img/pca.png", height: 38%)
    #chico[Variedades _lineales_: las direcciones de mayor variabilidad.]
  ],
  uncover("2-", align(center)[
    #image("img/isomap-2.png", height: 38%)
    #chico[Geodésicas por caminos mínimos en el grafo de $k$ vecinos.]
  ]),
  uncover("3-", align(center)[
    #image("img/distancia-cuadrada.svg", height: 38%)
    #chico[Caminos en el grafo completo, con aristas elevadas a $alpha >= 1$.]
  ]),
)
#fuente[Pearson (1901); Tenenbaum, de Silva y Langford (2000); Bijral, Ratliff y Srebro (2011); Groisman, Jonckheere y Sapienza (2022)]

#uncover("3-", intuicion[Elevar las distancias a una potencia abarata los saltos cortos: el camino más barato pasa por donde hay muchos datos.])

== Paseos al azar

// guion
#speaker-note[
  #set text(size: 17pt)
  Para verlo: desde la estrellita, 100 paseos al azar en los que se puede saltar a cualquier punto, con probabilidad inversamente proporcional a la distancia.

  ▸ Con la euclídea, a la izquierda, van de todas partes a todas partes y cruzan de un brazo al otro todo el tiempo: 6,3 cambios de brazo por paseo. Con Fermat y $alpha = 4$, a la derecha, siguen el brazo en el que están: 1,1 cambios por paseo. Los colores no significan nada; fíjense que los saltos largos se cuentan con los dedos de una mano.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  Lo que calculamos es la versión microscópica: el costo mínimo de los caminos a través de la muestra. Tiene una versión macroscópica, la integral a lo largo de una curva de $f^(-beta)$: un camino cuesta más donde hay pocos datos.

  ▸ Y Groisman, Jonckheere y Sapienza prueban que, bien escalada, la muestral converge a la macroscópica, con $beta = (alpha - 1) / d$. Lo práctico: $alpha$ se elige por validación cruzada, sin conocer $d$.

  ▸ Un detalle que va a importar: la escala depende de $n$. Si cada clase aprende su propia distancia y $alpha > 1$, la clase más grande queda «más cerca» de todo. En nuestros datasets desbalanceados, `pinguinos` y `vino`, la validación cruzada eligió $alpha = 1$, así que no nos afectó; pero es una deuda.

  [Clic a la diapo de foco.] ¿Servirá para clasificar? Poder, podemos. La pregunta es si funciona.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  Proponemos el clasificador de Loubes y Pelletier, pero multiclase y sobre una variedad desconocida. Para clasificar un punto nuevo $x$, lo sumamos a la muestra de cada clase, calculamos la distancia de Fermat de esa clase y estimamos la densidad de la clase en $x$.

  ▸ Con la regla de Bayes, la probabilidad de cada clase es su proporción en la muestra por su densidad, normalizada: un clasificador suave, que da probabilidades y no solo una etiqueta.

  ▸ La distancia de Fermat muestral reemplaza a la geodésica, y asumimos $theta equiv 1$: esta asunción es por limitaciones de mi capacidad cognitiva, no porque sea correcta. Con $alpha = 1$, es el `KDC` euclídeo de siempre. Y `fkNN` es lo mismo con $k$ vecinos: los $k$ más cercanos según Fermat votan por su clase.
]


El de Loubes y Pelletier, _multiclase_ y sobre una variedad _desconocida_. Para cada clase $i$, con $Q_i = {x} union {x_j : g_j = i}$ y $N_i = abs(Q_i) - 1$:
$ hat(f)_i (x) = 1 / N_i sum_(j : g_j = i) 1 / h^D K((D_(Q_i, alpha)(x, x_j)) / h), quad K(u) = (2 pi)^(-D slash 2) e^(-u^2 slash 2) $
#pause
$ hat(Pr)(G = i | X = x) = (hat(pi)_i hat(f)_i (x)) / (sum_k hat(pi)_k hat(f)_k (x)) $

#pause
#chico[La distancia de Fermat muestral reemplaza a la geodésica, y $theta equiv 1$. $(h, alpha)$ por validación cruzada con la regla de un desvío; con $alpha = 1$, es el #kdc euclídeo de siempre. *#fkn*: los $k$ más cercanos según la misma $D_(Q_i, alpha)$ votan por su clase.]

#intuicion[Para clasificar un punto nuevo, lo sumamos a cada clase y medimos con Fermat cuánto lo «abriga» cada una. Gana la que más lo abriga.]

== El código

// guion
#speaker-note[
  #set text(size: 17pt)
  No pudimos usar el paquete `fermat` de Facundo Sapienza, porque está pensado para _clustering_ y solo mide distancias dentro de la muestra. Implementamos la distancia muestral, con un algoritmo para puntos nuevos, y los dos clasificadores, compatibles con `scikit-learn`: se evalúan exactamente con el mismo procedimiento que los de siempre, con búsqueda de hiperparámetros por validación cruzada, sin tocar números a mano para ver con qué semilla anda. Y está publicado: `pip install fkdc`.

  [Decir «en PyPI» solo si ya está publicado; si no, «en TestPyPI».]
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  No alcanzaba con ponerle y sacarle Fermat a nuestros dos clasificadores: queríamos saber si son competitivos. Agregamos cuatro de referencia: Naive Bayes gaussiano, pariente cercano del `KDC`; la regresión logística, que no podía faltar; _gradient boosting_, que requiere muy poca preparación de los datos; y soporte vectorial, que solo da etiquetas, así que lo medimos solo en exactitud.

  ▸ Cada tarea se repite con 25 semillas, con una partición 50–50 de entrenamiento y evaluación, y comparamos la mediana del $R^2$ de McFadden, una transformación de la log-verosimilitud. $R^2 = 0$ es no ser mejor que adivinar con las proporciones de cada clase; $R^2 = 1$, acertar siempre, y sin dudar. Castiga a quien se equivoca con confianza.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  Veinte datasets: sintéticos en dos dimensiones, con poco y mucho ruido; en tres, con y sin 12 dimensiones extra de ruido puro; y cinco reales, chicos: `iris`, `vino`, `pinguinos`, `digitos` y `mnist`. De dos, tres y diez clases, y de 2 a 784 dimensiones.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  En muy resumidas cuentas, a los dos métodos les fue razonablemente bien. Primero, un _mea culpa_: cuando digo «mejor» no es un test de hipótesis; A es peor que B si la mediana de su $R^2$ queda por debajo de la caja de B. El test como corresponde lo hice después de entregar.

  Arriba, `fKDC` contra su versión euclídea: gana en 6 datasets y pierde en 2. Abajo, contra el mejor de los demás: gana en 3 y pierde en 7.

  Ningún clasificador domina. Por $R^2$ mediano, la familia de los núcleos tiene al mejor en 14 de los 20 datasets; `fKDC` solo, en 7. _Boosting_ gana en 5, casi todos con mucho ruido. Por exactitud, el panorama cambia: los núcleos bajan a 9 de 20, y el soporte vectorial lidera en 7. Los núcleos dan, sobre todo, mejores probabilidades; no necesariamente mejores etiquetas.
]


#align(center, image("img/defensa/resumen-fkdc.svg", width: 100%))
#leyenda-resumen

#intuicion[Contra su versión euclídea, #fkdc gana en 6 datasets y pierde en 2. Contra el mejor de los demás, gana en 3 y pierde en 7.]

== En resumen: #fkn

// guion
#speaker-note[
  #set text(size: 17pt)
  `fkNN` es más modesto: contra $k$ vecinos euclídeo gana en 4 datasets y no pierde en ninguno; contra el mejor de los demás, nunca gana. Fermat no le hace daño, pero $k$ vecinos, por sí solo, a veces no alcanza.
]


#align(center, image("img/defensa/resumen-fkn.svg", width: 100%))
#leyenda-resumen

#intuicion[Contra su versión euclídea, #fkn gana en 4 datasets y no pierde en ninguno. Contra el mejor de los demás, nunca gana.]

== Las fronteras en `espirales_lo`

// guion
#speaker-note[
  #set text(size: 17pt)
  Así se ve en las espirales: a qué clase, y con qué seguridad, asigna cada clasificador cada punto del plano. Arriba, la familia de los núcleos: los cuatro siguen la espiral. $k$ vecinos, con o sin Fermat, dibuja bordes muy rugosos; `KDC` deja islas de densidad que `fKDC` casi siempre logra conectar.

  Abajo, Naive Bayes y la logística, que solo pueden trazar fronteras simples, no pueden. _Boosting_ la aproxima superponiendo rectángulos, como un «¿quién es quién?» de preguntas por coordenada. El soporte vectorial dibuja una frontera coherente, pero dura y pegada a uno de los lados de cada brazo; `fKDC`, en cambio, se centra y deja una zona gris, de duda, alrededor de los brazos.
]


#grid(columns: 4, column-gutter: 0.4em, row-gutter: 0.3em,
  ..("kn", "fkn", "kdc", "fkdc", "gnb", "lr", "svc", "gbt").map(c => image("img/espirales_lo-" + c + "-decision_boundary.svg")),
)

#intuicion[Arriba, #kn, #fkn, #kdc y #fkdc siguen la espiral. Abajo, #gnb y #logr no pueden, #svc la sigue con una frontera dura, sin probabilidades, y #gbt la aproxima con rectángulos.]

== Ablación: con y sin Fermat

// guion
#speaker-note[
  #set text(size: 17pt)
  Ahora, la pregunta de fondo: ¿qué se pierde si le quitamos Fermat? Cada punto es una semilla: en el eje $x$, el $R^2$ sin Fermat; en el $y$, con. Por encima de la diagonal, Fermat ayudó. Con `KDC`, parece ayudar en `lunas_lo` y `espirales_lo`, menos en `circulos_lo`.

  ▸ Con $k$ vecinos, casi todo cae sobre la diagonal, salvo en `espirales_lo`, donde `fkNN` gana en algunas semillas, aunque en dos cae a $R^2$ cero.

  Pero que gane no quiere decir que gane por Fermat. Vamos a desarmarlo.
]


#let ablacion(par, c) = image("img/" + c + "_lo-" + par + "-r2-scatter.svg", height: 30%)
#grid(columns: (auto, 1fr, 1fr, 1fr), column-gutter: 0.4em, row-gutter: 0.2em, align: center + horizon,
  [], raw("lunas_lo"), raw("circulos_lo"), raw("espirales_lo"),
  [#kdc \ vs. \ #fkdc], ..("lunas", "circulos", "espirales").map(c => ablacion("kdc-fkdc", c)),
  uncover("2-")[#kn \ vs. \ #fkn], ..("lunas", "circulos", "espirales").map(c => uncover("2-", ablacion("kn-fkn", c))),
)

#chico[$R^2$ por semilla, sin Fermat ($x$) y con Fermat ($y$). #fkdc parece ganar en `lunas_lo` y `espirales_lo`. #uncover("2-")[#fkn y #kn casi coinciden, salvo en `espirales_lo`: #fkn gana en algunas semillas, pero en dos cae a $R^2 approx 0$.]]

#intuicion[¿Qué se pierde al _quitar_ la distancia de Fermat? Cada punto es una repetición: por encima de la diagonal, Fermat ayudó.]

== Bien muestreado, $alpha$ y $h$ se confunden

// guion
#speaker-note[
  #set text(size: 17pt)
  Esta es la superficie de pérdida de `fKDC` en `circulos_lo`: qué tan bien ajusta, en entrenamiento, cada combinación de hiperparámetros. En el eje horizontal, el ancho de banda; en el vertical, $alpha$; en amarillo, lo mejor. Y no hay un óptimo: hay un risco, esa milanesa amarilla en diagonal, de parametrizaciones equivalentes, $log h prop alpha$.

  ▸ ¿Por qué? En vecindarios muy chicos, la distancia de Fermat es casi proporcional a la euclídea: lo que $alpha$ cambia en la distancia lo compensa $h$. Donde los datos son densos, el camino más barato entre vecinos es el salto directo. La regla de parsimonia elige entonces $alpha = 1$, y `fKDC` se reduce a `KDC`.

  Un caso que les va a sonar: en `pionono`, donde Sapienza y coautores encontraron que, para _clustering_, Fermat le gana a Isomap con $alpha$ entre 1,7 y 2,2, nuestro `fKDC`, clasificando, eligió $alpha = 1$ en todas las semillas.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  Entonces, ¿de dónde sale la ventaja que vimos en `lunas_lo`? `fKDC` elige $alpha = 1$ en todas las semillas, y en `espirales_lo`, en 24 de 25.

  ▸ Su $h$ es apenas menor que el de `KDC`. En el gráfico, cada punto es una semilla: cuanto más ancha elige `KDC` su campana respecto de la de `fKDC`, más $R^2$ pierde.

  ▸ La ventaja no viene de $alpha$: viene de la regla de parsimonia sobre una grilla de hiperparámetros más amplia. `fKDC` gana, pero no por Fermat, sino por elegir una campana un poco más angosta.

  [Si preguntan por las grillas: en la réplica de 100 semillas usé grillas anidadas; la ventaja en `lunas_lo` se sostiene con la regla de la tesis y desaparece si se elige ignorando $alpha$.]
]


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

== Dos variedades enroscadas

// guion
#speaker-note[
  #set text(size: 17pt)
  ¿Hay casos donde $alpha > 1$ sea irremplazable? Dos datasets en tres dimensiones. `helices_0`: dos hélices, como dos resortes, una dentro de la otra, con muy poca distancia entre un brazo y el siguiente.

  ▸ Y `hueveras_0`: dos maples de huevos encastrados. En los dos, las clases son casi idénticas en cada coordenada por separado: Naive Bayes y la logística no se distinguen del azar.
]


#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center)[#image("img/helices-scatter-3d.svg", height: 62%) `helices_0`: dos hélices, una dentro de la otra],
  uncover("2-", align(center)[#image("img/hueveras-scatter-3d.svg", height: 62%) `hueveras_0`: $z = plus.minus sin(x) sin(y)$]),
)

#intuicion[Dos resortes enroscados, y dos maples de huevos encastrados: en cada coordenada por separado, las clases son casi idénticas.]

== Donde Fermat sí aporta: `helices_0`

// guion
#speaker-note[
  #set text(size: 17pt)
  En `helices_0`, `fKDC` elige $alpha > 1$ en 13 de las 25 semillas y supera a `KDC` en 19.

  ▸ A la derecha, una semilla: aparece una isla en $alpha$ cerca de 3, $h$ cerca de $10^(-3)$, y todo lo que está a menos de un desvío tiene $alpha$ de 2,5 o más. Ahí sí hay un $alpha$ que ningún ancho sustituye.

  Y lo digo antes de que me lo pregunten: el `KDC` de la tesis usaba el `KernelDensity` de `scikit-learn`, que calcula mal las colas. Con un `KDC` exacto, en la réplica, esta ventaja de `fKDC` deja de ser significativa. La de `fkNN`, que viene ahora, no cambia.
]


#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center + horizon, image("img/helices_0-kdc-fkdc-r2-scatter.svg", height: 60%)),
  uncover("2-", align(center + horizon, image("img/helices_0-1188-fkdc-bandwidth-alpha-loss_contour.svg", height: 60%))),
)

#chico[#fkdc elige $alpha > 1$ en 13 de las 25 semillas y supera a #kdc en 19. #uncover("2-")[Der.: con $s = 1188$, una «isla» en $alpha approx 3$, $h approx 10^(-3)$; todo lo que está a menos de un desvío tiene $alpha >= 2.5$, y le saca $0.23$ de $R^2$ a #kdc.]]

// TODO(autor): con un KDC exacto (el KernelDensity de scikit-learn falla en las colas), la ventaja de 5 pp deja de ser significativa; ver preguntas-defensa.typ. Las comparaciones de fkn contra kn no cambian.
#intuicion[Cuando la curva de la variedad es comparable a la distancia entre observaciones, aparece un $alpha$ que ningún ancho de campana sustituye.]

== Donde Fermat sí aporta: #fkn en `helices_0`

// guion
#speaker-note[
  #set text(size: 17pt)
  A la izquierda, `fkNN` le gana o empata a $k$ vecinos en casi todas las semillas, y gana en 19 de 25.

  ▸ A la derecha, el porqué: el mejor puntaje en entrenamiento para cada $k$. Con $k = 1$ coinciden: el vecino más cercano según Fermat es el euclídeo. Pero a medida que crece el vecindario, el de $k$ vecinos euclídeo se desploma, porque la otra hélice pasa ahí nomás, y el de Fermat casi no cae: crece a lo largo del resorte. Como en los paseos. El aporte de Fermat no está en los vecindarios chicos, sino en que permite agrandarlos sin cruzar a la otra variedad.
]


#grid(columns: (1fr, 1fr), column-gutter: 1em,
  align(center + horizon, image("img/r2-fkn-kn-helices_0.svg", height: 58%)),
  uncover("2-", align(center + horizon, image("img/helices_0-fkn_kn-mean_test_score.svg", width: 100%))),
)

#chico[Izq.: $R^2$ por semilla; #fkn supera a #kn en 19 de las 25. #uncover("2-")[Der.: mejor log-verosimilitud en validación cruzada para cada $k$; la de #fkn casi no cae, la de #kn se desploma.]]

#intuicion[Como en los paseos: el vecindario de Fermat sigue la hélice, y agrandarlo no trae vecinos de la otra clase. Con $k = 1$ coinciden; Fermat aporta cuando el vecindario crece.]

== Donde Fermat sí aporta: `hueveras_0`

// guion
#speaker-note[
  #set text(size: 17pt)
  En `hueveras_0`, solo la familia de los núcleos logra un $R^2$ positivo, entre 0,25 y 0,30.

  ▸ Y es el experimento más limpio. En las 12 repeticiones en que `fkNN` y $k$ vecinos eligen el mismo $k$: si `fkNN` elige $alpha = 1$, empatan las 4 veces; ▸ si elige $alpha > 1$, gana en 7 de 8. Mismo vecindario: lo único que cambia es cómo se mide.
]


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

// guion
#speaker-note[
  #set text(size: 17pt)
  Ahora, lo feo. `pinguinos`: medidas del cuerpo de tres especies de pingüinos. Tres en milímetros, entre 13 y 230; la masa, en gramos, entre 2700 y 6300.

  ▸ Es casi linealmente separable: la logística llega a 0,96, y la familia de los núcleos, a 0,40–0,50. Lleno, crudo; vacío, estandarizado.

  ▸ ¿Por qué? La distancia euclídea es, en la práctica, la diferencia de masa, y Adelie y Chinstrap tienen masas indistinguibles. Fermat la hereda: no arregla la métrica con que se construye. La logística, _boosting_ y Naive Bayes razonan cada variable por separado y no lo padecen. Estandarizando, `fKDC` pasa de 0,42 a 0,96. Todos aprendemos a estandarizar antes de clasificar; queríamos ver si un método que aprende la métrica podía arreglárselas solo. No tan fácilmente.
]


#grid(columns: (55%, 45%), column-gutter: 1em,
  [#only(1, image("img/pinguinos-scatter.svg"))#only("2-", image("img/pinguinos-crudo-vs-std-r2.svg"))],
  align(horizon)[
    `pinguinos`: picos y aletas en mm, de 13 a 230; masa en g, de 2700 a 6300.

    #pause
    Casi linealmente separable: #logr llega a $R^2 approx 0.96$; la familia de los núcleos, a $0.40$--$0.50$.

    #pause
    #v(0.5em)
    La distancia euclídea es, en la práctica, la diferencia de masa, y Fermat la hereda. Estandarizando, #fkdc pasa de $0.42$ a $0.96$.
  ],
)

#intuicion[Comparar pingüinos sumando milímetros y gramos es como comparar personas por su altura en kilómetros y su peso en gramos: solo cuenta el peso.]

== Lo feo: dimensiones de ruido

// guion
#speaker-note[
  #set text(size: 17pt)
  Algo parecido pasa con el ruido. `eslabones`: dos anillos enganchados, en tres dimensiones. Le agregamos 12 dimensiones de ruido gaussiano, con la misma distribución en las dos clases: no informan nada.

  ▸ La distancia suma todas las coordenadas: el ruido domina las aristas del grafo, y Fermat lo hereda.

  ▸ Lleno, en 3D; vacío, en 15D. En `helices_12` y `hueveras_12` nadie se distingue del azar; en `eslabones_12`, `fKDC` cae a 0,24, mientras _boosting_, que pregunta coordenada por coordenada, queda en 0,92. Naive Bayes también aguanta: ve que en esas dimensiones las marginales son iguales para las dos clases. Los métodos por distancia, no.
]


#grid(columns: (55%, 45%), column-gutter: 1em,
  [
    #only(1, align(center + horizon, image("img/eslabones-scatter-3d.svg", height: 66%)))
    #only("2-", grid(columns: 2, gutter: 2pt,
      ..("pionono", "eslabones", "helices", "hueveras").map(f => image("img/" + f + "-caida_r2-15d.svg")),
    ))
  ],
  align(horizon)[
    #only(1)[`eslabones`: dos anillos enganchados, en 3D. Le agregamos 12 dimensiones de ruido gaussiano, independientes de la clase.]
    #only("2-")[
      $ norm(x - y)^2 = sum_(k=1)^D (x_k - y_k)^2 $
      suma _todas_ las coordenadas: el ruido domina las aristas del grafo, y Fermat lo hereda.
    ]
    #only(3)[
      #v(0.4em)
      #chico[Lleno, en 3D; vacío, en 15D. En `eslabones_12`, #fkdc cae a $R^2 approx 0.24$; #gbt queda en $0.92$.]
    ]
  ],
)

#only(3, intuicion[Si casi todas las coordenadas son ruido, toda distancia, euclídea o de Fermat, mide sobre todo ruido. #gnb y #gbt pueden ignorar las coordenadas que no informan; #kdc y #fkdc, no.])

== Lo feo: reducir antes

// guion
#speaker-note[
  #set text(size: 17pt)
  La salida es limpiar antes de medir. En `mnist`, por cómputo y por todo lo anterior, no pudimos correr en las 784 dimensiones originales: primero PCA, aprendido solo en entrenamiento, de 784 a 96 dimensiones, con el 90 % de la variación.

  ▸ Después, `fKDC` llega a 0,74: un $R^2$ digno, a la par de la logística, 0,76, un poco por debajo de `KDC`, 0,77, y por encima de _boosting_, 0,67. Pero honestamente, no mejor que una logística sobre las mismas componentes. Vaya sorpresa: todos los métodos requieren un análisis cuidadoso antes de largarlos sobre los datos.
]


#grid(columns: (50%, 50%), column-gutter: 1em,
  align(horizon)[
    En `mnist`, PCA primero: de $784$ a $96$ dimensiones, con el 90 % de la variación.

    #v(0.5em)
    #uncover("2-")[Después, #fkdc llega a $R^2 approx 0.74$: a la par de #logr ($0.76$), por debajo de #kdc ($0.77$) y por encima de #gbt ($0.67$).]
  ],
  align(center + horizon, image("img/mnist-r2-boxplot.svg", height: 62%)),
)

#intuicion[Hay que limpiar antes: quedarse con las direcciones en que los datos más varían, y recién ahí medir.]

== En síntesis

// guion
#speaker-note[
  #set text(size: 17pt)
  Entonces, en síntesis.

  Funciona cuando la curvatura de la variedad es comparable a la separación entre observaciones, como en `helices_0` y `hueveras_0`, y sobre todo en $k$ vecinos: permite agrandar el vecindario sin cruzar a otra variedad.

  ▸ No aporta donde los datos son densos: $alpha$ y $h$ se confunden, y `fKDC` se reduce a `KDC`; sus ventajas en `lunas_lo` y `espirales_lo` vienen de la regla de parsimonia.

  ▸ Hereda los problemas de la euclídea: necesita atributos en escala común y pocas dimensiones de ruido puro.

  ▸ Y ningún clasificador domina: los núcleos tienen el mejor $R^2$ en 14 de 20 datasets, pero la mejor exactitud en 9. La distancia de Fermat no es una varita mágica: es una buena regla para medir cuando los datos siguen formas curvas y bien muestreadas.
]


#text(size: 20pt)[
- *Funciona* cuando la curvatura de la variedad es comparable a la separación entre observaciones (`helices_0`, `hueveras_0`), y sobre todo en #fkn: permite agrandar el vecindario sin cruzar a otra variedad.
#pause
- *No aporta* donde los datos son densos: $alpha$ y $h$ se confunden, y #fkdc se reduce a #kdc. Sus ventajas en `lunas_lo` y `espirales_lo` vienen de la regla de parsimonia.
#pause
- *Hereda* los problemas de la distancia euclídea: exige atributos en escala común y pocas dimensiones de ruido puro.
#pause
- *Ningún clasificador domina*: la familia de los núcleos tiene al mejor $R^2$ en 14 de 20 datasets, pero la mejor exactitud en 9.
]

#intuicion[La distancia de Fermat no es una varita mágica: es una buena regla para medir cuando los datos siguen formas curvas y bien muestreadas, y una regla más cuando no.]

= El futuro

== Después de la entrega

// guion
#speaker-note[
  #set text(size: 17pt)
  Hasta acá, la tesis. Después de entregar seguí, y lo digo sin orgullo y con bastante cansancio: buena parte del cuerpo de la tesis es mi texto, pero todo lo que les cuento de acá en adelante lo hice con asistentes de IA, y revisé sus resultados.

  Les pedí, básicamente, la versión 2 de la tesis: de 25 a 100 semillas, prerregistradas antes de mirar; de 8 a 22 clasificadores, incluidas redes para datos tabulares, modelos de lenguaje y clásicos que no habíamos considerado; de 24 a 33 datasets, con sintéticos nuevos donde esperábamos que Fermat anduviera muy bien o se rompiera. En total, 72 600 tareas, en un par de días en mi computadora, sin una falla. Les cuento dos resultados.
]


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
#chico[Los datasets cuentan las variantes estandarizadas. La réplica la hice con asistentes de IA, y revisé sus resultados.]

== Parte de la ventaja era calibración

// guion
#speaker-note[
  #set text(size: 17pt)
  El primero: ¿no estará $alpha$ haciendo que las probabilidades de `fKDC` sean demasiado confiadas, muy pegadas a cero y uno, y por eso vemos mejores $R^2$? Para vigilarlo, agregamos un paso de calibración: se recalibra cada clasificador por temperatura, que no cambia qué clase gana, solo cuán seguro está.

  En el gráfico, la diferencia de $R^2$ entre `fKDC` y `KDC`, semilla a semilla en las 100 semillas, con su intervalo de confianza del 95 %. Lleno, el $R^2$ crudo; vacío, el templado.

  ▸ En crudo, `fKDC` le gana a `KDC` en 12 datasets. Templado, quedan 7, casi todas por menos de un punto; la mayor, `hueveras_0`, por 2. Y en `eslabones_12` y `pionono_12`, se invierte. `KDC` acierta, pero duda; `fKDC` se juega más: parte de la ventaja era confianza, no puntería.
]


#grid(columns: (58%, 42%), column-gutter: 1em,
  align(center, image("img/defensa/fkdc-kdc-crudo-templado.svg", height: 68%)),
  align(horizon)[
    Diferencia de $R^2$ entre #fkdc y #kdc, semilla a semilla, con IC del 95 %: lleno, crudo; vacío, _templado_, con cada clasificador recalibrado por temperatura.

    #pause
    #v(0.4em)
    En crudo, #fkdc gana en 12 datasets. Templado, quedan 7, casi todas por menos de un punto; la mayor, `hueveras_0`, por 2. En `eslabones_12` y `pionono_12`, se invierte.
  ],
)

#intuicion[#kdc acierta, pero duda; #fkdc se juega más. Parte de la ventaja era esa: confianza, no puntería.]

== Contra el estado del arte

// guion
#speaker-note[
  #set text(size: 17pt)
  El segundo: ¿cómo le va a `fKDC` contra algo así como el estado del arte? Cuatro clasificadores, sobre los datasets de la tesis: `fKDC`; un `KDC` con un ancho de banda por variable, más fuerte que la línea de base de la tesis; un _superlearner_, un ensamble de 14 clasificadores clásicos; y TabPFN, una red neuronal enorme, entrenada para aprender a aprender: le pasás la tabla de entrenamiento y las nuevas observaciones, y responde, sin ajustar nada.

  ▸ `fKDC` nunca le gana a TabPFN. Pero fíjense que la escala estira la zona cercana a $R^2 = 1$: estamos mirando con lupa los nueves, y nunca queda demasiado lejos, salvo con 12 dimensiones de ruido, que ya sabemos que son su kriptonita. Ahí, el ancho por variable rescata al `KDC`, porque le asigna un ancho grueso a las dimensiones de ruido. Pero en datos reales como `digitos` y `mnist`, `fKDC` le gana a ese `KDC` adaptativo.
]


#grid(columns: (58%, 42%), column-gutter: 1em,
  align(center, image("img/defensa/mapa-tabpfn.svg", height: 68%)),
  align(horizon)[
    #only(1)[Mediana de $R^2$ en 100 semillas para cuatro clasificadores: #fkdc, #kdc con un ancho por variable, un _superlearner_ de 14 clasificadores clásicos, y TabPFN, una red preentrenada con millones de tablas simuladas.]
    #only("2-")[#fkdc nunca supera a TabPFN, pero queda cerca, salvo con 12 dimensiones de ruido.

      #v(0.4em)
      En `digitos` y `mnist`, le gana al #kdc con un ancho por variable.]
  ],
)
// Sin fuga ni contaminación: ver «¿El resultado de TabPFN es legítimo?» en preguntas-defensa.typ
// Para el autor: el superlearner clásico apila con pesos convexos, sobre probabilidades fuera del pliego, 14 clasificadores rápidos (fkdc.clasicos.biblioteca): LR, LDA, QDA, GNB, kNN con k = 5, 15 y 45, tres KDC (isótropo estandarizado, isótropo crudo como el de la tesis, y ARD), GPC, GBT y extra trees. No incluye a fkdc ni a TabPFN.
// Para el autor: en digitos y mnist, fkdc también supera al KDC con un ancho por variable (0.977 contra 0.896, y 0.748 contra 0.664; medianas de R² crudo con 100 semillas, data/defensa/mapa-tabpfn.csv).

#intuicion[Una red entrenada de antemano con millones de problemas inventados resuelve uno nuevo de un vistazo, sin ajustar nada. La escala estira la zona cercana a $R^2 = 1$, donde se pelea el podio.]

== Una pregunta abierta

// guion
#speaker-note[
  #set text(size: 17pt)
  Una pregunta que nos puede quedar es para qué sirve todo esto, si la versión 2 de la tesis la hizo una IA casi sin que le dijera nada. Creo que sirve, por dos razones. Primero, el problema que tuvo era más fácil: ya había un edificio de investigación hecho, y mejorar una métrica bien definida como el $R^2$ es mucho más fácil que buscar en el espacio abierto de los diseños de experimentos.

  ▸ Segundo, y más importante: todo lo que hablamos hoy son métodos para aprender distancias y estimar variedades que no conocemos. ▸ Los ámbitos más interesantes de hoy, el lenguaje, el video, la robótica, la genómica, son de altísima dimensión, y pareciera que alguna variante de la hipótesis de la variedad se sostiene en ellos. Tener herramientas para acercarnos a esas construcciones tan difíciles de caracterizar, y empezar a pensar en ellas, me parece valioso de por sí.

  [Clic a «¡Gracias!»] Muchas gracias por escucharme.
]


#align(center + horizon)[
  Lenguaje, video, robótica, genómica: datos de dimensión altísima que, según la hipótesis de la variedad, viven cerca de variedades que no conocemos.

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
