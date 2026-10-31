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

// Doble público: el cuerpo de la diapo es técnico sin abrumar; el recuadro al pie da
// la intuición para quien no lee matemática
#let intuicion(cuerpo) = {
  v(1fr)
  block(width: 100%, fill: luma(245), inset: 10pt, radius: 4pt, text(size: 16pt, cuerpo))
}
#let silbato = intuicion

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

#intuicion[Hay una idea más general y poderosa detrás, pero esta heurística de sumar las distancias infladas a lo largo del grafo alcanza para aproximarla.]

#focus-slide[¿Servirá para clasificar?]

= El plan

== Del estimador al clasificador

//TODO: Mismo estilo de linea temporal arriba, una diapo por parte abajo
#grid(columns: 4, column-gutter: 0.8em,
  [*2005* \ #chico[Pelletier: densidad por núcleos en variedades]],
  [*2008* \ #chico[Loubes y Pelletier: clasificador binario]],
  [*2009* \ #chico[Henry y Rodríguez: mejores cotas, y la esfera]],
  [*Esta tesis* \ #chico[multiclase, sobre una variedad *desconocida*]],
)

== Una diapo para el clasificador ciompleto fKDC

//TODO: en formato de formula, el clasxificador implementadoi como FermatKDClassifier. Zócalo un poco más pequeño par FermatKneigh


- La distancia de Fermat _muestral_ reemplaza a la de la variedad, que no conocemos.
- #fkdc: clasificador de densidad por núcleos con esa distancia. #fkn: $k$ vecinos más cercanos con la misma.
- Con $alpha = 1$, son los euclídeos de siempre.

#silbato[
  $hat(Pr)(G = g_i | X = x) prop hat(pi)_i hat(f)_i (x)$, con $hat(f)_i$ evaluada sobre $D_(Q_i, alpha)(x, dot)$ y $Q_i = {x} union {x_j : g_j = g_i}$
]

== El código

¿Qué se implementó?
- `SampleFermatDistance(X, alpha)`computa distancias entre dos (conjuntos de) puntos "a través" del grafo completo de X mínimamente eficiente, con distancia para puntos out-of-sample
  - FermatKNeighbors reutilizando scikit_learn::KNeighbors con distnace="precomnputed"
- KernelDensityClassifier con distancia configurable
  - FermatKDC como KernelDensityClassifier(dist="fermat")
// todo: signatures correctas
- Un marco de evaluación utilizando sickit-learn::Pipelines para comparar sistematicamente nuestros clasificadores, en repetidas Tareas sobre Datasets prerregistrados.

== libreria?

//todo: si llegamos con la l;ibreria, aca el link a pypi

== 8 estimadores
//todo: lsitarlos rapidito
== 20 datasets
// todo: algunos ejemplos visuales, 2 x 4 = 8 slots, algunos 2d, alguno 3d, los digitos de mnist no como scatter sino mosaico de imagenes chiquitas.
== 25 repeticiones
// todo: una mencion a la aleatporiedad, buscamos resultados repetibles asi que repetimos c/ tarea 25 veces y comapramos comportamiento _mediano_.
#silbato[(mediano)]

// Estas diapos rememplaan "el banco de pruebas"
// == El banco de pruebas

// #grid(columns: 4, column-gutter: 0.5em,
//   image("img/lunas_lo-scatter.svg"),
//   image("img/espirales_lo-scatter.svg"),
//   image("img/helices_0-scatter.svg"),
//   image("img/hueveras_0-scatter.svg"),
// )

#align(center)[
  #grid(columns: 3, column-gutter: 3em,
    cifra("20", "datasets: sintéticos y reales chicos"),
    cifra("8", "clasificadores"),
    cifra("25", "repeticiones cada uno"),
  )
]

= Resultados

// todo: estilo lo bueno, lo malo y lo feo
// no quiero mencionar nada o casi nada de lo post-tesis hasta la seccion "el futuro"

== Overall
// todo dos diapos, primero para fkdc y luego para fkn
// tres filas:
// 1. nombres de datasets (a 45 grados tal vez
// 2. fkdc mejor/igual/peor que kdc
// 3. fkdc mejor/igual/peor que best clf@dataset en Rˆ2
// 
// segunda diapo: igual para fkn versus kn y best-in-class
// 
== Lo bueno
=== fkdc en espirales
primera diapo: izq spuerficie de decision, derecha boxplot R^2
pocos puntos, mucha curva
segunbda diapo: decision boudnaries para todos los clfs
  #image("img/espirales_lo-fkdc-decision_boundary.svg")
=== fkn en helices_0
Fermat agranda el vecindario sin cruzar

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/r2-fkn-kn-helices_0.svg"),
  align(horizon)[
    En `helices_0`, con el mismo $k$, #fkn le gana a #kn en la gran mayoría de las semillas.

    // todo: imagen de jeor r^2 como funcion de k en fkn y kn con r^2 de fkn estable y el de kn cayendo
    #silbato[Es lo que mostraba la animación: el vecindario sigue la hélice.]
  ],
)

== lo malo

=== bien muestreado, $alpha$ y $h$ se confunden

#grid(columns: (50%, 50%), column-gutter: 1em,
  image("img/espirales_lo-1434-fkdc-bandwidth-alpha-loss_contour.svg"),
  align(horizon)[
    De cerca, la distancia de Fermat es la euclídea con otra escala.

    #v(0.5em)
    Aparece un «risco» de combinaciones equivalentes, y la regla de parsimonia elige $alpha = 1$.
  ],
)

== lo feo
=== Sin escalar, nada anda
// todo: dos diapos, la primera describiendo el datset de pinguinos y a la derecha el boxplot original; la segunda es esta que esta aca abajo, con el gráfico explicado
#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/pinguinos-crudo-vs-std-r2.svg"),
  align(horizon)[
    En `pinguinos`, un atributo en gramos tapa a los demás, en milímetros.

    #v(0.5em)
    Fermat es tan sensible a la escala como la euclídea. Estandarizar lo resuelve.
  ],
)

=== sensibilidad al ruido en altas dimensiones

- nota sobre cómo al calcular dist. euclidea en el grrafo completo se "come'
 el ruido de dimensiones que no difernecian en nada, GNB o GBT pueden ïgnorar"las dimensioens no-informativas, fermat KDC (o KDC a secas) no"
- primero reducir linealmente con PCA y lueg oprobar fermat funciona (caso MNIST -- boxplot o numero ehadline mencionado acá)


= El futuro

== Después de la entrega

// todo: Armar esto como un äntes y despues, 25 v 100, 8 vs 22, 25 vs 33, total tareas vs total tareas
// mencionar en una diapo sgte algunos de los clfs neuvos relevantes, y en otra algunos de los datasets nuevos mas interesantes
// mencion a la inclkusion de clf LLMs y NNs para datos tabulares
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
// todo: pregunta: el "siuperlearner clasico"que es? incluye tambien a fkdcv? sobre quienes esta componiendose? otros de esta misma tabla?
// todo: mismo grafico pero con r^2 en symlog scale apra que se distinga el peloton de frente cerca de uno y los rˆ2 de cero no rompan tampoco.
// dejar nota para el autor sobre como en digitos, mnist fkdc es superior tambien a kdc con un ancho por variable
// 
#grid(columns: (58%, 42%), column-gutter: 1em,
  image("img/defensa/mapa-tabpfn.svg"),
  align(horizon)[
    Una red preentrenada con millones de tablas simuladas: queda entre los mejores en 19 de 33 datasets.

    #v(0.5em)
    Los métodos de la tesis aprenden de 400 puntos.
  ],
)

// Sin fuga ni contaminación: ver «¿El resultado de TabPFN es legítimo?» en preguntas-defensa.typ
// TODO: estas dos diapos que sigue, ver si se les puede dar un minimo significado intuitivo sin agobiar a la audiencia, es solo apra decir "algo con ia". Si no, menci´øn al aprendizaje de\ variedades como aprendizaje de representaciones de Bengio y de ahí a transformers (encoder decoder) como componente fundamental de LLMs 
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

// todo: nada de respaldo
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
