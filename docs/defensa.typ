#import "@preview/touying:0.6.2": *

#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: [Defensa de tesis · Maestría en Estadística Matemática · FCEN-UBA],
)
#show link: set text(blue)
#set text(font: "Fira Sans", weight: "light", size: 20pt, lang: "es")
#set strong(delta: 100)
#set par(justify: false)

#let phi = math.phi.alt
#let ind = $ op(bb(1)) $
#let bu(x) = $bold(upright(#x))$
#let fkdc = `fKDC`
#let kdc = `KDC`
#let fkn = `fkNN`
#let kn = `kNN`
#let chico(cuerpo) = text(size: 15pt, cuerpo)
#let fuente(cuerpo) = text(size: 13pt, fill: gray.darken(30%), cuerpo)

// Tablas generadas por scripts/figuras_defensa.py
#let tabla-csv(ruta, alineacion: auto, tamano: 15pt) = {
  let filas = csv(ruta)
  set text(size: tamano)
  table(
    columns: filas.first().len(),
    align: alineacion,
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + gray.lighten(40%)) },
    table.header(..filas.first().map(c => strong(c))),
    ..filas.slice(1).flatten(),
  )
}
#let num(x, d: 2) = {
  let v = float(x)
  str(calc.round(v, digits: d))
}

#title-slide(
  author: [Lic. Gonzalo Barrera Borla · Director: Dr. Pablo Groisman],
  title: "Distancia de Fermat en Clasificadores de Densidad por Núcleos",
  subtitle: "Defensa de tesis de Maestría en Estadística Matemática",
  institution: [Departamento de Matemática, FCEN, UBA],
)

= El problema

== Clasificar con densidades

El _clasificador de Bayes_ asigna a cada $x$ la clase más probable. Con la regla homónima, también da probabilidades, y por eso un clasificador _suave_:

$
  hat(Pr)(G = g_i | X = x) = (hat(pi)_i hat(f)_i (x)) / (sum_(k=1)^K hat(pi)_k hat(f)_k (x))
$

Si cada $hat(f)_k$ es un estimador de densidad por núcleos, $hat(f)(x) = N^(-1) sum_i K_h (x - x_i)$, el resultado es el *clasificador de densidad por núcleos* (#kdc).

#fuente[ESL §6.6; Parzen 1962; Wand y Jones 1995]

== La maldición de la dimensión

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/curse-dim.svg"),
  [
    Con $X tilde.op "Unif"([-1, 1]^d)$ y una ventana de lado $2h$, la fracción esperada de puntos dentro de la ventana es $h^d$: para $d = 50$ y $h = 0.95$, menos del 1 %.

    #v(0.5em)
    *Hipótesis de la variedad*: los datos en $RR^D$ se concentran cerca de una variedad $cal(M)$ de dimensión $d_(cal(M)) << D$.
  ],
)

== Densidades en variedades

Pelletier (2005) adapta el estimador a una variedad de Riemann $(cal(M), g)$ compacta y sin frontera:

$
  f_(N, K)(p) = 1 / N sum_(i=1)^N 1 / h^d 1 / (theta_(X_i)(p)) K((d_g (p, X_i)) / h)
$

con $d_g$ la distancia de Riemann y $theta$ la densidad de volumen. Loubes y Pelletier (2008) lo usan en la regla de Bayes para clasificar.

#v(0.6em)
#align(center)[*Pero la variedad, y con ella $d_g$, son desconocidas.*]

= La distancia de Fermat

== Distancia de Fermat, macroscópica y muestral

Para $f$ positiva y $beta >= 0$, la distancia de Fermat minimiza un funcional de caminos:

$
  cal(D)_(f, beta)(x, y) = inf_(gamma in Gamma) integral_gamma f^(-beta)
$

Su versión muestral, para $alpha >= 1$, se calcula sobre la muestra $bu(X)$:

$
  D_(bu(X), alpha)(x, y) = inf { sum_(j=1)^(K-1) norm(q_(j+1) - q_j)^alpha : (q_1, dots, q_K) "camino de" x "a" y }
$

y converge: $n^beta D_(bu(X)_n, alpha) -> mu cal(D)_(f, beta)$, con $beta = (alpha - 1) slash d$.

#fuente[Groisman, Jonckheere y Sapienza (2022); Little et al. (2021)]

== La propuesta: #fkdc y #fkn

- Para clasificar un punto nuevo $x_0$, cada clase $g_i$ define $Q_i = {x_0} union {x_j : g_j = g_i}$, y el núcleo se evalúa sobre $D_(Q_i, alpha)(x_0, dot)$ en lugar de la distancia euclídea.
- #fkdc: el clasificador de Loubes y Pelletier con la distancia de Fermat muestral; hiperparámetros $(alpha, h)$.
- #fkn: $k$ vecinos más cercanos según la misma distancia; hiperparámetros $(alpha, k)$.
- Con $alpha = 1$, ambos coinciden con sus pares euclídeos #kdc y #kn.

= Diseño experimental

== 20 datasets, 8 clasificadores, 25 semillas

#grid(columns: 4, column-gutter: 0.5em,
  image("img/lunas_lo-scatter.svg"),
  image("img/espirales_lo-scatter.svg"),
  image("img/helices_0-scatter.svg"),
  image("img/hueveras_0-scatter.svg"),
)

#chico[
- Sintéticos en variedades (2D y 3D, con ruido bajo y alto, y con 12 dimensiones de ruido), más `anteojos`, `iris`, `vino`, `pinguinos`, `digitos` y `mnist`.
- #fkdc, #kdc, #fkn, #kn, `GNB`, `LR`, `SVC` y `GBT`; partición 50/50 por semilla; grilla por validación cruzada de 5 pliegos con la regla de $1 sigma$.
- Métrica: $R^2$ de McFadden, $1 - cal(l)(cal(C)) slash cal(l)(cal(C)_0)$, con $cal(C)_0$ las frecuencias de clase; y la exactitud.
]

= Resultados de la tesis

== Podio por $R^2$ mediano

- #fkdc obtuvo el máximo en 7 de los 20 datasets y #fkn en 5 (dos empatados con #kn); `GBT` en 5, #kdc en 2.
- Por exactitud la ventaja se diluye, y `SVC` resulta casi imbatible.
- En los vecindarios pequeños la distancia de Fermat coincide con la euclídea: en datasets bien muestreados hay un «risco» $log h prop alpha$ de parametrizaciones equivalentes, y la regla de $1 sigma$ elige $alpha = 1$.

#align(center)[#image("img/espirales_lo-1434-fkdc-bandwidth-alpha-loss_contour.svg", height: 46%)]

== Cuándo aporta: `helices_0`

#grid(columns: (55%, 45%), column-gutter: 1em,
  image("img/r2-fkn-kn-helices_0.svg"),
  [
    Con el mismo $k$ que #kn pero $alpha > 1$, #fkn mejora: la distancia de Fermat permite agrandar el vecindario sin cruzar a la otra hélice.

    #v(0.5em)
    La mejora es neta, atribuible a la distancia, y no a la interacción entre la regla de parsimonia y una grilla más grande.
  ],
)

== Cuándo no: ruido y escala

#grid(columns: 2, column-gutter: 1em,
  image("img/pionono-caida_r2-15d.svg"),
  image("img/pinguinos-crudo-vs-std-r2.svg"),
)
#chico[Con 12 dimensiones de ruido la familia de núcleos se desploma y la distancia de Fermat no lo remedia. La familia tampoco es invariante a la escala: en `pinguinos`, un atributo en gramos bastó para confundir dos especies; estandarizar lo resolvió.]

== Conclusiones de la tesis

- La distancia de Fermat aporta cuando la variedad es muy curva o está ralamente muestreada: `helices_0` y `hueveras_0`.
- En datasets bien muestreados, $alpha$ y $h$ son intercambiables y #fkdc colapsa a #kdc.
- Las dimensiones de ruido y la escala de los atributos son sus límites.
- Ningún algoritmo fue universalmente óptimo.

= Después de la entrega

== El trabajo futuro, hecho

La tesis cierra con una lista de pendientes. Tres se pudieron hacer:

#table(
  columns: (1fr, 1fr),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + gray.lighten(40%)) },
  inset: 8pt,
  table.header(strong[Pendiente en la tesis], strong[Lo que se hizo]),
  [Reemplazar el recuento de podios por una comparación estadística seria], [Wilcoxon de rangos con signo por dataset, con Holm, sobre 100 semillas],
  [Calibrar los clasificadores antes de comparar $R^2$], [$R^2$ templado: cada clasificador recalibrado por temperatura],
  [Aprender la diagonal $bu(H) in cal(D)$ en lugar de fijarla], [#kdc con un ancho de banda por variable, elegido por validación cruzada],
)

Además: una réplica prerregistrada con 100 semillas, y la comparación con el estado del arte.

== Réplica con 100 semillas, prerregistrada

#grid(columns: (48%, 52%), column-gutter: 1em,
  chico[
    - Las 25 semillas de la tesis más 75 nuevas; 72 600 tareas, 0 fallas.
    - #kdc exacto: el `KernelDensity` de `scikit-learn` calcula mal las colas profundas, y la tesis lo había usado.
    - Grillas emparejadas: la de $h$ de #fkdc es un subconjunto exacto de la de #kdc.
    - Prerregistro fechado antes de ver resultados: pruebas, márgenes y regla de veredicto.
  ],
  tabla-csv("data/defensa/replica-veredictos.csv", alineacion: (left, right, right, right, right), tamano: 13pt),
)
#fuente[132 afirmaciones de la tesis, cada una con su prueba (Wilcoxon apareado con Holm, equivalencia, intervalos).]

== Qué se sostiene y qué no

#grid(columns: 2, column-gutter: 1.5em,
  [
    *Se sostiene*
    #chico[
    - #fkdc > #kdc en `lunas_lo` (73 semillas a favor, 0 en contra, 27 empates) y en `espirales_lo` (53–11–36).
    - #fkn > #kn en `helices_0` (83–8; +0.16 de $R^2$) y en `hueveras_0` (61–15).
    - El podio casi no cambia: #fkdc 7, `GBT` 5, #fkn 4, #kdc 3.
    ]
  ],
  [
    *No se sostiene*
    #chico[
    - La ventaja de 5 pp de #fkdc sobre #kdc en `helices_0`: con #kdc exacto, 43–32–25, sin diferencia significativa.
    - «La distancia de Fermat aumenta la varianza»: en `helices_0`, #fkn tiene *menos* dispersión que #kn.
    - «Todos tuvieron al menos un dataset favorable»: `GNB` no tuvo ninguno.
    ]
  ],
)

== $R^2$ crudo contra $R^2$ templado

#grid(columns: (58%, 42%), column-gutter: 1em,
  image("img/defensa/fkdc-kdc-crudo-templado.svg"),
  [
    #chico[
    Diferencia de Hodges–Lehmann #fkdc − #kdc, con su intervalo del 95 %; \* = significativa con Holm.

    #v(0.4em)
    Parte de la ventaja de #fkdc en $R^2$ es calibración: #kdc es poco confiado y el exponente $alpha$ agudiza las probabilidades. En `lunas_lo` y `espirales_lo` la ventaja sobrevive al templado; en `anteojos`, `digitos` y `mnist`, no.
    ]
  ],
)

== Wilcoxon por dataset: el grupo líder

#grid(columns: (52%, 48%), column-gutter: 1em,
  {
    let filas = csv("data/defensa/grupo-lider.csv").slice(1).filter(f => f.at(0) == "tesis")
    set text(size: 15pt)
    table(
      columns: 3,
      align: (left, right, right),
      stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + gray.lighten(40%)) },
      table.header(strong[Clasificador], strong[$R^2$ crudo], strong[$R^2$ templado]),
      ..filas.map(f => (raw(f.at(1)), num(f.at(2), d: 0), num(f.at(3), d: 0))).flatten(),
    )
  },
  chico[
    En cada uno de los 33 datasets, el *grupo líder* es la mejor mediana más los clasificadores que no difieren significativamente de ella (Wilcoxon de rangos con signo, Holm, $alpha = 0.05$).

    #v(0.4em)
    Entre los clasificadores de la tesis, #fkdc es el que más veces queda en el grupo líder. Con el $R^2$ templado, #kdc pasa de 5 a 12: otra vez, calibración.
  ],
)

= El estado del arte

== TabPFN: la predictiva posterior, aprendida

Un _prior-data fitted network_ se entrena una sola vez sobre millones de tablas simuladas de un prior $p(phi)$:

$
  theta^star = arg min_theta EE_(phi tilde p(phi)) EE_(D, (x_star, y_star) tilde p(dot | phi)) [ -log q_theta (y_star | x_star, D) ]
$

Por ser la pérdida logarítmica una regla propia, el óptimo es la *predictiva posterior* bajo el prior,
$ q_(theta^star)(y | x, D) approx integral p(y | x, phi) p(phi | D) dif phi $
Con una tabla nueva no se ajusta nada: la tabla entra como contexto y la predicción sale de una pasada.

#fuente[Müller et al. (2022); Hollmann et al. (2023, 2025); Nagler (2023)]

== El #kdc es una cabeza de atención

Con probabilidades de clase empíricas, el #kdc gaussiano es la regresión de Nadaraya–Watson de las etiquetas $e_(g_i)$:

$
  hat(Pr)(dot | x) = sum_i "softmax"_i (- norm(x - x_i)^2 / (2 h^2)) e_(g_i)
$

y una cabeza de atención de un _transformer_ es $sum_i "softmax"_i (q^top k_i slash sqrt(d)) v_i$. Con $q = x slash h$, $k_i = x_i slash h$ y $v_i = e_(g_i)$ (el término en $norm(x)^2$ se cancela en el softmax), son la misma cuenta.

#v(0.4em)
#fkdc cambia el núcleo por uno basado en $D_(Q, alpha)$, fijado por una regla; TabPFN *aprende* el núcleo, con muchas capas y mirando todas las columnas.

== Cuatro métodos, veinte datasets

#grid(columns: (58%, 42%), column-gutter: 1em,
  image("img/defensa/mapa-tabpfn.svg"),
  chico[
    Mediana del $R^2$ crudo sobre 100 semillas.

    #v(0.4em)
    TabPFN-3.5 queda en el grupo líder del conjunto ampliado (22 clasificadores) en 19 de 33 datasets; el superlearner, en 5; #fkdc, en 2. TabPFN trae un prior aprendido de millones de tablas; los métodos de la tesis aprenden de 400 puntos.

    #v(0.4em)
    Con dos métodos clásicos, sin redes preentrenadas, se recupera buena parte de la distancia, y en los datasets con ruido se la invierte.
  ],
)

== Respuestas clásicas

- *#kdc con un ancho de banda por variable*, elegido maximizando la verosimilitud de $P(y | x)$ dejando uno afuera (Hall, Racine y Li, 2004). Es la diagonal $bu(H) in cal(D)$ del trabajo futuro: las variables irrelevantes reciben anchos grandes y quedan suavizadas.
  #chico[En las variantes con 12 dimensiones de ruido, el menor ancho de una variable de ruido es una mediana de 3.5 a 4.9 veces el mayor ancho de una variable de señal.]
- *Superlearner*: apilamiento convexo, óptimo en log-verosimilitud, de 13 clasificadores clásicos (van der Laan, Polley y Hubbard, 2007).
  #chico[Con 25 semillas, supera a TabPFN en `helices_12` y `hueveras_12` y no es significativamente peor en `circulos`, `eslabones`, `iris` y `pinguinos`.]

== Lo que da conocer la variedad

#grid(columns: (55%, 45%), column-gutter: 1em,
  tabla-csv("data/defensa/oraculo.csv", alineacion: (left, right, right, right, right), tamano: 14pt),
  chico[
    El *oráculo* invierte el proceso generador conocido de cada dataset sintético (radio, fase de la espiral, coordenadas del pionono) y ajusta una regresión logística sobre esas coordenadas.

    #v(0.4em)
    Conocer la variedad vale mucho en `hueveras` y en las variantes con ruido. En `lunas_lo`, #fkdc supera al oráculo: la inversión es solo aproximada con ese nivel de ruido.

    #fuente[Mediana de $R^2$ crudo, 25 semillas.]
  ],
)

== Para seguir

- *Localización*: según Nagler (2023), el sesgo de un PFN solo se anula si la predicción se localiza alrededor del punto de prueba, y la arquitectura actual no lo garantiza. Los núcleos localizan por construcción.
- *Fermat con relevancia*: combinar la distancia de Fermat con anchos por variable.
- *Teoría*: la consistencia del núcleo evaluado sobre la distancia de Fermat muestral sigue abierta.

#focus-slide[¡Gracias!]

= Respaldo

== Fronteras de decisión en `espirales_lo`

#grid(columns: 4, column-gutter: 0.5em,
  image("img/espirales_lo-kn-decision_boundary.svg"),
  image("img/espirales_lo-fkn-decision_boundary.svg"),
  image("img/espirales_lo-kdc-decision_boundary.svg"),
  image("img/espirales_lo-fkdc-decision_boundary.svg"),
)

== Método de las comparaciones por dataset

#chico[
- Unidad: la semilla. En los sintéticos y en `mnist`, cada semilla es una muestra nueva; en los de datos fijos, otra partición 50/50 de los mismos datos, con p-valores optimistas.
- Wilcoxon de rangos con signo, bilateral, sobre las diferencias por semilla; ceros por el método de Pratt; $|Delta| < 10^(-9)$ es empate.
- Holm dentro de cada dataset: 21 pares entre los 7 clasificadores suaves de la tesis, 190 en el conjunto ampliado.
- Hodges–Lehmann con su intervalo del 95 %; Friedman como prueba ómnibus; prueba bayesiana de rangos con signo con equivalencia práctica de $plus.minus 0.01$ en $R^2$.
- $R^2$ crudo con crudo y templado con templado; las comparaciones mixtas solo descompuestas en sus partes.
]

== Modelos de texto: al azar

#chico[
- Laya, GLiNER2.5-Decide y una réplica abierta de Jev reciben cada fila como texto, con el nombre del dataset, su descripción o el código que lo genera.
- En todos los regímenes y datasets quedaron en el nivel del azar; ajustado sobre el entrenamiento, GLiNER alcanza la exactitud de `LR` solo en `lunas`.
- Son modelos de un solo paso, sin razonamiento: decidir de qué círculo viene $(x_1, x_2)$ exige calcular $sqrt(x_1^2 + x_2^2)$ en una pasada.
]
