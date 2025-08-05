#import "@preview/ctheorems:1.1.3": *

// ################
// # definiciones #
// ################
#let phi = math.phi.alt
#let ind = $ op(bb(1)) $
#let sop = $ op("sop") $
#let Pr = $ op("Pr") $
#let bu(x) = $bold(upright(#x))$

// Copetes flexibles para outline y texto, adaptado para 0.12 de
// https://github.com/typst/typst/issues/1295#issuecomment-1853762154
#let in-outline = state("in-outline", false)
#let flex-caption(long, short) = (
  context if in-outline.get() {
    short
  } else {
    long
  }
)

#let defn = thmbox("definition", "Definición", inset: (x: 1.2em, top: 1em))
#let obs = thmplain("observation", "Observación").with(numbering: none)

// ##############
// ### estilo ###
// ##############

// Basado en How do I get the "LaTeX look?",
// de https://typst.app/docs/guides/guide-for-latex-users/
#set page(margin: 1.75in, numbering: "1 de 1")
#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern", lang: "es")
#set heading(numbering: "1.1")
#set strong(delta: 100)
#set par(justify: true)
#set math.equation(numbering: "(1)")
#set quote(block: true)

#show: thmrules
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)
#show link: it =>  underline(text(it, fill: blue))

// ### TOC y listados
#outline(depth: 2)

= TODOs
- [ ] Ponderar por $n^beta$
- [ ] Evitar coma entre sujeto y predicado
- [ ] Mensionar Rodríguez x2:
  - @forzaniPenalizationMethodEstimate2022
  - @henryKernelDensityEstimation2009
- Algo de densidad de volumen:
  - @berenfeldDensityEstimationUnknown2021
  - @bickelLocalPolynomialRegression2007

- @wandKernelSmoothing1995
= Sandbox

Dado el soporte del núcelo $sop K$ para la probabilidad $Pr(a + b/(c/d + 1))$

$vec(1, 2, 3)$
#figure(
  ```python
  print("hello world")
  print("iolii mundo")
  ```,
  caption: "Un cacho e' código",
)
#defn("La mar en coche")[
  A natural number is called a #highlight[_prime number_] if it is greater
  than 1 and cannot be written as the product of two smaller natural numbers.
] <la-mar-en-coche>

#obs[O sea, sería tremendo hundirse en medio de la mar]

Como se explica en @la-mar-en-coche, es muy arriesgado cruzar el océano en un auto. O sea, no #footnote[Al final qué se io, _kcyo_].

= Notación

#set terms(separator: h(2em, weak: true), spacing: 1em)

/ $RR$: los números reales
/ $d_x$: 
/ $RR^(d_x)$:
/ $[k]$:  el conjunto de los k números enteros, ${1, dots, k}$
/ $cal(M)$:
/ $bold(upright(H))$:
/ $norm(dot)$:
/ ${bold(upright(X))}$:
/ $X_(i, j)$:
/ $ind(dot)$: la función indicadora
/ $Pr(dot)$: función de probabilidad
/ vectores: ???
/ escalares: ???

= Preliminares

== El problema de clasificación

=== Definición y vocabulario
@hastieElementsStatisticalLearning2009[§2.2]
El _aprendizaje estadístico supervisado_ busca estimar (aprender) una variable _respuesta_ a partir de cierta(s) variable(s) _predictora(s)_. Cuando la _respuesta_ es una variable _cualitativa_, el problema de asignar cada observación $x$ a una clase $G in cal(G)={g^1, dots, g^K}$ se denomina _de clasificación_.

Un _clasificador_ es una función $hat(G)(x)$ que para cada observación $x$, intenta aproximar su verdadera clase $g$ por $hat(g)$ ("ge sombrero").

Para construir $hat(G)$, contamos con un _conjunto de entrenamiento_ de pares $(x_i, g_i), i in {1, dots, N}$ conocidos. Típicamente, las clases serán MECE, y las observaciones $X in RR^p$.

=== Clsasificador de Bayes

Una posible estrategia de clasificación consiste en asignarle a cada observación $x_0$, la clase más probable en ese punto, dada la información disponible. 

$
hat(G)(x) = arg max_(g in cal(G)) Pr(G=g|X=x)
$

Esta razonable solución es conocida como el _clasificador de Bayes_, y se puede reescribir usando la regla homónima como

$
hat(G)(x) = g_i <=> Pr(g_i|X=x) &= max_(g in cal(G)) Pr(G=g|X=x) \
&=max_(g in cal(G)) Pr(X=x|G=g) times Pr(G=g)
$

=== Clasificadores "suaves" y "duros"

- Un clasificador que responda "¿_qué clase_ es la que más probablemente contenga esta observación" es un clasificador "duro".
- Un clasificador que además puede responder "¿_cuán probable_ es que esta observación pertenezca a cada clase $g_j$?" es un clasificador "suave".
- La regla de Bayes para clasificación nos puede dar un clasificador duro al maximizar la probabilidad; más aún, también puede construir un clasificador suave:

$
hat(Pr)(G=g_i|X=x) &= (hat(Pr)(x|G=g_i) times hat(Pr)(G=g_i)) / (hat(Pr)(X=x)) \
&= (hat(Pr)(x|G=g_i) times hat(Pr)(G=g_i)) / (sum_(k in [K]) hat(Pr)(X=x, G=g_k)) \
$

== Estimación de Densidad por Núcleos
=== Clasificador de Bayes empírico
- Si el conjunto de entrenamiento ${(x_1, g_1), dots, (x_N, g_N)}$ proviene de un muestreo aleatorio uniforme, las probablidades de clase $pi_i = Pr(G=g^((i)))$ se pueden aproximar razonablemente por las proporciones muestrales $ hat(pi)_i = \#{g_j :g_j = g^((i))}slash N$ 

- Resta hallar una aproximación $Pr(x|G=g)$ para cada clase, ya sea a través de una función de densidad, de distribución, u otra manera.

=== Estimación unidimensional

[ESL §6.6, Parzen 1962]


Para fijar ideas, asumamos que $X in RR$ y consideremos la estimación de densidad en una única clase para la que contamos con $N$ ejemplos ${x_1, dots, x_N}$. Una aproximación $hat(f)$ directa sería
(1) $
  hat(f)(x_0) = \#{x_i in cal(N)(x_0)} / (N times h)
$ #label("eps-nn")

donde $cal(N)$ es un vecindario métrico de $x_0$ de diámetro $h$. Esta estimación es irregular, con saltos discretos en el numerador, por lo que se prefiere el estimador suavizado por núcleos de Parzen-Rosenblatt

$
  hat(f)(x_0) = 1/N sum_(i=1)^N K (x_0, x_i)
$ #label("parzen")

=== Función núcleo o "_kernel_"

Se dice que $K(x) : RR-> RR$ es una _función núcleo_ si

- toma valores reales no negativos: $K(u) >= 0 forall u in "sop"K$,
- está normalizada: $integral_(-oo)^(+oo) K(u) d u = 1$,
- es simétrica: $K(u) = K(-u)$ y
- alcanza su máximo en el centro: $max_u K(u) = K(0)$

Observación 1: Todas las funciones de densidad simétricas centradas en 0 son núcleos; en particular, la densidad "normal estándar" $phi(x) = 1/sqrt(2 pi) exp(-x^2 / 2 )$ lo es.

Observación 2: Si $K(u)$ es un núcleo, entonces $K_h (u) = 1/h op(K)(u / h)$ también lo es.

Observación 3: Si $ind(dot)$ es la función indicadora, resulta que $op(U_h)(x) = 1/h ind(-h/2 < x < h/2)$ es un núcleo válido, y el estimador de @parzen con núcleo $U_h$ devuelve el estimador @eps-nn

=== Núcleo uniforme


#image("img/unif-gaus-kern.png")

=== Clasificador de densidad por núcleos
[ESL §6.6.2]

Si $hat(f)_k, k in 1, dots, K$ son estimadores de densidad por núcleos #footnote[KDEs ó _Kernel Density Estimators_, por sus siglas en inglés] según @parzen, la regla de Bayes nos provee un clasificador suave
$
hat(Pr)(G=g_i|X=x) &= (hat(Pr)(x|G=g_i) times hat(Pr)(G=g_i)) / (hat(Pr)(X=x)) \
&=(hat(pi)_i hat(f)_i (x)  )/ (sum_(k=1)^K hat(pi)_k hat(f)_k (x)) \
$

=== Interludio: Naive Bayes
[ESL §6.6.3]

¿Y si las $X$ son multivariadas ($X in RR^d, d>= 2$)? ¿Se puede adaptar el clasificador?

Sí, pero es complejo. Un camino sencillo: asumir que condicional a cada clase $G=j$, los predictores $X_1, X_2, dots, X_p$ se distribuyen indenpendientemente entre sí.

$
  f_j (X) = product_(i=1)^p f_(j,i) (X_i)
$

Cada densidad marginal $ f_(j,i)$ condicional a la clase se puede estimar usando KDE univariado, y hasta se puede aplicar - usando histogramas - cuando algunas componentes $X_i$ son discretas.

A este procedimiento, se lo conoce cono "Naive Bayes".

=== KDE multivariado
[Wand & Jones 1995 §4]

En su forma más general, estimador de densidad por núcleos $d$-variado es 

$
   hat(f) (x; bu(H)) = N^(-1) sum_(i=1)^N K_bu(H)(x - x_i)
$

donde 
- $bu(H) in RR^(d times d)$ es una matriz simétrica def. pos. análoga a la ventana $h in RR$ para $d=1$,
-  $K_bu(H)(t) = abs(det bu(H))^(-1/2) K(bu(H)^(-1/2) t)$
- $K$ es una función núcleo $d$-variada tal que $integral K(bu(x)) d bu(x) = 1$   
Típicamente, K es la densidad normal multivariada
$
 Phi(x) : RR^d -> RR = (2 pi)^(-d/2) exp(- (||x||^2)/2)
$

=== Dificultades: elección de $bu(H)$
Sean las clases de matrices pertenecientes a $RR^(d times d)$ ...
- $cal(F)$, de matrices simétricas definidas positivas,
- $cal(D)$, de matrices diagonales definidas positivas ($cal(D) subset.eq cal(F)$) y
- $cal(S)$, de múltiplos escalares de la identidad: $cal(S) = {h^2 bu(I):h >0} subset.eq cal(D)$
 
Aún tomando una única $bu(H)$ para _toda_ la muestra, $bu(H) in dots$
- $cal(F)$, requiere definir $mat(d; 2) = d(d-1)/2$ parámetros de ventana,
- $cal(D)$ requiere $d$ parámetros, y
- $cal(S)$ tiene un único parámetro $h$.

 A priori no es posible saber qué parametrización conviene, pero en general $bu(H) in cal(D)$ parece un compromiso razonable: no se pierde demasiado contra $cal(F)$, pero tampoco se padece la "rigidez" de $bu(H) in cal(S)$.

=== Dificultades: La maldición de la dimensionalidad

[ESL §2.5, Wand & Jones 1995 §4.9 ej 4.1]

Sean $X_i tilde.op^("iid")"Uniforme"([-1, 1]^d), i in {1, dots, N}$, y consideremos la estimación de la densidad en el origen, $f(bu(0))$. Suponga que el núcleo $K_(bu(H))$ es un "núcleo producto" basado en la distribución univariada $"Uniforme(-1, 1)"$, y $bu(H) = h^2 bu(I)$. Derive una expresión para la proporción esperada de puntos incluidos dentro del soporte del núcleo $K_bu(H)$ para $h, d$. arbitrarios.

(... interludio de pizarrón ...)

$
  Pr(X in [-h, h]^d) &=  Pr(inter_(i=1)^d abs(X_i) <= h) \
  Pr(X in [-0.95, 0.95]^50) &approx 0.0077 \
$

=== Dificultades: La maldición de la dimensionalidad

#image("img/curse-dim.png")
Para $h <=0.5, Pr(dot) < 1 times 10^(-15)$. Aún para $h=0.95, Pr(dot) approx 0.0077$ #emoji.face.shock

== Clasificación en variedades

=== La hipótesis de la variedad ("manifold hypothesis")
[Bengio Repr learning]
[#link("https://www.reddit.com/r/MachineLearning/comments/mzjshl/d_who_first_advanced_the_manifold_hypothesis_to/")[Bengio en Reddit]
]

La hipótesis de la variedad postula que los datos $X in RR^(d_X)$ muestreados soportados en un espacio de alta dimensionalidad #footnote[E.g.: imágenes, audio, video, secuencias de nucleótidos]. tenderán a concentarse sobre una _variedad_ $cal(M)$, potencialmente de mucha menor dimensión $d_(cal(M)) << d_X$, embebida en el espacio original $cal(M) subset.eq RR^(d_X)$.

- Well suited for AI tasks such as those involving images, sounds or text, for which most uniformly sampled input configurations are unlike natural stimuli.
- archetypal manifold modeling algorithm is, not surprisingly, also the archetypal low dimensional representation learning algorithm: Principal Component Analysis, which models a linear manifold.
- Data manifold for complex real world domains are however expected to be strongly nonlinear.


=== IRL

#columns(2,[
  #image("img/hormiga-petalo.jpg", height: 70%)
  #colbreak()
  #image("img/bandera-argentina.png")])


Pero: ¿en qué variedad vive un dígito, o su trazo, o una canción? #emoji.cigarette

=== Interludio: Variedades de Riemann [Wikipedia]

#quote[Una variedad $d$-dimensional $cal(M)$ es un espacio _topológico_ tal que cada punto $p in cal(M)$ tiene un vecindario $U$ que resulta _homeomórfico_ a un conjunto abierto en $RR^d$]

- topológico: se puede definir cercanía (pero no necesariamente distancia), permite definir funciones continuas y límites
- homeomórfico a $RR^d$: para cada punto $p in cal(M)$, existe un mapa _biyectivo_ y _suave_ entre el vecindario de $p$ y $RR^d$. El conjunto de tales mapas se denomina _atlas_.

#grid(columns: (80%, 20%), [
  Sea $T_p cal(M)$ el _espacio tangente_ a un punto $p in cal(M)$, y $g_p : T_p cal(M) times T_p cal(M) -> RR$ una forma _bilinear pos. def._ para cada $p$ que induce una _norma_ $||v||_p= sqrt(g_p(v, v))$. 
  
  Decimos entonces que $g_p$ es una métrica Riemanniana y el par $(cal(M), g)$ es una variedad de Riemann, donde las nociones de _distancia, ángulo y geodésica_ están bien definidas.], image("img/Tangent_plane_to_sphere_with_vectors.svg",)
)

=== KDE en variedades de Riemann [Pelletier 2005]
- Sea $(cal(M), g)$ una variedad de Riemann compacta y sin frontera de dimensión $d$, y usemos $d_g$ para denotar la distancia de Riemann.
- Sea $K$ un _núcleo isotrópico en $cal(M)$ soportado en la bola unitaria_ (cf. conds. (i)-(v))
- Sean $p, q in cal(M)$, y $theta_p (q)$ la _función de densidad de volumen en $cal(M)$_ #footnote[¡Ardua definición! Algo así como el cociente entre las medida de volumen en $cal(M)$, y su transformación via el mapa local a $RR^d$]
Luego, el estimador de densidad para $X_i tilde.op^("iid")f$ es $f_(N,K):cal(M) ->RR$ que a cada $p in cal(M)$ le asocia el valor
$
  f_(N,K) (p) = N^(-1) sum_(i=1)^N K_h (p,X_i) = N^(-1) sum_(i=1)^N 1/h^d 1/(theta_X_i (p))K((op(d_g)(p, X_i))/h)
$

con la restricción de que la ventana $h <= h_0 <= op("inj")(cal(M))$, el _radio de inyectividad_ de $cal(M)$ #footnote[el ínfimo entre el supremo del radio de una bola en cada $p$ tal que su mapa es un difeomorfismo]

=== Interludio: densidad de volumen en la esfera [Henry y Rodríguez, 2009]

#columns(2)[
  En _"Kernel Density Estimation on Riemannian Manifolds: Asymptotic Results" (2009)_, Guillermo Henry y Daniela Rodriguez estudian algunas propiedades asintótica de este estimador, y las ejemplifican con datos de sitios volcánicos en la superficie terrestre.
  En particular, calculan la densidad de volumen $theta_p(q)$
#image("img/densidad-volumen-esfera.png")
#colbreak()
#image("img/henry-rodriguez-bolas.png")
]

=== Clasificación en variedades [Loubes y Pelletier 2008]

¡Clasificador de Bayes + KDE en Variedades = Clasificación (suave o dura) en variedades!

Plantean una regla de clasificación $hat(G)$ para 2 clases adaptable a K clases de forma directa. Sea $p in cal(M)$ una variedad riemanniana como antes, y ${(x_1, g_1), dots, (x_N, g_N)}$ nuestras observaciones y sus clases. Luego,

$
  hat(G) (p) = arg max_(g in cal(G)) sum_(i=1)^N ind(g_i = g)K_h (p,X_i)
$

 
#align(center)[Pero... ¿y si la variedad es desconocida?]

== Aprendizaje de distancias

=== El ejemplo canónica: Análisis de Componentes Principales (PCA)

#align(center)[#image("img/pca.png", height:90%)]
#text(size: 12pt)[Karl Pearson (1901), _"LIII. On lines and planes of closest fit to systems of points in space."_]


=== El algoritmo más _cool_: Isomap
#grid(columns: (35%, 65%), column-gutter:20pt, [
  1. Construya el grafo de $k, epsilon$-vecinos, $bu(N N)=(bu(X), E)$

  2. Compute los caminos mínimos - las geodésicas entre observaciones, $d_(bu(N N))(x, y)$.

  3. Construya una representación ("_embedding"_) $d^*$−dimensional que minimice la discrepancia ("stress") entre $d_(bu(N N))$ y la distancia euclídea en $RR^d^*$
],image("img/isomap-2.png", height:90%))
[Tenenbaum et al (2000), _"A Global Geometric Framework for Nonlinear Dimensionality Reduction"_]

=== Distancia de Fermat [Groisman, Jonckheere, Sapienza (2019); Little et al (2021)]

#quote(attribution: "P. Groisman et al (2019)")[
  #set text(size: 12pt)
_We tackle the problem of learning a distance between points, able to capture both the geometry of the manifold and the underlying density. We define such a sample distance and prove the convergence, as the sample size goes to infinity, to a macroscopic one that we call Fermat distance as it minimizes a path functional, resembling Fermat principle in optics._]

Sea $f$ una función continua y positiva, $beta >=0$
 y $x, y in S subset.eq RR^d$. Definimos la _Distancia de Fermat_ $cal(D)_(f, beta)(x, y)$ como:

$
cal(T)_(f, beta)(gamma) = integral_gamma f^(-beta) space, quad quad quad cal(D)_(f, beta)(x, y) = inf_(gamma in Gamma) cal(T)_(f, beta)(gamma)  quad #emoji.face.shock 
$

... donde el ínfimo se toma sobre el conjunto $Gamma$ de todos los caminos rectificables entre $x$ e $y$ contenidos en $overline(S)$, la clausura de $S$, y la integral es entendida con respecto a la longitud de arco dada por la distancia euclídea.
 
=== Distancia de Fermat muestral

Para $alpha >=1$ y $x, y in RR^d$, la _Distancia Muestral de Fermat_ se define como

$
D_(bu(X), alpha) = inf {sum_(j=1)^(K-1) ||q_(j+1) - q_j||^alpha : (q_1, dots, q_K) "es un camino de de x a y", K>=1}
$

donde los $q_j$ son elementos de la muestra $bu(X)$. Nótese que $D_(bu(X), alpha)$ satisface la desigualdad triangular, define una métrica sobre $bu(X)$ y una pseudo-métrica sobre $RR^d$.

En su paper, Groisman et al. muestran que 
$
  lim_(N -> oo) n^beta D_(bu(X)_n, alpha) (x, y)= mu cal(D)_(f, beta)(x, y)
$
donde $beta = (a-1) slash d, thick n >= n_0 $ y $mu$ es una constante adecuada. 


¡Esta sí la podemos aprender de los datos! #emoji.arm.muscle

== Todo junto:
Clasificación en variedades desconocidas por estimación de densidad por núcleos con Distancia de Fermat Muestral

=== Algunas dudas

- Entrenar el clasificador por validación cruzada está OK: como $bu(X)_"train" subset.eq bu(X)$ y $bu(X)_"test" subset.eq bu(X)$, se sigue que $forall (a, b) in {bu(X)_"train" times in bu(X)_"test"} subset.eq {bu(X) times bu(X)}$ y $D_(bu(X), alpha) (a, b)$ está bien definida.  ¿Cómo sé la distancia _muestral_ de una _nueva_ observación $x_0$, a los elementos de cada clase?\


Para cada una de las $g_i in cal(G)$ clases, definimos el conjunto $
Q_i= {x_0} union {x_j : x_j in bu(X), g_j = g_i, j in {1, dots, N}}
$
y calculamos $D_(Q_i, alpha) (x_0, dot)$

=== Algunas dudas

- El clasificador de Loubes & Pelletier asume que todas las clases están soportadas en la misma variedad $cal(M)$. ¿Quién dice que ello vale para las diferentes clases?


¡Nadie! Pero
1. No hace falta dicho supuesto, y en el peor de los casos, podemos asumir que la unión de las clases está soportada en _cierta_ variedad de Riemman, que resulta de (¿la clausura de?) la unión de sus soportes individuales. 
2. Sí es cierto que si las variedades (y las densidades que soportan) difieren, tanto el $alpha_i^*$ como el $h_i*$ "óptimos" para los estimadores de densidad individuales no tienen por qué coincidir. 
3. Aunque las densidades individuales $f_i$ estén bien estimadas, el clasificador resultante puede ser mal(ard)o si no diferencia bien "en las fronteras". Por simplicidad, además, decidimos parametrizar el clasificador con dos únicos hiperparámetros globales: $alpha, h$.

=== Diseño experimental

1. Desarrollamos un clasificador compatible con el _framework_ de #link("https://arxiv.org/abs/1309.0238", `scikit-learn`)  según los lineamientos de Loubes & Pelleteir, que apodamos `KDC`. 
2. Implementamos el estimador de la distancia muestral de Fermat, y combinándolo con KDC, obtenemos la titular "Clasificación por KDE con Distancia de Fermat", `FKDC`. 
3.  Evaluamos el _pseudo-$R^2$_ y la _exactitud_ ("accuracy") de los clasificadores propuestos en diferentes _datasets_, relativa a técnicas bien establecidas: 
#columns(2)[
- regresión logística (`LR`)
- clasificador de  soporte vectorial (`SVC`) #footnote[sólo se consideró su exactitud. ya que no es un clasificador suave]
- _gradient boosting trees_ (`GBT`)
#colbreak()
- k-vecinos-más-cercanos (`KN`)
- Naive Bayes Gaussiano (`GNB`)]

#pagebreak()

- La implementación de `KNeighbors` de referencia acepta distancias precomputadas, así que incluimos una versión con distancia de Fermat, que apodamos `F(ermat)KN`. 

- Para ser "justos", se reservó una porción de los datos para la evaluación comparada, y del resto, cada algoritmo fue entrenado repetidas veces por validación cruzada de 5 pliegos, en una extensísima grilla de hiperparametrizaciones. Este procedimiento *se repitió 25 veces en cada dataset*. 

- La función de score elegida fue `neg_log_loss` ($ = cal(l)$) para los clasificadores suaves, y `accuracy` para los duros.

#pagebreak()
- Para tener una idea "sistémica" de la performance de los algoritmos, evaluaremos su performance con _datasets_ que varíen en el tamaño muestral $N$, la dimensión $p$ de las $X_i$, el nro. de clases $k$ y su origen ("real" o "sintético"). 

- Cuando creamos datos sintéticos en variedades  con dimensión intrínseca menor a la ambiente, (casi) cualquier clasificador competente alcanza exactitud perfecta; para complejizar la tarea, agegamos un poco de "ruido" a las observaciones, y también analizamos sus efectos.

=== Regla de Parsimonia

- ¿Qué parametrización elegir cuando "en test da todo igual"? 

#align(center)[ #emoji.knife de Occam: la más "sencilla" (TBD)]


- ¿Qué parametrización elegir cuando "en test da *casi* todo igual"? 


#align(center)[*Regla de $1sigma$*: De las que estén a $1sigma$ de la mejor, la más sencilla.]



¿Sabemos cuánto vale $sigma$?

=== $R^2$ de McFadden
Sea $cal(C)_0$ el clasificador "base", que asigna a cada observación y posible clase, la frecuencia empírica de clase encontrada en la muestra $bu(X)$. Para todo clasificador suave $cal(C)$, definimos el $R^2$ de McFadden como
  $ op(R^2)(cal(C) | bu(X)) = 1 - (op(cal(l))(cal(C))) / (op(cal(l))(cal(C)_0)) $


donde $cal(l)(dot)$ es la log-verosimilitud clásica. Nótese que $op(R^2)(cal(C)_0) = 0$.  A su vez, para un clasificador perfecto $cal(C)^star$ que otorgue toda la masa de probabilidad a la clase correcta, tendrá $op(L)(cal(C)^star) = 1$ y log-verosimilitud igual a 0, de manera que $op(R^2)(cal(C)^star) = 1 - 0 = 1$.


Sin embargo, un clasificador _peor_ que $cal(C)_0$ en tanto asigne bajas probabilidades ($approx 0$) a las clases correctas, puede tener un $R^2$ infinitamente negativo.

= Resultados
=== 2D, 2 clases: excelente $R^2$ con exactitud competitiva

=== Con Bajo Ruido
#align(center)[#image("img/2d-lo-datasets.png")]
#pagebreak()
#columns(3)[
  #image("img/lunas_lo-overall.png")
  #colbreak()
  #image("img/circulos_lo-overall.png")
  #colbreak()
  #image("img/espirales_lo-overall.png")
  
]
#pagebreak()
=== Boxplot Accuracy
#align(center)[#image("img/2d-lo-acc.png")]
#pagebreak()
=== Boxplot $R^2$
#align(center)[#image("img/2d-lo-r2.png")]

=== Superposición de parámetros: $alpha$ y $h$


- El uso de la distancia de Fermat muestral no hiere la performance, pero las mejoras son nulas o marginales. ¿Por qué?


Si recordamos $hat(f)_(K,N)$ según Loubes & Pelletier, al núcleo $K$ se lo evalúa sobre 
$
 (d (x_0, X_i)) / h, quad d = D_(Q_i, alpha)
$

Lo que $alpha$ afecta a $hat(f)$ vía $d$, también se puede conseguir vía $h$.

Si $D_(Q_i, alpha) prop ||dot|| $ (la distancia de fermat es proporcional a la euclídea), los efectos de $alpha$ y $h$ se "solapan" 


... y sabemos que localmente, eso es cierto #emoji.face.tear

=== Parámetros óptimos para $"(F)KDC"$ en `espirales_lo`
#align(center)[#image("img/optimos-espirales_lo.png", height: 80%)]


=== Superficies (o paisajes) de _score_ para `(espirales_lo, 1434)`

#align(center)[#image("img/heatmap-fkdc-2d-lo-new.svg", height: 110%)]

=== Alt-viz: Perfiles de pérdida para `(espirales_lo, 1434)`

#align(center)[#image("img/perfiles-perdida-espirales-1434.png", height: 110%)]

=== Fronteras de decisión para `(espirales_lo, 1434)`

#align(center)[#image("img/gbt-lr-espirales.png")]
#pagebreak()
#align(center)[#image("img/kn-espirales.png")]
#pagebreak()
#align(center)[#image("img/kdc-espirales.png")]
#pagebreak()
#align(center)[#image("img/gnb-svc-espirales.png")]



=== 3D, 2 clases + piononos

#align(center)[#image("img/3d.png")]
#pagebreak()
#align(center)[#image("img/pionono.png", height: 110%)]
#pagebreak()
#columns(4)[
  #image("img/pionono_0-overall.png")
  #colbreak()
  #image("img/eslabones_0-overall.png")
  #colbreak()
  #image("img/helices_0-overall.png")
  #colbreak()
  #image("img/hueveras_0-overall.png")  
]
#pagebreak()
#align(center)[#image("img/pionono-eslabones-r2.png")]
#pagebreak()
#align(center)[#image("img/helices-hueveras-r2.png")]

=== Parámetros óptimos para $"(F)KDC"$ en `helices_0`
#align(center)[#image("img/optimos-helices_0.png", height: 100%)]

=== Microindiferencia, macrodiferencia

- En zonas con muchas observaciones (por tener alta $f$ o alto $N$) sampleadas, la distancia de Fermat y la euclídea coinciden. 
- "Localmente", siempre van a coincidir, aunque sea en un vecindario muy pequeño. 
- Si el algoritmo de clasificación sólo depende de ese vencindario local para clasificar, no hay ganancia en la distancia de Fermat. 
- ¡Pero tampoco hay pérdida si se elige mal `n_neighbors`! #emoji.person.shrug


=== $R^2$ por semilla para $"(F)KN"$ en `helices_0`
#align(center)[#image("img/r2-fkn-kn-helices_0.png", height: 100%)]

=== $R^2$ y $alpha^star$ para $"(F)KN"$ en `helices_0`, `n_neighbors` seleccionados
#align(center)[#image("img/r2-fkn-kn-n_neighbors-seleccionados.png", height: 65%)]

=== Mejor $R^2$ para $"(F)KN"$ en `helices_0`, en función de `n_neighbors`

#image("img/helices_0-fkn_kn-mean_test_score.png")


=== $R^2$ por semilla para $"(F)KN"$ en `eslabones_0`
#align(center)[#image("img/outputa.png", height: 100%)]

=== $R^2$ y $alpha^star$ para $"(F)KN"$ en `eslabones_0`, `n_neighbors` seleccionados
#align(center)[#image("img/Screenshot 2025-07-18 at 11.43.27 AM.png", height: 65%)]

=== Mejor $R^2$ para $"(F)KN"$ en `eslabones_0`, en función de `n_neighbors`

#image("img/outputb.png")

=== Otros datasets: 2D mucho ruido
#columns(3)[
  #image("img/lunas_hi-overall.png")
  #colbreak()
  #image("img/circulos_hi-overall.png")
  #colbreak()
  #image("img/espirales_hi-overall.png")
]
=== Otros datasets: 15D
#columns(4)[
  #image("img/pionono_12-overall.png")
  #colbreak()
  #image("img/eslabones_12-overall.png")
  #colbreak()
  #image("img/helices_12-overall.png")
  #colbreak()
  #image("img/hueveras_12-overall.png")  
]
=== Otros datasets: multiclase
#columns(4)[
  #image("img/iris-overall.png")
  #colbreak()
  #image("img/vino-overall.png")
  #colbreak()
  #image("img/pinguinos-overall.png")
  #colbreak()
  #image("img/anteojos.png")  
]
=== Otros datasets: `digitos` y `mnist`

#columns(2)[
  #image("img/digitos-overall.png")
  #colbreak()
  #image("img/mnist-overall.png")
]

=== El problema de clasificación


#defn("problema de clasificación")[] <clf-prob>
#defn("clasificador vecino más cercano")[] <nn-clf>

=== Definición del problema unidimensional
[muestra aleatoria]Sea

#let dimx = $d_x$
Consideremos el problema de clasificación:
[problema de clasificación]<def:prob-clf-1> Sea $X={ X_{1},dots,X_{N}} , X_{i} in R^{"dimx"} forall i in [N]$
una muestra de $N$ observaciones aleatorias $dimx-$dimensionales,
repartidas en $M$ clases $C_{1},dots,C_{M}$ mutuamente excluyentes
y conjuntamente exhaustivasfootnote{es decir, $forall i in[N]equiv{ 1,dots,N} ,X_{i}in C_{j} arrow.l.r.double<=> X_{i} not in C_{k},k in mu,k =/= j$.
Asumamos además que la muestra está compuesta de observaciones independientes
entre sí, y las observaciones de cada clase están idénticamente distribuidas
según su propia ley: si $|C_{j}| =N_{j}$ y $X_{i}^{(j)}$representa
la i-ésima observación de la clase $j$, resulta que $X_{i}^{(j)} {L}_{j}(X) forall j in mu,i in [N_{j}]$.

Dada una nueva observación $x_{0}$ cuy


=== Clasificadores "duros" y "suaves"
#defn("clasificación dura")[¿a qué clase deberíamos asignarla? ] <clf-dura>
#defn("clasificación suave")[ ¿qué probabilidad tiene de pertenecer a cada
  clase $C_{j},j in [M]$ ?] <clf-suave>

Cualquier método o algoritmo que pretenda responder el problema de
clasificación, prescribe un modo u otro de combinar toda la información
muestral disponible, ponderando las $N$ observaciones relativamente
a su cercanía o similitud con $x_{0}$. Por caso, $k-$vecinos más
cercanos ($k-$NN) asignará la nueva observación $x_{0}$ a la clase
modal - la más frecuente - entre las $k$ observaciones de entrenamiento
más cercanasemph{ }en distancia euclídea $norm{x_{0} - .}$. $k-$NN
no menciona explícitamente las leyes de clase #math.cal("L")},
lo cual lo mantiene sencillo a costa de ignorar la estructura del
problema.
#defn("k-NN")[k-Nearest Neighbors] <knn>
=== Clasificador de Bayes

=== KDE: Estimación de la densidad por núcleos
#defn("KDE")[] <kde>

=== La maldición de la (alta) dimensionalidad

=== NB: El clasificador "ingenuo" de Bayes
#defn("Naïve Bayes")[] <gnb>

==== Una muestra adversa

=== KDC Multivariado
#defn("KDE Multivariado")[] <kde-mv>
#defn("KDC")[] <kdc>

==== El caso 2-dimensional

==== Relación entre H y la distancia de Mahalanobis

=== La hipótesis de la variedad

#figure(
  caption: flex-caption[La variedad $cal(U)$ con $dim(cal(U)) = 1$ embebida en $RR^2$. Nótese que en el espacio ambiente, el punto rojo está más cerca del verde, mientras que a través de $cal(U)$, el punto amarillo está más próximo que el rojo][Variedad $cal(U)$],
)[#image("img/variedad-u.svg", width: 70%)]
=== KDE en variedades de Riemann

=== Variedades desconocidas

=== Aprendizaje de distancias

=== Isomap

=== Distancias basadas en densidad

=== Distancia de Fermat
- Groisman & Jonckheere @groismanNonhomogeneousEuclideanFirstpassage2019
- Little & Mackenzie @littleBalancingGeometryDensity2021
- Bijral @bijralSemisupervisedLearningDensity2012
- Vincent & Bengio @vincentDensitySensitiveMetrics2003
#defn("Distancia Muestral de Fermat")[]<sample-fermat-distance>
== Propuesta Original

Habiendo andado este sendero teórico, la pregunta natural que asoma es: ¿es posible mejorar un algoritmo de clasificación reemplazando la distancia euclídea por una aprendida de los datos, como la de Fermat? Para investigar la cuestión, nos propusimos:
1. Implementar un clasificador basado en estimación de densidad por núcleos (@kde) según @loubesKernelbasedClassifierRiemannian2008, que llamaremos "KDC". Además,
2. Implementar un estimador de densidad por núcleos basado en la distancia de Fermat, a fines de poder comparar la _performance_ de KDC con distancia euclídea y de Fermat.

Nótese que el clasificador enunciado al inicio (k-NN, @knn), tiene un pariente cercano, $epsilon-upright("NN")$

#defn($epsilon-"NN"$)[] <eps-NN>

@eps-NN es esencialmente equivalente a KDC con un núcleo "rectangular", $k(t) =  ind(d(x, t) < epsilon) / epsilon$, pero su definición es considerablemente más sencilla. Luego, propondremos también
3. Implementar un clasificador cual @knn, pero con distancia muestral de Fermat en lugar de euclídea.

=== KDC con Distancia de Fermat Muestral

=== f-KNN

== Evaluación

Nos interesa conocer en qué circunstancias, si es que hay alguna, la distancia muestral de Fermat provee ventajas a la hora de clasificar por sobre la distancia euclídea. Además, en caso de existir, quisiéramos en la medida de lo posible comprender por qué (o por qué no) es que tal ventaja existe.
A nuestro entender resulta imposible hacer declaraciones demasiado generales al respecto de la capacidad del clasificador: la cantidad de _datasets_ posibles, junto con sus _configuraciones de evaluación_ es tan densamente infinita como lo permita la imaginación del evaluador. Con un ánimo exploratorio, nos proponemos explorar la _performance_ de nuestros clasificadores basados en distancia muestral de Fermat en algunas _tareas_ puntuales.

=== Métricas de _performance_

En tareas de clasificación, la métrica más habitual es la _exactitud_ #footnote([Más conocida por su nombre en inglés, _accuracy_.])

#let clfh = $op(upright(R))$
#let clfs = $op(cal(R))$

#defn("exactitud")[Sean $bu(("X, y")) in RR^(n times p) times RR^n$ una matriz de $n$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(y)) = clfh(bu(X))$ las predicciones de clase resultado de una regla de clasificación #clfh. La _exactitud_ ($"exac"$) de #clfh en #bu("X") se define como la proporción de coincidencias con las clases verdaderas #bu("y"):
  $ op("exac")(clfh | bu(X)) = n^(-1) sum_(i=1)^n ind(hat(y)_i = y_i) $
] <exactitud>

La exactitud está bien definida para cualquier clasificador que provea una regla _dura_ de clasificación, segun @clf-dura. Ahora bien, cuando un clasificador provee una regla _suave_ (@clf-suave), la exactitud como métrica "pierde información": dos clasificadores binarios que asignen respectivamente 0.51 y 1.0 de probabilidad de pertenecer a la clase correcta a todas als observaciones tendrán la misma exactitud, $100%$, aunque el segundo es a las claras mejor. A la inversa, cuando un clasificador erra al asignar la clase: ¿lo hace con absoluta confianza, asignando una alta probabilidad a la clase equivocada, o con cierta incertidumbre, repartiendo la masa de probabilidad entre varias clases que considera factibles?

Una métrica natural para evaluar una regla de clasificación suave, es la _verosimilitud_ (y su logaritmo) de las predicciones.

#defn("verosimilitud")[Sean $bu(("X, y")) in RR^(n times p) times RR^n$ una matriz de $n$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(Y)) = clfs(bu(X)) in RR^(n times k)$ la matriz de probabilidades de clase resultado de una regla suave de clasificación #clfs. La _verosimilitud_ ($"vero"$) de #clfs en #bu("X") se define como la probabilidad conjunta que asigna #clfs a las clases verdaderas #bu("y"):
  $
    op(L)(clfs) = op("vero")(
      clfs | bu(X)
    ) = Pr(hat(bu(y)) = bu(y)) = product_(i=1)^n Pr(hat(y)_i =y_i) = product_(i=1)^n hat(bu(Y))_((i, y_i))
  $

  Por conveniencia, se suele considerar la _log-verosimilitud promedio_,
  $ op(cal(l))(clfs) = n^(-1) log(op("L")(clfs)) = n^(-1)sum_(i=1)^n log(hat(bu(Y))_((i, y_i))) $
] <vero>

// La verosimilitud de una muestra varía en $[0, 1]$ y su log-verosimilitud, en $(-oo, 0]$, pero como métrica esta sólo se vuelve comprensible _relativa a otros clasificadores_. Una forma de "normalizar" la log-verosimilitud, se debe a @mcfaddenConditionalLogitAnalysis1974.

#defn([$R^2$ de McFadden])[Sea $clfs_0$ el clasificador "nulo", que asigna a cada observación y posible clase, la frecuencia empírica de clase encontrada en la muestra de entrenamiento $bu(X)_("train")$. Para todo clasificador suave $clfs$, definimos el $R^2$ de McFadden como
  $ op(R^2)(clfs | bu(X)) = 1 - (op(cal(l))(clfs)) / (op(cal(l))(clfs_0)) $
] <R2-mcf>

#obs[ $op(R^2)(clfs_0) = 0$. A su vez, para un clasificador perfecto $clfs^star$ que otorgue toda la masa de probabilidad a la clase correcta, tendrá $op(L)(clfs^star) = 1$ y log-verosimilitud igual a 0, de manera que $op(R^2)(clfs^star) = 1 - 0 = 1$.

  Sin embargo, un clasificador _peor_ que $clfs_0$ en tanto asigne bajas probabilidades a las clases correctas, puede tener un $R^2$ infinitamente negativo.
]
#let fkdc = $cal(F)"-KDC"$
#let kdc = "KDC"
#let fknn = $cal(F)"-kNN"$
#let knn = $k-"NN"$

Visto y considerando que tanto #fkdc como #fknn son clasificadores suaves, evaluaremos su comportamiento en comparación con ambas métricas, la exactitud y el $R^2$ de McFadden #footnote[de aquí en más, $R^2$ para abreviar]

=== Algoritmos de referencia

Además de medir qué (des)ventajas otorga el uso de una distancia aprendida de los datos en la tarea de clasificación, quisiéramos entender (a) por qué sucede, y (b) si tal (des)ventaja es significativa en el amplio abanico de algoritmos disponibles. Pírrica victoria sería mejorar con la distancia de Fermat la _performance_ de cierto algoritmo, para encontrar que aún con la mejora, el algoritmo no es competitivo en la tarea de referencia.

#let gnb = `GNB` // $("GNB")$
#let lr = `LR`
#let svc = `SVC`

Consideraremos a modo de referencia los siguientes algoritmos:
- Naive Bayes Gaussiano (@gnb, #gnb),
- Regresión Logistica (#lr) y
- Clasificador de Soporte Vectorial (#svc)
Esta elección no pretende ser exhaustiva, sino que responde a un "capricho informado" del investigador. #gnb es una elección natural, ya que es la simplificación que surge de asumir independencia en las dimensiones de ${bu(X)}$ para KDE multivariado (@kde-mv), y se puede computar para grandes conjuntos de datos en muy poco tiempo. #lr es "el" método para clasificación binaria, y su extensión a múltiples clases no es particularmente compleja: para que sea mínimamente valioso un nuevo algoritmo, necesita ser al menos tan bueno como #lr, que tiene ya más de 65 años en el campo (TODO REF bliss1935, cox1958). Por último, fue nuestro deseo incorporar algún método más cercano al estado del arte. A tal fin, consideramos incorporar alguna red neuronal (TODO REF), un método de _boosting_ (TODO REF) y el antedicho clasificador de soporte vectorial, #svc. Finalmente, por la sencillez de su implementación dentro del marco elegido #footnote[Utilizamos _scikit-learn_, un poderoso y extensible paquete para tareas de aprendizaje automático en Python] y por la calidad de los resultados obtenidos, decidimos incorporar #svc, en dos variantes: con núcleos (_kernels_) lineales y RBF.
=== Uno complejo: SVC
#defn("clasificador por sporte vectorial")[]
=== Uno conocido: LR - tal vez?
#defn("regresión logística multinomial")[]

=== Metodología
#let X = ${bu(X)}_n$

La unidad de evaluación de los algoritmos a considerar es una `Tarea`, que se compone de:
- un _diccionario de algoritmos_ a evaluar en condiciones idénticas, definidas por
- un _dataset_ con el conjunto de $n$ observaciones en $d_x$ dimensiones repartidas en $k$ clases, ${bu(X)}_n in R^(n times d_x),  {bold(y)}_n in [k]^n$,
- un _split de evaluación_ $r in (0, 1)$, que determina las proporciones de los datos a usar durante el entrenamiento ($1 - r$) y la evaluación ($r$), junto con
- una _semilla_ $s in [2^32]$ que alimenta el generador de números aleatorios y define determinísticamente cómo realizar la división antedicha.

=== Entrenamiento de los algoritmos
La especificación completa de un clasificador, requiere, además de la elección del algoritmo, la especificación de sus _hiperparámetros_, de manera tal de optimizar su rendimiento bajo ciertas condiciones de evaluación. Para ello, se definió de antemano para cada clasificador una _grilla_ de hiperparámetros: durante el proceso de entrenamiento, la elección de los "mejores" hiperparámetros se efectuó maximizando la exactitud (@exactitud) con una búsqueda exhaustiva por convalidación cruzada de 5 pliegos #footnote[Conocida en inglés como _Grid Search 5-fold Cross-Validation_] sobre la grilla entera.

=== Estimación de la variabilidad en la _performance_ reportada
#let reps = 16
En última instancia, cualquier métrica evaluada, no es otra cosa que un _estadístico_ que representa la "calidad" del clasificador en la Tarea a mano. A fines de conocer no sólo su estimación puntual sino también darnos una idea de la variabilidad de su performance, para cada dataset y colección de algoritmos, se entrenaron y evaluaron #reps Tareas idénticas salvo por la semilla $s$, que luego se usaron para estimar la varianza y el desvío estándar en la exactitud (@exactitud) y el pseudo-$R^2$ (@R2-mcf).

Cuando el conjunto de datos proviene del mundo real y por lo tanto _preexiste a nuestro trabajo_, las #reps semillas $s_1, dots, s_#reps$ fueron utilizadas para definir el split de entrenamiento/evaluación. Por el contrario, cuando el conjunto de datos fue generado sintéticamente, las semillas se utilizaron para generar #reps versiones distintas pero perfectamente replicables del dataset, y en todas se utilizó una misma semilla maestra $s^star$ para definir el split de evaluación.

=== Resultados

=== Chequeo de sanidad: `blobs`
Antes de considerar ningún tipo de sofisticación, comenzamos asegurándonos que en condiciones benignas, nuestros clasificadores funcionan correctamente. La
// #set figure(supplement: "Figura")
#figure(
  image("img/2-blobs.png"),
  caption: flex-caption[`make_blobs(n_features=2, centers=((0, 0), (10, 0)), random_state=1984)`][2 blobs],
) <2-blobs>

En este ejemplo, $d_cal(M) = d_x = 2; thick k=2; thick n_1 = n_2 = 400$ tenemos dos clases perfectamente separables, con lo cual cualquier clasificador razonable debería alcanzar $op("exac") approx 1, thick cal(l) approx 0, R^2 approx 1$. La evaluación de nuestros clasificadores resulta ser:
#let tabla_csv(path) = {
  let data = csv(path)
  let eval_scope = (fkdc: fkdc, kn: knn, fkn: fknn, kdc: kdc, lr: lr, svc: svc, lsvc: `LSVC`, gnb: gnb, base: "base")
  table(columns: data.at(0).len(), ..data.flatten().map(eval.with(mode: "markup", scope: eval_scope)))
}

#figure(tabla_csv("data/2-blobs.csv"), caption: [Resultados de entrenamiento en @2-blobs])

¡Excelentes noticias! Todos los clasificadores bajo estudio tienen exactitud perfecta, y salvo por una ligeramente negativa $cal(l)$ para #lr, el resto da exactamente 0. Pasemos entonces a algunos dataset mínimamente más complejos.

=== Datasets sintéticos baja dimensión

Consideremos ahora algunas curvas unidimensionales embebidas en $RR^2$:

#figure(
  image("img/datasets-lunas-circulos-espirales.svg", width: 125%),
  caption: flex-caption["Lunas", "Círculos" y "Espirales", con $d_x = 2, d_(cal(M)) = 1$ y $s=4107$][ "Lunas", "Círculos" y "Espirales" ],
) <fig-2>

Resultará obvio al lector que los conjuntos de datos expuestos en @fig-2 no son exactamente variedades "1D" embebidas en "2D", sino que tienen un poco de "ruido blanco" agregado para incrementar la dificultad de la tarea.

#defn("ruido blanco")[Sea $X = (X_1, dots, X_d) in RR^d$ una variable aleatoria tal que $"E"(X_i)=0, "Var"(X_i)=sigma thick forall i in [d]$. Llamaremos "ruido blanco con escala $sigma$" a toda realización de $X$.] <ruido-blanco>

Veamos entonces cómo les fue a los contendientes, considerando primero la exactitud. Recordemos que para cada experimento se realizaron #reps repeticiones: en cada celda reportaremos la exactitud _promedio_, y a su lado entre paréntesis el error estándar cpte.:

#figure(
  image("img/boxplot-lunas-espirales-circulos.svg", width: 120%),
  caption: flex-caption[Boxplots con la distribución de dxactitud en las #reps repeticiones de cada experimento de @fig-2][Boxplots para exactitud de @fig-2],
) <bp-exac-2d>
#figure(
  image("img/boxplot-lunas-espirales-circulos-new.svg", width: 120%),
  caption: flex-caption[Boxplots con la distribución de dxactitud en las #reps repeticiones de cada experimento de @fig-2][Boxplots para exactitud de @fig-2],
)

#figure(tabla_csv("data/exac-ds-2d.csv"), caption: flex-caption[ "mi caption, bo". ][])
#let lsvc = `LSVC`
KDC (en sus dos variantes), KNN y SVC (con kernel RBF) parecieran ser los métodos más competitivos, con mínimas diferencias de performance entre sí: sólo en "círculos" se observa un ligero ordenamiento de los métodos, $svc succ  kdc succ knn $, aunque la performance mediana de #svc está dentro de "los bigotes" de todos los métodos antedichos. La tarea "lunas" pareciera ser la más fácil de todas, en la que hasta una regresión logística sin modelado alguno es adecuada. Para "espirales" y "círculos", #gnb, #lr y #lsvc no logran performar significativamente mejor que el clasificador base.

#defn("clasificador base")[] <clf-base>

¿Cómo se comparan los métodos en términos de la log-verosimilitud y el $R^2$?

#figure(
  image("img/boxplot-r2-lunas-espirales-circulos.svg", width: 120%),
  caption: flex-caption[Boxplots con la distribución de $R^2$ en las #reps repeticiones de cada experimento.][Boxplots para $R^2$ de lunas-circulos-espirales],
)
#figure(tabla_csv("data/r2-ds-2d.csv"), caption: flex-caption[ "mi caption, bo-bo". ][])

Como los métodos basados en máquinas de soporte vectorial resultan en clasificadores _duros_ (@clf-dura), no es posible analizar la log-verosimilitud u otras métricas derivadas. De entre los dos métodos con exactitud similar a esos, es notoriamente mejor el $R^2$ que alcanzan ambos #kdc.
A primera vista, se ve que la dispersión de la métrica es considerable, pues las "cajas" del rango intercuartil son bastante amplias, y aún así se observan _outliers_. En las tres tareas, los clasificadores de estimación de densidad por núcleos tienen las cajas más angostas y los bigotes más cortos, con #kdc mostrando una dispersión menor o igual que #fkdc. En la @bp-exac-2d, observamos que la exactitud de los métodos de k vecinos más cercanos era muy similar a la de #kdc y #svc, sin embargo en términos de $R^2$,
- en el dataset de "espirales" el $R^2$ promedio y mediano son _negativos_, y
- en el de "círculos", aunque la locación #footnote[Entendemos tanto al _promedio_ o _media_ y la _mediana_ como _medidas de locación_] es positiva, la distribución tiene una pesada cola a izquierda, que entra de lleno en los negativos.
En otras palabras, pareciera ser que aunque la exactitud de los métodos basados en vecinos más cercanos es buena, cuando clasifican _mal_, lo hacen con _alta seguridad_, lo que resulta en un pésimo $R^2$.

En esta terna inicial de _datasets_, obtenemos unos resultados aceptables:
- Observamos que el clasificador de @loubesKernelbasedClassifierRiemannian2008 es competitivo (es decir que no sólo Loubes y Pelletier propusieron ),
- aunque la distancia de Fermat muestral no parece mejorar significativamente la exactitud de los calsificadores en ella basados.

#let sfd = $D_(Q, alpha)$
#let euc = $norm(thin dot thin)_2$
Que la bondad de los clasificadores _no empeore_ con el uso de #sfd en lugar de #euc es importante. Por una parte, cuando $alpha = 1$ y $n->oo, quad sfd -> cal(D)_(f, beta) = euc$, con lo cual #fkdc debería performar al menos tan bien como #kdc cuando la grilla de hiperparámetros en la que lo entrenamos incluye a $alpha = 1$. Sin embargo, el cómputo de #sfd es numéricamente bastante complejo, y bien podríamos haber encontrado dificultades computacionales #footnote[De hecho, hubo montones de ellas, cuya resolución progresiva dio lugar a la pequeña librería que acompaña esta tesis y documentamos en el anexo. A mi entender, ningún error de cálculo persiste en el producto final].

#let params = json("data/best_params-2d-lo.json")
==== Comparación entre #kdc y #fkdc para #params.corrida
Concentrémosnos en un segundo en una corrida específica de un ecperimento particular. Por caso, tomemos el dataset "#params.corrida.at(0)", con la semilla #params.corrida.at(1). Los parámetros óptimos de #fkdc resultaron ser
#{
  let from = params.best_params.fkdc
  let d = (:)
  for key in from.keys() {
    d.insert(key, calc.round(float(from.at(key)), digits: 4))
  }
  d
}
, mientras que los de #kdc fueron
#{
  let d = (:)
  let from = params.best_params.kdc
  for key in from.keys() {
    d.insert(key, calc.round(float(from.at(key)), digits: 4))
  }
  d
}. Los anchos de banda son diferentes, y el $alpha$ óptimo encontrado por #fkdc es distinto de 1. Sin embargo, la exactitud de #fkdc fue #params.exac.fkdc, y la de #kdc, #params.exac.kdc, prácticamente idénticas #footnote[Con 400 observaciones para evaluación, dichos porcentajes representan 352 y 354 observaciones correctamente clasificadas, resp.]. ¿Por qué? ¿Será que los algoritmos no son demasiado sensibles a los hiperparámetros elegidos?

Recordemos que la elección de hiperparámetros se hizo con una búsqueda exhaustiva por convalidación cruzada de 5 pliegos. Por lo tanto, _durante el entrenamiento_ se generaron suficientes datos como para graficar la exactitud promedio en los pliegos, en función de $(alpha, h)$. A esta función de los hiperparámetros a una función de pérdida #footnote[En realidad, la exactitud es un "score" o puntaje - mientras más alto mejor-, pero el negativo de cualquier puntaje es una pérdida - mientras más bajo, mejor.] se la suele denominar _superficie de pérdida_.


#figure(
  image("img/heatmap-fkdc-2d-lo.svg"),
  caption: flex-caption()[Exactitud promedio en entrenamiento para la corrida #params.corrida. Las cruces rojas indican la ventana $h$ óptima para cada $alpha$.][Superficie de pérdida para #params.corrida],
)

Nótese que la región amarilla, que representa los máximos puntajes durante el entrenamiento, se extiende diagonalmente a través de todos los valores de $alpha$. Es decir, no hay un _par_ de hiperparámetros óptimos $(alpha^star, h^star)$, sino que fijando $alpha$, siempre pareciera existir un(os) $h^star (alpha)$ que alcanza (o aproxima) la máxima exactitud _posible_ con el método en el dataset. En este ejemplo en particular, hasta pareciera ser que una relación log-lineal captura bastante bien el fenómeno, $log(h^star) prop alpha$. En particular, entonces, $"exac"(h^star (1), 1) approx "exac"(h^star, alpha^star)$, y se entiende que el algoritmo #fkdc, que agrega el hiperparámetro $alpha$ a #kdc no mejore significativamente su exactitud.

Ahora bien, esto es sólo en _un_ dataset, con _una_ semilla especfíca. ¿Se replicará el fenómeno en los otros datasets estudiados? Y si tomásemos datasets con otras características?

#figure(image("img/many-heatmaps-fkdc-2d-lo.svg", width: 140%), caption: "It does replicate")

Antes de avanzar hacia el siguiente conjunto de datos, una pregunta más: ¿qué sucede si aumentamos el nivel de ruido? Es decir, mantenemos los dataset hasta aquí considerados, pero subimos $sigma$ de @ruido-blanco?

==== Efecto del ruido en la performance de clasificación


==== Datasets reales en "mediana" dimensión



==== Dígitos

==== PCA-red MNIST

= Análisis de Resultados

=== Datasets sintéticos, Baja dimensión

=== Datasets orgánicos, Mediana dimensión

=== Alta dimensión: Dígitos

=== Efecto de dimensiones guillemot ruidosasguillemot

=== fKDC: Interrelación entre $h,alpha$

=== fKNN: Comportamiento local-global

= Comentarios finales

=== Conclusiones

=== Posibles líneas de desarrollo

=== Relación con el estado del arte

= Referencias

= Código Fuente

=== sklearn

=== fkdc

=== Datasets
=== Datasets 2d

Esto digo yo

@rosenblattRemarksNonparametricEstimates1956
@carpioFingerprintsCancerPersistent2019
@chaconDatadrivenDensityDerivative2013

== Glosario

/ clausura: ???
/ Riemanniana, métrica: sdfsdf
/ Lebesgue, medida de: ???
/ densidad, estimación de: cf. @berenfeldDensityEstimationUnknown2021
/ ventana: parámetro escalar que determina la "unidad" de distancia
/ núcleo, función: $K$


#outline(target: figure.where(kind: image), title: "Listado de Figuras")
#outline(target: figure.where(kind: table), title: "Listado de Tablas")
#outline(target: figure.where(kind: raw), title: "Listado de código")

#bibliography("../bib/references.bib", style: "harvard-cite-them-right")
