#import "@preview/touying:0.6.1": *

#import themes.metropolis: *

#show: metropolis-theme.with(
  footer: [Seminario ModEsto 2025]
)
#show link: set text(blue)
#show link: underline
#set text(font: "Fira Sans", weight: "light", size: 20pt, lang: "es")
// #show math.equation: set text(font: "Fira Math")
#set strong(delta: 100)
#set par(justify: true)
#set math.equation(numbering: "(1)")

#set quote(block: true)

#let phi = math.phi.alt
#let ind = $ op(bb(1)) $
#let bu(x) = $bold(upright(#x))$

#title-slide(
  author: [Lic. Gonzalo Barrera Borla, Dr. Pablo Groisman - FCEyN, UBA],
  title: "Distancia de Fermat en Clasificadores de Densidad por Núcleos",
)

= El problema de clasificación

== Definición y vocabulario
[ESL §2.2]
- El _aprendizaje estadístico supervisado_ busca estimar (aprender) una variable _respuesta_ a partir de cierta(s) variable(s) _predictora(s)_. 

- Cuando la _respuesta_ es una variable _cualitativa_, el problema de asignar cada observación $x$ a una clase $G in cal(G)={g^1, dots, g^K}$ se denomina _de clasificación_.

- Un _clasificador_ es una función $hat(G)(x)$ que para cada observación $x$, intenta aproximar su verdadera clase $g$ por $hat(g)$ ("ge sombrero").
- Para construir $hat(G)$, contamos con un _conjunto de entrenamiento_ de pares $(x_i, g_i), i in {1, dots, N}$ conocidos. Típicamente, las clases serán MECE, y las observaciones $X in RR^p$.

== Clsasificador de Bayes

Una posible estrategia de clasificación consiste en asignarle a cada observación $x_0$, la clase más probable en ese punto, dada la información disponible. 

$
hat(G)(x) = arg max_(g in cal(G)) Pr(G=g|X=x)
$

Esta razonable solución es conocida como el _clasificador de Bayes_, y se puede reescribir usando la regla homónima como

$
hat(G)(x) = g_i <=> Pr(g_i|X=x) &= max_(g in cal(G)) Pr(G=g|X=x) \
&=max_(g in cal(G)) Pr(X=x|G=g) times Pr(G=g)
$

== Clasificadores "suaves" y "duros"

- Un clasificador que responda "¿_qué clase_ es la que más probablemente contenga esta observación" es un clasificador "duro".
- Un clasificador que además puede responder "¿_cuán probable_ es que esta observación pertenezca a cada clase $g_j$?" es un clasificador "suave".
- La regla de Bayes para clasificación nos puede dar un clasificador duro al maximizar la probabilidad; más aún, también puede construir un clasificador suave:

$
hat(Pr)(G=g_i|X=x) &= (hat(Pr)(x|G=g_i) times hat(Pr)(G=g_i)) / (hat(Pr)(X=x)) \
&= (hat(Pr)(x|G=g_i) times hat(Pr)(G=g_i)) / (sum_(k in [K]) hat(Pr)(X=x, G=g_k)) \
$

= Estimación de Densidad por Núcleos
== Clasificador de Bayes empírico
- Si el conjunto de entrenamiento ${(x_1, g_1), dots, (x_N, g_N)}$ proviene de un muestreo aleatorio uniforme, las probablidades de clase $pi_i = Pr(G=g^((i)))$ se pueden aproximar razonablemente por las proporciones muestrales $ hat(pi)_i = \#{g_j :g_j = g^((i))}slash N$ 

- Resta hallar una aproximación $Pr(x|G=g)$ para cada clase, ya sea a través de una función de densidad, de distribución, u otra manera.

== Estimación unidimensional

[ESL §6.6, Parzen 1962]


Para fijar ideas, asumamos que $X in RR$ y consideremos la estimación de densidad en una única clase para la que contamos con $N$ ejemplos ${x_1, dots, x_N}$. Una aproximación $hat(f)$ directa sería
(1) $
  hat(f)(x_0) = \#{x_i in cal(N)(x_0)} / (N times h)
$ #label("eps-nn")

donde $cal(N)$ es un vecindario métrico de $x_0$ de diámetro $h$. Esta estimación es irregular, con saltos discretos en el numerador, por lo que se prefiere el estimador suavizado por núcleos de Parzen-Rosenblatt

$
  hat(f)(x_0) = 1/N sum_(i=1)^N K (x_0, x_i)
$ #label("parzen")

== Función núcleo o "_kernel_"

Se dice que $K(x) : RR-> RR$ es una _función núcleo_ si

- toma valores reales no negativos: $K(u) >= 0 forall u in "sop"K$,
- está normalizada: $integral_(-oo)^(+oo) K(u) d u = 1$,
- es simétrica: $K(u) = K(-u)$ y
- alcanza su máximo en el centro: $max_u K(u) = K(0)$

Observación 1: Todas las funciones de densidad simétricas centradas en 0 son núcleos; en particular, la densidad "normal estándar" $phi(x) = 1/sqrt(2 pi) exp(-x^2 / 2 )$ lo es.

Observación 2: Si $K(u)$ es un núcleo, entonces $K_h (u) = 1/h op(K)(u / h)$ también lo es.

Observación 3: Si $ind(dot)$ es la función indicadora, resulta que $op(U_h)(x) = 1/h ind(-h/2 < x < h/2)$ es un núcleo válido, y el estimador de @parzen con núcleo $U_h$ devuelve el estimador @eps-nn
== Núcleo uniforme


#image("img/unif-gaus-kern.png")

== Clasificador de densidad por núcleos
[ESL §6.6.2]

Si $hat(f)_k, k in 1, dots, K$ son estimadores de densidad por núcleos #footnote[KDEs ó _Kernel Density Estimators_, por sus siglas en inglés] según @parzen, la regla de Bayes nos provee un clasificador suave
$
hat(Pr)(G=g_i|X=x) &= (hat(Pr)(x|G=g_i) times hat(Pr)(G=g_i)) / (hat(Pr)(X=x)) \
&=(hat(pi)_i hat(f)_i (x)  )/ (sum_(k=1)^K hat(pi)_k hat(f)_k (x)) \
$

== Interludio: Naive Bayes
[ESL §6.6.3]

¿Y si las $X$ son multivariadas ($X in RR^d, d>= 2$)? ¿Se puede adaptar el clasificador?

Sí, pero es complejo. Un camino sencillo: asumir que condicional a cada clase $G=j$, los predictores $X_1, X_2, dots, X_p$ se distribuyen indenpendientemente entre sí.

$
  f_j (X) = product_(i=1)^p f_(j,i) (X_i)
$

Cada densidad marginal $ f_(j,i)$ condicional a la clase se puede estimar usando KDE univariado, y hasta se puede aplicar - usando histogramas - cuando algunas componentes $X_i$ son discretas.

A este procedimiento, se lo conoce cono "Naive Bayes".

== KDE multivariado
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

== Dificultades: elección de $bu(H)$
Sean las clases de matrices pertenecientes a $RR^(d times d)$ ...
- $cal(F)$, de matrices simétricas definidas positivas,
- $cal(D)$, de matrices diagonales definidas positivas ($cal(D) subset.eq cal(F)$) y
- $cal(S)$, de múltiplos escalares de la identidad: $cal(S) = {h^2 bu(I):h >0} subset.eq cal(D)$
 
Aún tomando una única $bu(H)$ para _toda_ la muestra, $bu(H) in dots$
- $cal(F)$, requiere definir $mat(d; 2) = d(d-1)/2$ parámetros de ventana,
- $cal(D)$ requiere $d$ parámetros, y
- $cal(S)$ tiene un único parámetro $h$.

 A priori no es posible saber qué parametrización conviene, pero en general $bu(H) in cal(D)$ parece un compromiso razonable: no se pierde demasiado contra $cal(F)$, pero tampoco se padece la "rigidez" de $bu(H) in cal(S)$.

== Dificultades: La maldición de la dimensionalidad

[ESL §2.5, Wand & Jones 1995 §4.9 ej 4.1]

Sean $X_i tilde.op^("iid")"Uniforme"([-1, 1]^d), i in {1, dots, N}$, y consideremos la estimación de la densidad en el origen, $f(bu(0))$. Suponga que el núcleo $K_(bu(H))$ es un "núcleo producto" basado en la distribución univariada $"Uniforme(-1, 1)"$, y $bu(H) = h^2 bu(I)$. Derive una expresión para la proporción esperada de puntos incluidos dentro del soporte del núcleo $K_bu(H)$ para $h, d$. arbitrarios.

(... interludio de pizarrón ...)

$
  Pr(X in [-h, h]^d) &=  Pr(inter_(i=1)^d abs(X_i) <= h) \
  Pr(X in [-0.95, 0.95]^50) &approx 0.0077 \
$

== Dificultades: La maldición de la dimensionalidad

#image("img/curse-dim.png")
Para $h <=0.5, Pr(dot) < 1 times 10^(-15)$. Aún para $h=0.95, Pr(dot) approx 0.0077$ #emoji.face.shock

= Clasificación en variedades

== La hipótesis de la variedad ("manifold hypothesis")
[Bengio Repr learning]
[#link("https://www.reddit.com/r/MachineLearning/comments/mzjshl/d_who_first_advanced_the_manifold_hypothesis_to/")[Bengio en Reddit]
]

La hipótesis de la variedad postula que los datos $X in RR^(d_X)$ muestreados soportados en un espacio de alta dimensionalidad #footnote[E.g.: imágenes, audio, video, secuencias de nucleótidos]. tenderán a concentarse sobre una _variedad_ $cal(M)$, potencialmente de mucha menor dimensión $d_(cal(M)) << d_X$, embebida en el espacio original $cal(M) subset.eq RR^(d_X)$.

- Well suited for AI tasks such as those involving images, sounds or text, for which most uniformly sampled input configurations are unlike natural stimuli.
- archetypal manifold modeling algorithm is, not surprisingly, also the archetypal low dimensional representation learning algorithm: Principal Component Analysis, which models a linear manifold.
- Data manifold for complex real world domains are however expected to be strongly nonlinear.


== IRL

#columns(2,[
  #image("img/hormiga-petalo.jpg", height: 70%)
  #colbreak()
  #image("img/bandera-argentina.png")])


Pero: ¿en qué variedad vive un dígito, o su trazo, o una canción? #emoji.cigarette

== Interludio: Variedades de Riemann [Wikipedia]

#quote[Una variedad $d$-dimensional $cal(M)$ es un espacio _topológico_ tal que cada punto $p in cal(M)$ tiene un vecindario $U$ que resulta _homeomórfico_ a un conjunto abierto en $RR^d$]

- topológico: se puede definir cercanía (pero no necesariamente distancia), permite definir funciones continuas y límites
- homeomórfico a $RR^d$: para cada punto $p in cal(M)$, existe un mapa _biyectivo_ y _suave_ entre el vecindario de $p$ y $RR^d$. El conjunto de tales mapas se denomina _atlas_.

#grid(columns: (80%, 20%), [
  Sea $T_p cal(M)$ el _espacio tangente_ a un punto $p in cal(M)$, y $g_p : T_p cal(M) times T_p cal(M) -> RR$ una forma _bilinear pos. def._ para cada $p$ que induce una _norma_ $||v||_p= sqrt(g_p(v, v))$. 
  
  Decimos entonces que $g_p$ es una métrica Riemanniana y el par $(cal(M), g)$ es una variedad de Riemann, donde las nociones de _distancia, ángulo y geodésica_ están bien definidas.], image("img/Tangent_plane_to_sphere_with_vectors.svg",)
)

== KDE en variedades de Riemann [Pelletier 2005]
- Sea $(cal(M), g)$ una variedad de Riemann compacta y sin frontera de dimensión $d$, y usemos $d_g$ para denotar la distancia de Riemann.
- Sea $K$ un _núcleo isotrópico en $cal(M)$ soportado en la bola unitaria_ (cf. conds. (i)-(v))
- Sean $p, q in cal(M)$, y $theta_p (q)$ la _función de densidad de volumen en $cal(M)$_ #footnote[¡Ardua definición! Algo así como el cociente entre las medida de volumen en $cal(M)$, y su transformación via el mapa local a $RR^d$]
Luego, el estimador de densidad para $X_i tilde.op^("iid")f$ es $f_(N,K):cal(M) ->RR$ que a cada $p in cal(M)$ le asocia el valor
$
  f_(N,K) (p) = N^(-1) sum_(i=1)^N K_h (p,X_i) = N^(-1) sum_(i=1)^N 1/h^d 1/(theta_X_i (p))K((op(d_g)(p, X_i))/h)
$

con la restricción de que la ventana $h <= h_0 <= op("inj")(cal(M))$, el _radio de inyectividad_ de $cal(M)$ #footnote[el ínfimo entre el supremo del radio de una bola en cada $p$ tal que su mapa es un difeomorfismo]

== Interludio: densidad de volumen en la esfera [Henry y Rodríguez, 2009]

#columns(2)[
  En _"Kernel Density Estimation on Riemannian Manifolds: Asymptotic Results" (2009)_, Guillermo Henry y Daniela Rodriguez estudian algunas propiedades asintótica de este estimador, y las ejemplifican con datos de sitios volcánicos en la superficie terrestre.
  En particular, calculan la densidad de volumen $theta_p(q)$
#image("img/densidad-volumen-esfera.png")
#colbreak()
#image("img/henry-rodriguez-bolas.png")
]

== Clasificación en variedades [Loubes y Pelletier 2008]

¡Clasificador de Bayes + KDE en Variedades = Clasificación (suave o dura) en variedades!

Plantean una regla de clasificación $hat(G)$ para 2 clases adaptable a K clases de forma directa. Sea $p in cal(M)$ una variedad riemanniana como antes, y ${(x_1, g_1), dots, (x_N, g_N)}$ nuestras observaciones y sus clases. Luego,

$
  hat(G) (p) = arg max_(g in cal(G)) sum_(i=1)^N ind(g_i = g)K_h (p,X_i)
$

#pause 
#align(center)[Pero... ¿y si la variedad es desconocida?]

= Aprendizaje de distancias

== El ejemplo canónica: Análisis de Componentes Principales (PCA)

#align(center)[#image("img/pca.png", height:90%)]
#text(size: 17pt)[Karl Pearson (1901), _"LIII. On lines and planes of closest fit to systems of points in space."_]


== El algoritmo más _cool_: Isomap
#grid(columns: (35%, 65%), column-gutter:20pt, text(size: 19pt)[
  1. Construya el grafo de $k, epsilon$-vecinos, $bu(N N)=(bu(X), E)$

  2. Compute los caminos mínimos - las geodésicas entre observaciones, $d_(bu(N N))(x, y)$.

  3. Construya una representación ("_embedding"_) $d^*$−dimensional que minimice la discrepancia ("stress") entre $d_(bu(N N))$ y la distancia euclídea en $RR^d^*$
],image("img/isomap-2.png", height:90%))
#text(size: 17pt)[Tenenbaum et al (2000), _"A Global Geometric Framework for Nonlinear Dimensionality Reduction"_]

== Distancia de Fermat [Groisman, Jonckheere, Sapienza (2019); Little et al (2021)]

#quote(attribution: "P. Groisman et al (2019)")[
  #set text(size: 17pt)
_We tackle the problem of learning a distance between points, able to capture both the geometry of the manifold and the underlying density. We define such a sample distance and prove the convergence, as the sample size goes to infinity, to a macroscopic one that we call Fermat distance as it minimizes a path functional, resembling Fermat principle in optics._]

Sea $f$ una función continua y positiva, $beta >=0$
 y $x, y in S subset.eq RR^d$. Definimos la _Distancia de Fermat_ $cal(D)_(f, beta)(x, y)$ como:

$
cal(T)_(f, beta)(gamma) = integral_gamma f^(-beta) space, quad quad quad cal(D)_(f, beta)(x, y) = inf_(gamma in Gamma) cal(T)_(f, beta)(gamma) #pause quad #emoji.face.shock #pause
$

... donde el ínfimo se toma sobre el conjunto $Gamma$ de todos los caminos rectificables entre $x$ e $y$ contenidos en $overline(S)$, la clausura de $S$, y la integral es entendida con respecto a la longitud de arco dada por la distancia euclídea.
 
== Distancia de Fermat muestral

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
#pause

¡Esta sí la podemos aprender de los datos! #emoji.arm.muscle

= Todo junto:
#text(size: 25pt)[Clasificación en variedades desconocidas por estimación de densidad por núcleos con Distancia de Fermat Muestral]

== Algunas dudas

- Entrenar el clasificador por validación cruzada está OK: como $bu(X)_"train" subset.eq bu(X)$ y $bu(X)_"test" subset.eq bu(X)$, se sigue que $forall (a, b) in {bu(X)_"train" times in bu(X)_"test"} subset.eq {bu(X) times bu(X)}$ y $D_(bu(X), alpha) (a, b)$ está bien definida. #pause ¿Cómo sé la distancia _muestral_ de una _nueva_ observación $x_0$, a los elementos de cada clase?\
#pause

Para cada una de las $g_i in cal(G)$ clases, definimos el conjunto $
Q_i= {x_0} union {x_j : x_j in bu(X), g_j = g_i, j in {1, dots, N}}
$
y calculamos $D_(Q_i, alpha) (x_0, dot)$

== Algunas dudas

- El clasificador de Loubes & Pelletier asume que todas las clases están soportadas en la misma variedad $cal(M)$. ¿Quién dice que ello vale para las diferentes clases?

#pause
¡Nadie! Pero
1. No hace falta dicho supuesto, y en el peor de los casos, podemos asumir que la unión de las clases está soportada en _cierta_ variedad de Riemman, que resulta de (¿la clausura de?) la unión de sus soportes individuales. #pause
2. Sí es cierto que si las variedades (y las densidades que soportan) difieren, tanto el $alpha_i^*$ como el $h_i*$ "óptimos" para los estimadores de densidad individuales no tienen por qué coincidir. #pause
3. Aunque las densidades individuales $f_i$ estén bien estimadas, el clasificador resultante puede ser mal(ard)o si no diferencia bien "en las fronteras". Por simplicidad, además, decidimos parametrizar el clasificador con dos únicos hiperparámetros globales: $alpha, h$.

== Diseño experimental

1. Desarrollamos un clasificador compatible con el _framework_ de #link("https://arxiv.org/abs/1309.0238", `scikit-learn`)  según los lineamientos de Loubes & Pelleteir, que apodamos `KDC`. #pause
2. Implementamos el estimador de la distancia muestral de Fermat, y combinándolo con KDC, obtenemos la titular "Clasificación por KDE con Distancia de Fermat", `FKDC`. #pause
3.  Evaluamos el _pseudo-$R^2$_ y la _exactitud_ ("accuracy") de los clasificadores propuestos en diferentes _datasets_, relativa a técnicas bien establecidas: #pause
#columns(2)[
- regresión logística (`LR`)
- clasificador de  soporte vectorial (`SVC`) #footnote[sólo se consideró su exactitud. ya que no es un clasificador suave]
- _gradient boosting trees_ (`GBT`)
#colbreak()
- k-vecinos-más-cercanos (`KN`)
- Naive Bayes Gaussiano (`GNB`)]

#pagebreak()

- La implementación de `KNeighbors` de referencia acepta distancias precomputadas, así que incluimos una versión con distancia de Fermat, que apodamos `F(ermat)KN`. #pause

- Para ser "justos", se reservó una porción de los datos para la evaluación comparada, y del resto, cada algoritmo fue entrenado repetidas veces por validación cruzada de 5 pliegos, en una extensísima grilla de hiperparametrizaciones. Este procedimiento *se repitió 25 veces en cada dataset*. #pause

- La función de score elegida fue `neg_log_loss` ($ = cal(l)$) para los clasificadores suaves, y `accuracy` para los duros.

#pagebreak()
- Para tener una idea "sistémica" de la performance de los algoritmos, evaluaremos su performance con _datasets_ que varíen en el tamaño muestral $N$, la dimensión $p$ de las $X_i$, el nro. de clases $k$ y su origen ("real" o "sintético"). #pause

- Cuando creamos datos sintéticos en variedades  con dimensión intrínseca menor a la ambiente, (casi) cualquier clasificador competente alcanza exactitud perfecta; para complejizar la tarea, agegamos un poco de "ruido" a las observaciones, y también analizamos sus efectos.

== Regla de Parsimonia

- ¿Qué parametrización elegir cuando "en test da todo igual"? 
#pause
#align(center)[ #emoji.knife de Occam: la más "sencilla" (TBD)]

#pause
- ¿Qué parametrización elegir cuando "en test da *casi* todo igual"? 
#pause

#align(center)[*Regla de $1sigma$*: De las que estén a $1sigma$ de la mejor, la más sencilla.]

#pause

¿Sabemos cuánto vale $sigma$?

== $R^2$ de McFadden
Sea $cal(C)_0$ el clasificador "base", que asigna a cada observación y posible clase, la frecuencia empírica de clase encontrada en la muestra $bu(X)$. Para todo clasificador suave $cal(C)$, definimos el $R^2$ de McFadden como
  $ op(R^2)(cal(C) | bu(X)) = 1 - (op(cal(l))(cal(C))) / (op(cal(l))(cal(C)_0)) $


donde $cal(l)(dot)$ es la log-verosimilitud clásica. Nótese que $op(R^2)(cal(C)_0) = 0$. #pause A su vez, para un clasificador perfecto $cal(C)^star$ que otorgue toda la masa de probabilidad a la clase correcta, tendrá $op(L)(cal(C)^star) = 1$ y log-verosimilitud igual a 0, de manera que $op(R^2)(cal(C)^star) = 1 - 0 = 1$.
#pause

Sin embargo, un clasificador _peor_ que $cal(C)_0$ en tanto asigne bajas probabilidades ($approx 0$) a las clases correctas, puede tener un $R^2$ infinitamente negativo.

= Resultados
== 2D, 2 clases: excelente $R^2$ con exactitud competitiva

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

== Superposición de parámetros: $alpha$ y $h$


- El uso de la distancia de Fermat muestral no hiere la performance, pero las mejoras son nulas o marginales. ¿Por qué?

#pause
Si recordamos $hat(f)_(K,N)$ según Loubes & Pelletier, al núcleo $K$ se lo evalúa sobre 
$
 (d (x_0, X_i)) / h, quad d = D_(Q_i, alpha)
$
#pause
Lo que $alpha$ afecta a $hat(f)$ vía $d$, también se puede conseguir vía $h$.
#pause
Si $D_(Q_i, alpha) prop ||dot|| $ (la distancia de fermat es proporcional a la euclídea), los efectos de $alpha$ y $h$ se "solapan" 
#pause

... y sabemos que localmente, eso es cierto #emoji.face.tear

== Parámetros óptimos para $"(F)KDC"$ en `espirales_lo`
#align(center)[#image("img/optimos-espirales_lo.png", height: 80%)]


== Superficies (o paisajes) de _score_ para `(espirales_lo, 1434)`

#align(center)[#image("img/heatmap-fkdc-2d-lo-new.svg", height: 110%)]

== Alt-viz: Perfiles de pérdida para `(espirales_lo, 1434)`

#align(center)[#image("img/perfiles-perdida-espirales-1434.png", height: 110%)]

== Fronteras de decisión para `(espirales_lo, 1434)`

#align(center)[#image("img/gbt-lr-espirales.png")]
#pagebreak()
#align(center)[#image("img/kn-espirales.png")]
#pagebreak()
#align(center)[#image("img/kdc-espirales.png")]
#pagebreak()
#align(center)[#image("img/gnb-svc-espirales.png")]



== 3D, 2 clases + piononos

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

== Parámetros óptimos para $"(F)KDC"$ en `helices_0`
#align(center)[#image("img/optimos-helices_0.png", height: 100%)]

== Microindiferencia, macrodiferencia

- En zonas con muchas observaciones (por tener alta $f$ o alto $N$) sampleadas, la distancia de Fermat y la euclídea coinciden. #pause
- "Localmente", siempre van a coincidir, aunque sea en un vecindario muy pequeño. #pause
- Si el algoritmo de clasificación sólo depende de ese vencindario local para clasificar, no hay ganancia en la distancia de Fermat. #pause
- ¡Pero tampoco hay pérdida si se elige mal `n_neighbors`! #emoji.person.shrug


== $R^2$ por semilla para $"(F)KN"$ en `helices_0`
#align(center)[#image("img/r2-fkn-kn-helices_0.png", height: 100%)]

== $R^2$ y $alpha^star$ para $"(F)KN"$ en `helices_0`, `n_neighbors` seleccionados
#align(center)[#image("img/r2-fkn-kn-n_neighbors-seleccionados.png", height: 65%)]

== Mejor $R^2$ para $"(F)KN"$ en `helices_0`, en función de `n_neighbors`

#image("img/helices_0-fkn_kn-mean_test_score.png")


== $R^2$ por semilla para $"(F)KN"$ en `eslabones_0`
#align(center)[#image("img/outputa.png", height: 100%)]

== $R^2$ y $alpha^star$ para $"(F)KN"$ en `eslabones_0`, `n_neighbors` seleccionados
#align(center)[#image("img/Screenshot 2025-07-18 at 11.43.27 AM.png", height: 65%)]

== Mejor $R^2$ para $"(F)KN"$ en `eslabones_0`, en función de `n_neighbors`

#image("img/outputb.png")

== Otros datasets: 2D mucho ruido
#columns(3)[
  #image("img/lunas_hi-overall.png")
  #colbreak()
  #image("img/circulos_hi-overall.png")
  #colbreak()
  #image("img/espirales_hi-overall.png")
]
== Otros datasets: 15D
#columns(4)[
  #image("img/pionono_12-overall.png")
  #colbreak()
  #image("img/eslabones_12-overall.png")
  #colbreak()
  #image("img/helices_12-overall.png")
  #colbreak()
  #image("img/hueveras_12-overall.png")  
]
== Otros datasets: multiclase
#columns(4)[
  #image("img/iris-overall.png")
  #colbreak()
  #image("img/vino-overall.png")
  #colbreak()
  #image("img/pinguinos-overall.png")
  #colbreak()
  #image("img/anteojos.png")  
]
== Otros datasets: `digitos` y `mnist`

#columns(2)[
  #image("img/digitos-overall.png")
  #colbreak()
  #image("img/mnist-overall.png")
]
