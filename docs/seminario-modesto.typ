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


#image("../img/unif-gaus-kern.png")

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

#image("curse-dim.png")
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
  #image("hormiga-petalo.jpg", height: 70%)
  #colbreak()
  #image("bandera-argentina.png")])


Pero: ¿en qué variedad vive un dígito, o su trazo, o una canción? #emoji.cigarette

== Interludio: Variedades de Riemann


== KDE en variedades de Riemann


== ¿Y si la variedad es desconocida?

== Aprendizaje de distancias

== Distancia de Fermat

= Todo junto: clasificación por estimación de densidad por núcleos con Distancia de Fermat

== Diseño experimental

== Resultados: excelente $R^2$ con exactitud competitiva

== Resultados: superposición de parámetros: $alpha$ y $h$

== Resultados: mejora local, indiferencia global, not ideal for clf