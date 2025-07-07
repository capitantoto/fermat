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

#let phi = math.phi.alt
#let ind = $ op(bb(1)) $

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
hat(G)(x) = arg max_(g in cal(G)) Pr(g|X=x)
$

Esta razonable solución es conocida como el _clasificador de Bayes_, y se puede reescribir usando la regla homónima como

$
hat(G)(x) = g_i <=> Pr(g_i|X) &= max_(g in cal(G)) Pr(g|X=x) \
&=max_(g in cal(G)) Pr(x|G=g) times Pr(G=g)
$

= Estimación de Densidad por Núcleos
== Clasificador de Bayes empírico
- Si el conjunto de entrenamiento ${(x_1, g_1), dots, (x_N, g_N)}$ proviene de un muestreo aleatorio uniforme, las probablidades de clase $pi_i = Pr(G=g^((i)))$ se pueden aproximar razonablemente por las proporciones muestrales $ hat(pi)_i = \#{g_j :g_j = g^((i))}slash N$ 

- Resta hallar una aproximación $Pr(x|G=g)$ para cada clase, ya sea a través de una función de densidad, de distribución, u otra manera.

== Estimación unidimensional

[ESL §6.6, Parzen 1962]


Para fijar ideas, asumamos que $X in RR$ y consideremos la estimación de densidad en una única clase para la que contamos con $N$ ejemplos ${x_1, dots, x_N}$. Una aproximación $hat(f)$ directa sería
$
  hat(f)(x_0) = \#{x_i in cal(N)(x_0)} / (N times h)
$

donde $cal(N)$ es un vecindario métrico de $x_0$ de diámetro $h$. Esta estimación es irregular, con saltos discretos en el numerador, por lo que se prefiere el estimador suavizado por núcleos de Parzen-Rosenblatt

$
  hat(f)(x_0) = 1/N sum_(i=1)^N K (x_0, x_i)
$

== Función núcleo o "_kernel_"

Se dice que $K(x) : RR-> RR$ es una _función núcleo_ si

- toma valores reales no negativos: $K(u) >= 0 forall u in "sop"K$,
- está normalizada: $integral_(-oo)^(+oo) K(u) d u = 1$,
- es simétrica: $K(u) = K(-u)$ y
- alcanza su máximo en el centro: $max_u K(u) = K(0)$

Observación 1: Todas las funciones de densidad simétricas centradas en 0 son núcleos; en particular, la densidad "normal estándar" $phi(x) = 1/sqrt(2 pi) exp(-x^2 / 2 )$ lo es.

Observación 2: Si $K(u)$ es un núcleo, entonces $K_h (u) = 1/h op(K)(u / h)$ también lo es.

== Núcleo uniforme

Si $ind(dot)$ es la función indicadora, resulta que $U_h(x) = 1/h ind(-h/2 < x < h/2)$ es un núcleo válido, y el estimador de Parzen-Rosenblatt con núcleo $U_h$ devuelve el estimador 

#image("../img/unif-gaus-kern.png")

\#{x_i in cal(N)(x_0)} / (N times h)
== Alta dimensión : Naive Bayes

== La maldición de la dimensionalidad

== La hipótesis de la variedad

= Clasificación en variedades

== KDE en variedades de Riemann

== ¿Y si la variedad es desconocida?

== Aprendizaje de distancias

== Distancia de Fermat

= Todo junto: clasificación por estimación de densidad por núcleos con Distancia de Fermat

== Diseño experimental

== Resultados: excelente $R^2$ con exactitud competitiva

== Resultados: superposición de parámetros: $alpha$ y $h$

== Resultados: mejora local, indiferencia global, not ideal for clf