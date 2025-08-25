#import "@preview/ctheorems:1.1.3": *

// ################
// # definiciones #
// ################
#let phi = math.phi.alt
#let ind = $ op(bb(1)) $
#let iid = "i.i.d."
#let sop = $ op("sop") $
#let Pr = $ op("Pr") $
#let bu(x) = $bold(upright(#x))$
#let GG = $ cal(G) $
#let MM = $ cal(M) $
#let HH = $ bu(H) $
#let KH = $ op(K_HH) $
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
#show link: it => underline(text(it, fill: blue))

// ### TOC y listados
#outline(depth: 2)

= TODOs
- [ ] Ponderar por $n^beta$
- [ ] Evitar coma entre sujeto y predicado
- [ ] Mencionar Rodríguez x2:
  - @forzaniPenalizationMethodEstimate2022
  - @henryKernelDensityEstimation2009
- Algo de densidad de volumen:
  - @berenfeldDensityEstimationUnknown2021
  - @bickelLocalPolynomialRegression2007

= Vocabulario y Notación
A lo largo de esta monografía tomaremos como referencia enciclopédica al _Elements of Statistical Learning_ @hastieElementsStatisticalLearning2009, de modo que en la medida de lo posible, basaremos nuestra notación en la suya también.

Típicamente, denotaremos a las variables independientes #footnote[También conocidas como predictoras, o _inputs_] con $X$. Si $X$ es un vector, accederemos a sus componentes con subíndices, $X_j$. En el contexto del problema de clasificación, la variable _cualitativa_ dependiente #footnote[También conocida como variable respuesta u _output_] será $G$ (de $G$rupo). Usaremos letras mayúsculas como $X, G$ para referirnos a los aspectos genéricos de una variable. Los valores _observados_ se escribirán en minúscula, de manera que el i-ésimo valor observado de $X$ será $x_i$ (de nuevo, $x_i$ puede ser un escalar o un vector).

Representaremos a las matrices con letras mayúsculas en negrita, $bu(X)$; e.g.: el conjunto de de $N$ vectores $p$-dimensionales ${x_i, i in {1, dots, N}}$ será representado por la matrix $bu(X)$ de dimensión $N times p$.

En general, los vectores _no_ estarán en negrita, excepto cuando tengan $N$ componentes; esta convención distingue el $p-$vector de _inputs_ para la i-ésima observación,  $x_i$, del $N-$vector $bu(x)_j$ con todas las observaciones de la variable $X_j$. Como todos los vectore se asumen vectores columna, la i-ésima fila de $bu(X)$ es $x_i^T$, la traspuesta de la i-ésima observación $x_i$.

A continuación, algunos símbolos y operadores utilizados a lo largo del texto:

#set terms(separator: h(2em, weak: true), spacing: 1em)

/ $RR$: los números reales
/ $d_x$:
/ $RR^(d_x)$:
/ $[k]$: el conjunto de los k números enteros, ${1, dots, k}$
/ $cal(M)$:
/ $bold(upright(H))$:
/ $norm(dot)$:
/ ${bold(upright(X))}$:
/ $X_(i, j)$:
/ $ind(x)$: la función indicadora, $ind(x)=cases(1 "si" x "es verdadero", 0 "si no")$
/ $Pr(x)$: función de probabilidad,
/ $EE(x)$: esperanza,
/ $iid$: independiente e idénticamente distribuido (suele aplicar a una muestra $bu(X)$
/ $emptyset$: el conjunto vacío
= Preliminares

== El problema de clasificación

=== Definición y vocabulario
El _aprendizaje estadístico supervisado_ busca estimar (aprender) una variable _respuesta_ a partir de cierta(s) variable(s) _predictora(s)_. Cuando la _respuesta_ es una variable _cualitativa_, el problema de asignar cada observación $X$ a una clase $G in GG={GG^1, dots, GG^K}$ se denomina _de clasificación_. En general, reemplazaremos los nombres o "etiquetas" de clases $g_i$ por los enteros correspondientes, $G in [K]$. En esta definición del problema, las clases son mutuamente excluyentes y conjuntamente exhaustivas:

- mutuamente excluyentes: cada observación $X_i$ está asociada a lo sumo a una clase
- conjuntamente exhaustivas: cada observación $X_i$ está asociada a alguna clase.

#defn("clasificador")[
  Un _clasificador_ es una función $hat(G)(X)$ que para cada observación intenta aproximar su verdadera clase $G$ por $hat(G)$ ("ge sombrero").
] <clasificador>

Para construir $hat(G)$, contaremos con una muestra o _conjunto de entrenamiento_ $bu(X), bu(g)$,  de pares $(x_i, g_i), i in {1, dots, N}$ conocidos.

Para discernir cuán bien se "ajusta" un clasificador a los datos, la teoría requiere de una función de _pérdida_ $L(G, hat(G)(X))$. #footnote[_loss function_ en inglés. A veces también "función de riesgo" - _risk function_.]. Será de especial interés la función de clasificación $f$ que minimiza la _esperanza de predicción errada_ $"EPE"$:

$
  hat(G) = arg min_f "EPE"(f) =arg min_f EE(L(G, f(X)))
$
donde la esperanza es contra la distribución conjunta $X, G)$. Por la ley de la probablidad total, podemos condicionar a X y expresar el EPE como

$
  "EPE"(f) & = EE(L(G, hat(G)(X))) \
           & = sum_(k in [K]) EE(L(GG_k, hat(G)(X)) Pr(GG_k | X)) EE(X) \
           & = EE(X) sum_(k in [K]) EE(L(GG_k, g) Pr(GG_k | X)) \
$
Y basta con minimizar punto a punto para obtener una expresión computable de $hat(G)$:
$
  hat(G)(x) & = arg min_(g in GG) sum_(k in [K]) L(GG_k, g) Pr(GG_k | X = x) \
            & = arg min_(g in GG) sum_(k in [K]) L(GG_k, g) Pr(GG_k | X = x)
$
Con la _pérdida 0-1_ #footnote[que no es otra cosa que la función indicadora de un error en la predicción, $bu(01)(hat(G), G) = ind(hat(G) != G)$], la expresión se simplifica a
$
  hat(G)(x) & = arg min_(g in GG) sum_(k in [K]) ind(cal(G)_k != g) Pr(GG_k|X=x) \
            & = arg min_(g in GG) [1-Pr(g|X=x)] \
            & = arg max_(g in GG) Pr(g | X = x)
$

Esta razonable solución se conoce como el _clasificador de Bayes_, y sugiere que clasifiquemos a cada observación según la clase modal condicional a su distribución conjunta $Pr(G|X)$. Su error esperado de predicción $"EPE"$ se conoce como la _tasa de Bayes_. Un aproximador directo de este resultado es el clasificador de k vecinos más cercano #footnote[_k-nearest-neighbors classifier_]

#defn("clasificador de k-vecinos-más-cercanos")[
  Sean $x^((1)), dots, x^((k)))$ los $k$ vecinos más cercanos a $x$, y $GG^((1)), dots, GG^((k))$ sus respetivas clases. El clasificador de k-vecinos-más-cercanos le asignará a $x$ la clase más frecuente entre $GG^((1)), dots, GG^((k))$. Más formalmente:
  $
    hat(G)_("kNN")(x) & =GG_("kNN") = arg max_(g in GG) sum_(i in [k]) ind(GG^((i)) = g) \
                      & <=> \#{i : GG^((i)) = GG_("kNN"), i in [k]} = max_(g in GG) \#{i : GG^((i)) = g, i in [k]} \
  $

] <knn-clf>

=== Clasificador de Bayes empírico

La _Regla de Bayes_,
$
  Pr(G|X) = (Pr(X| G) times Pr(G)) / (Pr(X))
$
nos sugiere una reescritura de $hat(G)$:
$
  hat(G)(x) = g & = arg max_(g in GG) Pr(g | X = x) \
                & <=> Pr(g|X=x) = max_(g in GG) Pr(g|X=x) \
                & <=> Pr(g|X=x) =max_(g in GG) Pr(X=x|g) times Pr(g) \
                & <=> Pr(GG_k|X=x) =max_(k in [K]) Pr(X=x|GG_k) times Pr(GG_k) \
$

A las probablidades "incondicionales" de clase $Pr(GG_k)$ se las suele llamar su "distribución a priori", y notarlas por $pi = (pi_1, dots, pi_K)^T$, con #box[$pi_k = Pr(GG_k) forall k in [K], quad sum pi_k = 1$]. Una aproximación razonable, si es que el conjunto de entrenamiento se obtuvo por muestreo aleatorio simple #footnote[_simple random sampling_, o s.r.s.], es tomar las proporciones muestrales
$
  forall k in [K], quad hat(pi)_k & = N^(-1) sum_(i in [N]) ind(g_i = GG_k) \
                                  & = \#{g_i : g_i = GG_k, i in [N]} / N
$


Resta hallar una aproximación $hat(Pr)(X=x|GG_k)$ a las probabilidades condicionales $X|GG_k$ para cada clase.

== Estimación de densidad por núcleos

De conocer las $K$ densidades $f_(X|GG_k)$, el cómputo de las mentada probabilidades es directo. Tal vez la metodología más estudiada a tales fines es la _estimación de densidad por núcleos_, comprensivamente reseñada en @hastieElementsStatisticalLearning2009[§6.6]. Al estimador resultante, sobre todo en el caso unidimensional, se lo conoce con el nombre de Parzen-Rosenblatt, por sus contribuciones fundacionales en el área @parzenEstimationProbabilityDensity1962
@rosenblattRemarksNonparametricEstimates1956
.
==== Estimación unidimensional


Para fijar ideas, asumamos que $X in RR$ y consideremos la estimación de densidad en una única clase para la que contamos con $N$ ejemplos ${x_1, dots, x_N}$. Una aproximación $hat(f)$ directa sería
$
  hat(f)(x_0) = \#{x_i in cal(N)(x_0)} / (N times h)
$ #label("eps-nn")


donde $cal(N)$ es un vecindario métrico de $x_0$ de diámetro $h$.

Esta estimación es irregular, con saltos discretos en el numerador, por lo que se prefiere el estimador "suavizado por núcleos" de Parzen-Rosenblatt. Pero primero: ¿qué es un núcleo?


#defn([función núcleo o _kernel_])[

  Se dice que $K(x) : RR-> RR$ es una _función núcleo_ si  cumple que

  + toma valores reales no negativos: $K(u) >= 0$,
  + está normalizada: $integral K(u) d u = 1$,
  + es simétrica: $K(u) = K(-u)$ y
  + alcanza su máximo en el centro: $max_u K(u) = K(0)$
] <kernel>

#obs[Todas las funciones de densidad simétricas centradas en 0 son núcleos; en particular, la densidad "normal estándar" $phi(x) = 1/sqrt(2 pi) exp(-x^2 / 2)$ lo es.]

#obs[Si $K(u)$ es un núcleo, entonces $K_h (u) = 1/h op(K)(u / h)$ también lo es.]

Ahora sí estamos en condiciones de enunciar el
#defn("estimador de densidad por núcleos")[


  Sea $bu(x) = (x_1, dots, x_N)^T$ una muestra #iid de cierta variable aleatoria escalar $X in RR$ con función de densidad $f$. Su estimador de densidad por núcleos, o estimador de Parzen-Rosenblatt es
  $
    hat(f)(x_0) = 1/N sum_(i=1)^N K (x_0, x_i)
  $

  donde $K$ es una @kernel
] <parzen>

#obs[
  La densidad de la distribución uniforme centrada en 0 de diámetro 1, $U(x) = ind(1/2 < x <= 1/2)$ es un núcleo.  Luego, $ U_h (x) = 1/h ind(-h/2 < x < h/2) $ también es un núcleo válido, y por ende el estimador de @eps-nn es un caso particular del estimador de @parzen:
  $
    hat(f)(x_0) & = \#{x_i in cal(N)(x_0)} / (N times h) \
                & = 1 / N sum_(i in [N]) 1/ h thick U((x_i - x_0) / h) \
                & = 1 / N sum_(i = 1)^N U_h (x_i - x_0)
  $
]
=== Clasificador de densidad por núcleos

Si $hat(f)_k, k in [K]$ son estimadores de densidad por núcleos #footnote[KDEs ó _Kernel Density Estimators_, por sus siglas en inglés] de cada una de las $K$ densidades condicionales $X|GG_k$ según @parzen, podemos construir el siguiente clasificador

#defn(
  "clasificador de densidad por núcleos",
)[ Sean $hat(f)_1, dots, hat(f)_K$ estimadores de densidad por núcleos según @parzen. Sean además $hat(pi)_1, dots, hat(pi)_K$ las estimaciones de la probabilidad incondicional de pertenecer a cada grupo $GG_1, dots, GG_k$. Luego, la siguiente regla constituye un clasificador de densidad por núcleos:
  $
    hat(G)_"KD" (x) = g & = arg max_(i in [K]) hat(Pr)(GG_i | X = x) \
                        & = arg max_(i in [K]) hat(Pr)(X=x|GG_i) times hat(Pr)(GG_i) \
                        & = arg max_(i in [K]) hat(f)_i (x) times hat(pi)_i \
  $] <kdc-duro>

=== Clasificadores duros y suaves

Un clasificador que asigna a cada observación _una clase_ - la más probable, se suele llamar _clasificador duro_. Un clasificador que asigna a cada observación _una distribución de probabilidades de clase_ $hat(gamma)$ #footnote[ Todas las restricciones habituales aplican: dado $hat(gamma) = (hat(gamma)_1, dots, hat(gamma)_K)^T$, todas sus componentes deben pertenecer al intervalo $[0, 1]$ y su suma ser exactamente $1$.] se suele llamar _clasificador blando_. Dado un clasificador _blando_ $hat(G)_"Blando"$, es trivial construir el clasificador duro asociado $hat(G)_"Duro"$:
$
  hat(G)_"Duro" (x_0) = arg max_i hat(G)_"Blando" (x_0) = arg max_i hat(gamma)_i
$

#obs[
  El clasificador de de @kdc-duro es en realidad la versión dura de un clasificador blando $hat(G)_"KD" (x) = hat(gamma)$, donde $ hat(gamma)_i = (hat(f)_i (x) times hat(pi)_i) / (sum_(i in [K]) hat(f)_i (x) times hat(pi)_i) $
]

#obs[
  Algunos clasificadores sólo pueden ser duros, como $hat(G)_"1-NN"$, el clasificador de @knn-clf con $k=1$.
]

Dos clasificadores _blandos_ pueden tener la misma pérdida $0-1$, pero "pintar" dos panoramas muy distintos respecto a cuán "seguros" están de cierta clasificación. Por caso,
$
  hat(G)_"C(onfiado)" (x_0) &: hat(Pr)(GG_i | X = x_0) = cases(1 - epsilon times (K - 1) &" si " i = 1, epsilon &" si " i != 1) \
  hat(G)_"D(udoso)" (x_0) &: hat(Pr)(GG_i | X = x_0) = cases(1/K + epsilon times (K - 1) &" si " i = 1, 1 / K - epsilon &" si " i != 1)
$
$hat(G)_C$ está "casi seguro" de que la clase correcta es $GG_1$, mientras que $hat(G)_D$ está prácticamente indeciso entre todas las clases. Para el entrenamiento y análisis de clasificadores blandos como el de densidad por núcleos, será relevanta encontrar funciones de pérdida que recompensen y penalicen adecuadamente esta capacidad.

== Estimación de densidad multivariada
=== Naive Bayes
Una manera "ingenua" de adaptar el procedimiento de estimación de densidad ya mencionado a $X$ multivariadas, consiste en sostener el desde-luego-falso-pero-útil supuesto de que sus componentes $X_1, dots, X_p$ son independientes entre sí. De este modo, la estimación de densidad conjunta se reduce a la estimación de $p$ densidades marginales univariadas. Dada cierta clase $j$, podemos escribir la densidad condicional $X|j$ como
$
  f_j (X) = product_(k = 1)^p f_(j k) (X_k)
$ <naive-bayes>

A este procedimiento se lo conoce como "Naive Bayes", y a pesar de su aparente ingenuidad es competitivo contra algoritmos mucho más sofisticados en un amplio rango de tareas. En términos de cómputo, permite resolver la estimación con $K times p$ KDE univariados. Además, permite que en $X$ se combinen variables cuantitativas y cualitativas: basta con reemplazar la estimación de densidad para los $X_k$ cualitativos por su correspondiente histograma.

=== KDE multivariado

#figure(caption: flex-caption(
  "Dos círculos concéntricos y sus KDE marginales por clase: a pesar de que la frontera entre ambos grupos de puntos es muy clara, es casi imposible disinguirlas a partir de sus densidades marginales.",
  "Dos círculos concéntricos",
))[#image("img/dos-circulos-jointplot.png", width: 75%)]

En casos como este, el procedimiento de Naive Bayes falla miserablemente, y será necesario adaptar el procedimiento de KDE unidimensional a $d >= 2$ sin basarnos en el supuesto de independencia de las $X_1, dots, X_k$. A lo largo de las cuatro décadas posteriores a las publicaciones de Parzen y Rosenblatt, el estudio de los estimadores de densidad por núcleos avanzó considerablemente, de manera que ya para mediados de los '90 existen minuciosos libros de referencia como "Kernel Smoothing" @wandKernelSmoothing1995, que seguiremos en la presente sección

#defn([KDE multivariada, @wandKernelSmoothing1995[§4]])[
  En su forma más general, estimador de densidad por núcleos #box[$d-$ variado] es

  $
    hat(f) (x; HH) = N^(-1) sum_(i=1)^N K_H (x - x_i)
  $

  donde
  - $HH in RR^(d times d)$ es una matriz simétrica def. pos. análoga a la ventana $h in RR$ para $d=1$,
  - $KH(t) = abs(det HH)^(-1/2) K(HH^(-1/2) t)$
  - $K$ es una función núcleo $d$-variada tal que $integral K(bu(x)) d bu(x) = 1$
] <kde-mv>

Típicamente, K es la densidad normal multivariada
$
  Phi(x) : RR^d -> RR = (2 pi)^(-d/2) exp(- (||x||^2)/2)
$

=== La elección de $HH$
Sean las clases de matrices $RR^(d times d)$ ...
- $cal(F)$, de matrices simétricas definidas positivas,
- $cal(D)$, de matrices diagonales definidas positivas ($cal(D) subset.eq cal(F)$) y
- $cal(S)$, de múltiplos escalares de la identidad: $cal(S) = {h^2 bu(I):h >0} subset.eq cal(D)$


Aún tomando una única $HH$ para _toda_ la muestra, $HH in dots$, la elección de $HH$ en dimensión $d$ requiere definir...
- $mat(d; 2) = (d^2 - d) slash 2$ parámetros de ventana si  $HH in cal(F)$,
- $d$ parámetros si $HH in cal(D)$ y
- un único parámetro $h$ si $HH = h^2 bu(I)$.

La evaluación de la conveniencia relativa de cada parametrización se vuelve muy compleja, muy rápido. @wandComparisonSmoothingParameterizations1993 proveen un análisis detallado para el caso $d = 2$, y concluyen que aunque cada caso amerita su propio estudio, $HH in cal(D)$ suele ser "adecuado". Sin embargo, este no es un gran consuelo para valores de $d$ verdaderamente altos, en cuyo caso existe aún un problema más fundamental.

=== La maldición de la dimensionalidad

Uno estaría perdonado por suponer que el problema de estimar densidades en alta dimensión se resuelve con una buena elección de $HH$, y una muestra "lo suficientemente grande". Considérese, sin embargo, el siguiente ejercicio, adaptado de  para ilustrar ese "suficientemente grande":

#quote(attribution: [adaptado de @wandKernelSmoothing1995[§4.9 ej 4.1]])[
  Sean $X_i tilde.op^("iid")"Uniforme"([-1, 1]^d), thick i in [N]$, y consideremos la estimación de la densidad en el origen, $hat(f)(bu(0))$. Suponga que el núcleo $K_(HH)$ es un "núcleo producto" basado en la distribución univariada $"Uniforme"(-1, 1)$, y $HH = h^2 bu(I)$. Derive una expresión para la proporción esperada de puntos incluidos dentro del soporte del núcleo $KH$ para $h, d$. arbitrarios.
]

El "núcleo producto" multivariado basado en la ley $"Uniforme(-1, 1)"$ evaluado alrededor del origen es:
$
  K(x - 0)= K(x) = product_(i = 1)^d ind(-1 <= x_i <= 1) = ind(inter.big_(i=1)^d thick abs(x_i) <= 1) \
$
De la @kde-mv y el hecho de que $det HH = h^(2d); thick HH^(-1/2) = h^(-1) bu(I)$, se sigue que
$
  KH(x) & = abs(h^(2d))^(-1/2) K(h^(-1)bu(I) x) = h^(-d) K(x/h) \
        & = h^(-d) ind(inter.big_(i=1)^d thick abs(x_i / h) <= 1) = h^(-d) ind(inter_(i=1)^d thick abs(x_i) <= h) \
        & = h^(-d) ind(x in [-h, h]^d)
$
De modo que $sop KH = [-h, h]^d$, y ahora nos resta encontrar la esperanza. Como las componentes de una ley uniforme multivariada son independientes entre sí,
$
  Pr(X in [-h, h]^d) & = product_(i=1)^d Pr(X_i in [-h, h]) \
                     & = Pr(-h <= X_1 <= h])^d \
                     & = [(h - (-h))/(1-(-1))]^d = h^d quad square.stroked
$

#let h = 0.5
#let d = 20

Para $h =#h, d=#d, thick Pr(X in [-#h,#h]^#d) = #h^(-#d) approx #calc.round(calc.pow(h, d), digits: 8)$, ¡menos de uno en un millón! En general, la caída es muy rápida, aún para valores altos de $h$. Si $X$ representa un segundo de audio respete el estandar _mínimo_ de llamadas telefónicas  #footnote[De Wikipedia: La tasa #link("https://en.wikipedia.org/wiki/Digital_Signal_0")[DS0], o _Digital Signal 0_, fue introducida para transportar una sola llamada de voz "digitizada". La típica llamada de audio se digitiza a $8 "kHz"$, o a razón de 8.000 veces por segundo. se]
#image("img/curse-dim.png")tiene $d=8000$.
En tal espacio ambiente, aún con $h=0.999$,
$Pr(dot) approx #calc.round(calc.pow(0.999, 8000), digits: 6)$, o 1:3.000.

=== La hipótesis de la variedad ("manifold hypothesis")

Ahora, si el espacio está _tan_, pero _tan_ vacío en alta dimensión, ¿cómo es que el aprendizaje supervisado _sirve de algo_? La reciente explosión en capacidades y herramientas de procesamiento (¡y generación!) de formatos de altísima dimensión #footnote[audio, video, texto y data genómica por citar sólo algunos] pareciera ser prueba fehaciente de que la tan mentada _maldición de la dimensionalidad_ no es más que un cuento de viejas.

Pues bien, el ejemplo del segundo de audio antedicho _es_ sesgado, ya que simplemente no es cierto que si $X$ representa $1s$ de voz humana , su ley sea uniforme 8000 dimensiones #footnote[El audio se digitiza usando 8 bits para cada muestra, así que más precisamente, $sop X = [2^8]^8000$ o $64 "kbps"$, kilobits-por-segundo.]: si uno muestreara un segundo de audio siguiendo cualquier distribución en la que muestras consecutivas no tengan ninguna correlación, obtiene #link("https://es.wikipedia.org/wiki/Ruido_blanco")[_ruido blanco_]. La voz humana, por su parte, tiene _estructura_, y por ende correlación instante-a-instante. Cada voz tiene un _timbre_ característico, y las palabras enuncidas posibles están ceñidas por la _estructura fonológica_ de la lengua locutada.

Sin precisar detalles, podríamos postular que las realizaciones de la variable de interés $X$ (el habla), que registramos en un soporte $cal(S) subset.eq RR^d$ de alta dimensión, en realidad se concentran en cierta _variedad_ #footnote[Término que ya precisaremos. Por ahora, $cal(M)$ es el _subespacio de realizaciones posibles_ de $X$] $cal(M) subset.eq cal(S)$ sobre , potencialmente de mucha menor dimensión $dim (M) = d_cal(M) << d$, en el la noción de distancia entre observaciones aún conserva significado. A tal postulado se lo conoce como "la hipótesis de la variedad", o _manifold hypothesis_. <hipotesis-variedad>

La hipótesis de la variedad no es exactamente una hipótesis contrastable en el sentido tradicion al del método científico; de hecho, ni siquiera resulta obvio que de existir, sean bien definibles las variedades en las que existen los elementos del mundo real: un dígito manuscrito, el canto de un pájaro, o una flor. Y de existir, es de esperar que sean altamente #box[no-lineales].

#figure(caption: flex-caption(
  [Ejemplos de variedades en el mundo físico: tanto la hoja de un árbol como una bandera flameando al viento tienen dimensión intrínseca $d_cal(M) = 2$, están embedidas en $RR^3$, y son definitivamente no-lineales.],
  "Ejemplos de variedades en el mundo físico",
))[
  #columns(2, [
    #image("img/hormiga-petalo.jpg")
    #colbreak()
    #image("img/bandera-argentina.png")
  ])
]


Más bien, corresponde entenderla como un modelo mental, que nos permite aventurar ciertas líneas prácticas de trabajo en alta dimensión #footnote[TODO: @galleseRootsEmpathyShared2003 : shared manifold hypothesis y @bengioConsciousnessPrior2019]. Pero antes de profundizar en esta línea, debemos platearnos algunas preguntas básicas:

#align(center)[
  \
  ¿Qué es, exactamente, una variedad? \ \
  ¿Es posible construir un KDE con soporte en cierta variedad _conocida_? \ \
  ¿Sirve de algo todo esto si _no conocemos_ la variedad en cuestión?
]

== Variedades de Riemann

Adelantando la respuesta a la segunda pregunta, resulta ser que si el soporte de $X$ es una "variedad de Riemann", bajo ciertas condiciones razonables sí es posible estimar su densidad por núcleos en la variedad @pelletierKernelDensityEstimation2005.

A continuación, damos un recorrido sumario e idiosincrático por ciertos conceptos básicos de topología y variedades que consideramos necesarios para motivar la definición de variedades Riemannianas, que de paso precisarán la respuesta a la primer pregunta - ¿qué es una variedad? - en el contexto que nos interesa. A tal fin, seguimos la exposición de la monografía _Estimación no paramétrica de la densidad en variedades Riemannianas_ @munozEstimacionNoParametrica2011, que a su vez sigue, entre otros, el clásico _Introduction to Riemannian Manifolds_ @leeIntroductionRiemannianManifolds2018.

=== Variedades Diferenciables

#defn([espacio topológico (TODO: ARROBA CITA WIKIPEDIA)])[

  Formalmente, se llama *espacio topológico* al par ordenado $(X, T)$ formado por un conjunto $X$ y una _topología_ $T$ sobre $X$, es decir una colección de subconjuntos de $X$ que cumple las siguientes tres propiedades:
  + El conjunto vacío y $X$ están en T: $emptyset in T, X in T$
  + La intersección de cualquier subcolección _finita_ de $T$ está en $T$:
  $ X in T, Y in T => X inter Y in T $La unión de _cualquier_ subcolección de conjuntos de $T$
  está en $T$:
  $
    forall S subset T, thick union.big_(O in S) O in T
  $
]
A los conjuntos pertenecientes a la topología $T$ se les llama conjuntos abiertos o simplemente abiertos de $(X, T)$; y a sus complementos en $X$, conjuntos cerrados.

#defn([entorno (TODO arroba wikipedia)])[
  Si $(X,Τ)$ es un espacio topológico y $p$ es un punto perteneciente a X, un _entorno_ #footnote[ También se los conoce como "vecindarios" - por _neighborhoods_, su nombre en inglés.] de $p$ es un conjunto $V$ en el que está contenido un conjunto abierto $U$ que tiene como elemento al punto $p$: $ p in U subset.eq V $.
]
#defn([espacio de Hausdorff (TODO: ARROBA CITA WIKIPEDIA)])[

  Sea $(X, T)$ un espacio topológico. Se dice que dos puntos $p, q in X$ cumplen la propiedad de Hausdorff si existen dos entornos $U_p$ de $p$ y $U_q$ de $q$ tales que $U_p inter U_q = emptyset$ (i.e., son disjuntos).

  Se dice que un espacio topológico es un espacio de Hausdorff #footnote[o que verifica la propiedad de Hausdorff, o que es separado o que es $bu(T_2)$] si todo par de puntos distintos del espacio verifican la propiedad de Hausdorff.
]
En términos coloquiales, un espacio de Hausdorff es aquél donde todos sus puntos están "bien separados".

#defn(
  [variedad topológica @munozEstimacionNoParametrica2011[Def. 3.1.1], @leeIntroductionRiemannianManifolds2018[Apéndice A]],
)[
  Una variedad topológica de dimensión $d in NN$ es un espacio topológico $(MM, T)$ de Hausdorff, de base numerable, que es #strong[localmente homeomorfo a $RR^d$]. Es decir, para cada $p in MM$ existe un abierto $U in T$ y un abierto $A subset.eq RR^d$, tal que $p in U$ ($U$ es un entorno de $p$) y existe un homemorfismo $#sym.phi : U -> A$.
]

En una variedad topológica, cobra sentido cierto concepto de cercanía - pero no necesariamente de _distancia_, y es posible definir funciones continuas y límites.

Un _homeomorfismo_ #footnote[del griego _homo-_: igual, _-morfo_: forma; de igual forma] es una función #sym.phi entre dos espacios topológicos si es biyectiva y tanto ella como su inversa son continuas. El par ordenado $(U, #sym.phi)$ es una _carta #footnote[_chart_ en inglés] alrededor de $p$_.

A un conjunto numerable de tales cartas que cubran completamente la variedad se lo denomina "atlas". Simbólicamente, #box[$cal(A) = {(U_alpha, #sym.phi _alpha) : alpha in cal(I)}$] es un atlas sí y sólo si $MM = union_alpha U_alpha$.

Cuando un homeomorfismo - y su inversa - es $r-$veces diferenciable, se le llama _$C^r$-difeomorfismo_, o simplemente difeomorfismo #footnote[Luego, un homeomorfismo es un $C^0-$difeomorfismo]. En particular, un $C^oo-$difeomorfismo es un difeomorfismo _suave_.

Sean $(MM, T)$ una variedad topolóogica de dimensión $d$ y sean $(U, #sym.phi), (V, psi)$ dos cartas. Diremos que son _suavemente compatibles_ #footnote[_smoothly compatible_ según @leeIntroductionRiemannianManifolds2018[ § "Smooth Manifolds and Smooth Maps"]. @munozEstimacionNoParametrica2011 lo denomina _compatible_ a secas.] si $U inter V = emptyset$ o bien si la función cambio de coordenadas restringida a $U inter V$ es un difeormorfismo.

La compatibilidad requiere que la transición entre mapas no sea sólo continua, sino también _suave_. El motivo de esta condición es asegurar que el concepto de _suavidad_ esté bien definido en toda la variedad $MM$, independientemente de qué carta se use: si una función es diferenciable vista a través de una carta, también lo será al analizarla desde cualquier carta compatible.

#defn([estructura diferenciable @munozEstimacionNoParametrica2011[Def. 3.1.3]])[
  Un atlas $cal(A) = {(U_alpha, #sym.phi _alpha) : alpha in cal(I)}$ es diferenciable si sus cartas son compatibles entre sí. Si un atlas diferenciable $cal(D)$ es _maximal_ lo llamaremos una _estructura diferenciable de la variedad $MM$ _. Con maximal queremos decir lo siguiente: Si $(U, #sym.phi)$ es una carta de $MM$ que es compatible con todas las cartas de $cal(D)$, entonces $(U, #sym.phi) in cal(D)$ #footnote[i.e., no existe otro atlas diferenciable que contenga propiamente a $cal(D)$, lo cual desambigua la referencia.] 
]
Definici ́ on 3.1.4. Una variedad diferenciable de dimensi ́on d es una terna (M, τ, D) donde (M, τ ) es una variedad topol ́ ogica de dimensi ́on d y D una estructura diferenciable.


#grid(
  columns: (80%, 20%),
  [
    Sea $T_p cal(M)$ el _espacio tangente_ a un punto $p in cal(M)$, y $g_p : T_p cal(M) times T_p cal(M) -> RR$ una forma _bilinear pos. def._ para cada $p$ que induce una _norma_ $||v||_p= sqrt(op(g_p)(v, v))$.

    Decimos entonces que $g_p$ es una métrica Riemanniana y el par $(cal(M), g)$ es una variedad de Riemann, donde las nociones de _distancia, ángulo y geodésica_ están bien definidas.],
  image("img/Tangent_plane_to_sphere_with_vectors.svg"),
)

=== KDE en variedades de Riemann [Pelletier 2005]
@pelletierKernelDensityEstimation2005
- Sea $(cal(M), g)$ una variedad de Riemann compacta y sin frontera de dimensión $d$, y usemos $d_g$ para denotar la distancia de Riemann.
- Sea $K$ un _núcleo isotrópico en $cal(M)$ soportado en la bola unitaria_ (cf. conds. (i)-(v))
- Sean $p, q in cal(M)$, y $theta_p (q)$ la _función de densidad de volumen en $cal(M)$_ #footnote[¡Ardua definición! Algo así como el cociente entre las medida de volumen en $cal(M)$, y su transformación via el mapa local a $RR^d$]
Luego, el estimador de densidad para $X_i tilde.op^("iid")f$ es $f_(N,K):cal(M) ->RR$ que a cada $p in cal(M)$ le asocia el valor
$
  f_(N,K) (p) = N^(-1) sum_(i=1)^N K_h (p,X_i) = N^(-1) sum_(i=1)^N 1/h^d 1/(theta_X_i (p))K((op(d_g)(p, X_i))/h)
$

con la restricción de que la ventana $h <= h_0 <= op("inj")(cal(M))$, el _radio de inyectividad_ de $cal(M)$ #footnote[el ínfimo entre el supremo del radio de una bola en cada $p$ tal que su mapa es un difeomorfismo]

=== Algunas variedades conocidas
$S^1$: @vonmisesUberGanzzahligkeitAtomgewicht1918
$S^2$: @fisherDispersionSphere1957
$S^p$: @mardiaDistributionTheoryMisesFisher1975
=== Interludio: densidad de volumen en la esfera [Henry y Rodríguez, 2009]

#columns(2)[
  En _"Kernel Density Estimation on Riemannian Manifolds: Asymptotic Results" (2009)_, Guillermo Henry y Daniela Rodriguez estudian algunas propiedades asintótica de este estimador, y las ejemplifican con datos de sitios volcánicos en la superficie terrestre.
  En particular, calculan la densidad de volumen $theta_p(q)$
  #image("img/densidad-volumen-esfera.png")
  #colbreak()
  #image("img/henry-rodriguez-bolas.png")
]
== Clasificación en variedades
@vincentManifoldParzenWindows2002
=== Clasificación en variedades [Loubes y Pelletier 2008]
@loubesKernelbasedClassifierRiemannian2008
@hallBandwidthChoiceNonparametric2005 h optimo para clasificacion con KDE
@devroyeProbabilisticTheoryPattern1996[§6 Consistencia]

¡Clasificador de Bayes + KDE en Variedades = Clasificación (suave o dura) en variedades!

Plantean una regla de clasificación $hat(G)$ para 2 clases adaptable a K clases de forma directa. Sea $p in cal(M)$ una variedad riemanniana como antes, y ${(x_1, g_1), dots, (x_N, g_N)}$ nuestras observaciones y sus clases. Luego,

$
  hat(G) (p) = arg max_(g in GG) sum_(i=1)^N ind(g_i = g)K_h (p,X_i)
$


#align(center)[Pero... ¿y si la variedad es desconocida?]

== Aprendizaje de distancias
#figure(
  caption: flex-caption[La variedad $cal(U)$ con $dim(cal(U)) = 1$ embebida en $RR^2$. Nótese que en el espacio ambiente, el punto rojo está más cerca del verde, mientras que a través de $cal(U)$, el punto amarillo está más próximo que el rojo][Variedad $cal(U)$],
)[#image("img/variedad-u.svg", width: 70%)]

En un extenso censo del campo de _aprendizaje de representaciones_, Bengio et ales la asocian directamente al campo de _aprendizaje de representaciones_:


#quote(attribution: [ @bengioRepresentationLearningReview2014[§8]])[
  (...) [L]a principal tarea del aprendizaje no-supervisado se considera entonces como el modelado de la estructura de la variedad que sustenta los datos. La representación asociada que se aprende puede asociarse con un sistema de coordenadas intrínseco en la variedad embebida. El algoritmo arquetípico de modelado de variedades es, como era de esperar, también el algoritmo arquetípico de aprendizaje de representaciones de baja dimensión: el Análisis de Componentes Principales, PCA.
]
El concepto, aunque no figure con ese nombre hasta principios de este siglo, existe desde mucho más atrás #footnote[estas referencias vienen del mismo Bengio #link("https://www.reddit.com/r/MachineLearning/comments/mzjshl/comment/gwq8szw/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button")[comentando en Reddit sobre el origen del término]].
@rifaiManifoldTangentClassifier2011
@caytonAlgorithmsManifoldLearning2005: PCA, SOMs, Isomap, etc.

=== El ejemplo canónica: Análisis de Componentes Principales (PCA)

#align(center)[#image("img/pca.png")]
#text(
  size: 12pt,
)[@pearsonLIIILinesPlanes1901, _"LIII. On lines and planes of closest fit to systems of points in space."_]


=== El algoritmo más _cool_: Isomap
==== previo: self-organizing mapas
@kohonenSelforganizedFormationTopologically1982
@kohonenSelfOrganizationAssociativeMemory1988
==== isometric feature mapping
@tenenbaumMappingManifoldPerceptual1997
@tenenbaumGlobalGeometricFramework2000

#grid(
  columns: (35%, 65%),
  column-gutter: 20pt,
  [
    1. Construya el grafo de $k, epsilon$-vecinos, $bu(N N)=(bu(X), E)$

    2. Compute los caminos mínimos - las geodésicas entre observaciones, $d_(bu(N N))(x, y)$.

    3. Construya una representación ("_embedding"_) $d^*$−dimensional que minimice la discrepancia ("stress") entre $d_(bu(N N))$ y la distancia euclídea en $RR^d^*$
  ],
  image("img/isomap-2.png"),
)
[Tenenbaum et al (2000), _"A Global Geometric Framework for Nonlinear Dimensionality Reduction"_]

=== Distancias basadas en densidad
- Bijral @bijralSemisupervisedLearningDensity2012
- @brandChartingManifold2002
@chuExactComputationManifold2019
@vincentDensitySensitiveMetrics2003
=== Distancia de Fermat [Groisman, Jonckheere, Sapienza (2019); Little et al (2021)]
@groismanNonhomogeneousEuclideanFirstpassage2019
@littleBalancingGeometryDensity2021
@mckenziePowerWeightedShortest2019
#defn("Distancia Muestral de Fermat")[]<sample-fermat-distance>

#quote(attribution: "P. Groisman et al (2019)")[
  #set text(size: 12pt)
  _We tackle the problem of learning a distance between points, able to capture both the geometry of the manifold and the underlying density. We define such a sample distance and prove the convergence, as the sample size goes to infinity, to a macroscopic one that we call Fermat distance as it minimizes a path functional, resembling Fermat principle in optics._]

Sea $f$ una función continua y positiva, $beta >=0$
y $x, y in S subset.eq RR^d$. Definimos la _Distancia de Fermat_ $cal(D)_(f, beta)(x, y)$ como:

$
  cal(T)_(f, beta)(gamma) = integral_gamma f^(-beta) space, quad quad quad cal(D)_(f, beta)(x, y) = inf_(gamma in Gamma) cal(T)_(f, beta)(gamma) quad #emoji.face.shock
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
donde $beta = (a-1) slash d, thick n >= n_0$ y $mu$ es una constante adecuada.


¡Esta sí la podemos aprender de los datos! #emoji.arm.muscle

== Propuesta Original
== Todo junto

Habiendo andado este sendero teórico, la pregunta natural que asoma es: ¿es posible mejorar un algoritmo de clasificación reemplazando la distancia euclídea por una aprendida de los datos, como la de Fermat? Para investigar la cuestión, nos propusimos:
1. Implementar un clasificador basado en estimación de densidad por núcleos (TODO: ref) según @loubesKernelbasedClassifierRiemannian2008, que llamaremos "KDC". Además,
2. Implementar un estimador de densidad por núcleos basado en la distancia de Fermat, a fines de poder comparar la _performance_ de KDC con distancia euclídea y de Fermat.

Nótese que el clasificador enunciado al inicio (k-NN, @eps-nn), tiene un pariente cercano, $epsilon-upright("NN")$
#defn("k-vecinos más cercanos")[] <knn>

@eps-nn es esencialmente equivalente a KDC con un núcleo "rectangular", $k(t) = ind(d(x, t) < epsilon) / epsilon$, pero su definición es considerablemente más sencilla. Luego, propondremos también
3. Implementar un clasificador cual @knn, pero con distancia muestral de Fermat en lugar de euclídea.

=== KDC con Distancia de Fermat Muestral

=== f-KNN

=== Algunas dudas

- Entrenar el clasificador por validación cruzada está OK: como $bu(X)_"train" subset.eq bu(X)$ y $bu(X)_"test" subset.eq bu(X)$, se sigue que $forall (a, b) in {bu(X)_"train" times in bu(X)_"test"} subset.eq {bu(X) times bu(X)}$ y $D_(bu(X), alpha) (a, b)$ está bien definida.  ¿Cómo sé la distancia _muestral_ de una _nueva_ observación $x_0$, a los elementos de cada clase?\


Para cada una de las $g_i in GG$ clases, definimos el conjunto $ Q_i= {x_0} union {x_j : x_j in bu(X), g_j = g_i, j in {1, dots, N}} $
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
3. Evaluamos el _pseudo-$R^2$_ y la _exactitud_ ("accuracy") de los clasificadores propuestos en diferentes _datasets_, relativa a técnicas bien establecidas:
#columns(2)[
  - regresión logística (`LR`)
  - clasificador de  soporte vectorial (`SVC`) #footnote[sólo se consideró su exactitud. ya que no es un clasificador suave]
  - _gradient boosting trees_ (`GBT`)
  #colbreak()
  - k-vecinos-más-cercanos (`KN`)
  - Naive Bayes Gaussiano (`GNB`)
]


- La implementación de `KNeighbors` de referencia acepta distancias precomputadas, así que incluimos una versión con distancia de Fermat, que apodamos `F(ermat)KN`.

- Para ser "justos", se reservó una porción de los datos para la evaluación comparada, y del resto, cada algoritmo fue entrenado repetidas veces por validación cruzada de 5 pliegos, en una extensísima grilla de hiperparametrizaciones. Este procedimiento *se repitió 25 veces en cada dataset*.

- La función de score elegida fue `neg_log_loss` ($= cal(l)$) para los clasificadores suaves, y `accuracy` para los duros.

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


== Evaluación

Nos interesa conocer en qué circunstancias, si es que hay alguna, la distancia muestral de Fermat provee ventajas a la hora de clasificar por sobre la distancia euclídea. Además, en caso de existir, quisiéramos en la medida de lo posible comprender por qué (o por qué no) es que tal ventaja existe.
A nuestro entender resulta imposible hacer declaraciones demasiado generales al respecto de la capacidad del clasificador: la cantidad de _datasets_ posibles, junto con sus _configuraciones de evaluación_ es tan densamente infinita como lo permita la imaginación del evaluador. Con un ánimo exploratorio, nos proponemos explorar la _performance_ de nuestros clasificadores basados en distancia muestral de Fermat en algunas _tareas_ puntuales.

=== Métricas de _performance_

En tareas de clasificación, la métrica más habitual es la _exactitud_ #footnote([Más conocida por su nombre en inglés, _accuracy_.])

#let clfh = $op(upright(R))$
#let clfs = $op(cal(R))$

#defn(
  "exactitud",
)[Sean $bu(("X, y")) in RR^(n times p) times RR^n$ una matriz de $n$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(y)) = clfh(bu(X))$ las predicciones de clase resultado de una regla de clasificación #clfh. La _exactitud_ ($"exac"$) de #clfh en #bu("X") se define como la proporción de coincidencias con las clases verdaderas #bu("y"):
  $ op("exac")(clfh | bu(X)) = n^(-1) sum_(i=1)^n ind(hat(y)_i = y_i) $
] <exactitud>

La exactitud está bien definida para cualquier clasificador que provea una regla _dura_ de clasificación, segun clasi. Ahora bien, cuando un clasificador provee una regla _suave_ (clasificador-suave), la exactitud como métrica "pierde información": dos clasificadores binarios que asignen respectivamente 0.51 y 1.0 de probabilidad de pertenecer a la clase correcta a todas als observaciones tendrán la misma exactitud, $100%$, aunque el segundo es a las claras mejor. A la inversa, cuando un clasificador erra al asignar la clase: ¿lo hace con absoluta confianza, asignando una alta probabilidad a la clase equivocada, o con cierta incertidumbre, repartiendo la masa de probabilidad entre varias clases que considera factibles?

Una métrica natural para evaluar una regla de clasificación suave, es la _verosimilitud_ (y su logaritmo) de las predicciones.

#defn(
  "verosimilitud",
)[Sean $bu(("X, y")) in RR^(n times p) times RR^n$ una matriz de $n$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(Y)) = clfs(bu(X)) in RR^(n times k)$ la matriz de probabilidades de clase resultado de una regla suave de clasificación #clfs. La _verosimilitud_ ($"vero"$) de #clfs en #bu("X") se define como la probabilidad conjunta que asigna #clfs a las clases verdaderas #bu("y"):
  $
    op(L)(clfs) = op("vero")(
      clfs | bu(X)
    ) = Pr(hat(bu(y)) = bu(y)) = product_(i=1)^n Pr(hat(y)_i =y_i) = product_(i=1)^n hat(bu(Y))_((i, y_i))
  $

  Por conveniencia, se suele considerar la _log-verosimilitud promedio_,
  $ op(cal(l))(clfs) = n^(-1) log(op("L")(clfs)) = n^(-1)sum_(i=1)^n log(hat(bu(Y))_((i, y_i))) $
] <vero>

La verosimilitud de una muestra varía en $[0, 1]$ y su log-verosimilitud, en $(-oo, 0]$, pero como métrica esta sólo se vuelve comprensible _relativa a otros clasificadores_. Una forma de "normalizar" la log-verosimilitud, se debe a @mcfaddenConditionalLogitAnalysis1974.

#defn(
  [$R^2$ de McFadden],
)[Sea $clfs_0$ el clasificador "nulo", que asigna a cada observación y posible clase, la frecuencia empírica de clase encontrada en la muestra de entrenamiento $bu(X)_("train")$. Para todo clasificador suave $clfs$, definimos el $R^2$ de McFadden como
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
- Naive Bayes Gaussiano (gnb, #gnb),
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
- un _dataset_ con el conjunto de $n$ observaciones en $d_x$ dimensiones repartidas en $k$ clases, ${bu(X)}_n in R^(n times d_x), {bold(y)}_n in [k]^n$,
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

#defn(
  "ruido blanco",
)[Sea $X = (X_1, dots, X_d) in RR^d$ una variable aleatoria tal que $"E"(X_i)=0, "Var"(X_i)=sigma thick forall i in [d]$. Llamaremos "ruido blanco con escala $sigma$" a toda realización de $X$.] <ruido-blanco>

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
KDC (en sus dos variantes), KNN y SVC (con kernel RBF) parecieran ser los métodos más competitivos, con mínimas diferencias de performance entre sí: sólo en "círculos" se observa un ligero ordenamiento de los métodos, $svc succ kdc succ knn$, aunque la performance mediana de #svc está dentro de "los bigotes" de todos los métodos antedichos. La tarea "lunas" pareciera ser la más fácil de todas, en la que hasta una regresión logística sin modelado alguno es adecuada. Para "espirales" y "círculos", #gnb, #lr y #lsvc no logran performar significativamente mejor que el clasificador base.

#defn("clasificador base")[] <clf-base>

¿Cómo se comparan los métodos en términos de la log-verosimilitud y el $R^2$?

#figure(
  image("img/boxplot-r2-lunas-espirales-circulos.svg", width: 120%),
  caption: flex-caption[Boxplots con la distribución de $R^2$ en las #reps repeticiones de cada experimento.][Boxplots para $R^2$ de lunas-circulos-espirales],
)
#figure(tabla_csv("data/r2-ds-2d.csv"), caption: flex-caption[ "mi caption, bo-bo". ][])

Como los métodos basados en máquinas de soporte vectorial resultan en clasificadores _duros_ (clasificador-duro), no es posible analizar la log-verosimilitud u otras métricas derivadas. De entre los dos métodos con exactitud similar a esos, es notoriamente mejor el $R^2$ que alcanzan ambos #kdc.
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
  for key in from.keys() { d.insert(key, calc.round(float(from.at(key)), digits: 4)) }
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


=== 2D, 2 clases: excelente $R^2$ con exactitud competitiva

=== Con Bajo Ruido
#align(center)[#image("img/2d-lo-datasets.png")]
#columns(3)[
  #image("img/lunas_lo-overall.png")
  #colbreak()
  #image("img/circulos_lo-overall.png")
  #colbreak()
  #image("img/espirales_lo-overall.png")

]
=== Boxplot Accuracy
#align(center)[#image("img/2d-lo-acc.png")]
=== Boxplot $R^2$
#align(center)[#image("img/2d-lo-r2.png")]

=== Superposición de parámetros: $alpha$ y $h$


- El uso de la distancia de Fermat muestral no hiere la performance, pero las mejoras son nulas o marginales. ¿Por qué?


Si recordamos $hat(f)_(K,N)$ según Loubes & Pelletier, al núcleo $K$ se lo evalúa sobre
$
  (d (x_0, X_i)) / h, quad d = D_(Q_i, alpha)
$

Lo que $alpha$ afecta a $hat(f)$ vía $d$, también se puede conseguir vía $h$.

Si $D_(Q_i, alpha) prop ||dot||$ (la distancia de fermat es proporcional a la euclídea), los efectos de $alpha$ y $h$ se "solapan"


... y sabemos que localmente, eso es cierto #emoji.face.tear

=== Parámetros óptimos para $"(F)KDC"$ en `espirales_lo`
#align(center)[#image("img/optimos-espirales_lo.png")]


=== Superficies (o paisajes) de _score_ para `(espirales_lo, 1434)`

#align(center)[#image("img/heatmap-fkdc-2d-lo-new.svg")]

=== Alt-viz: Perfiles de pérdida para `(espirales_lo, 1434)`

#align(center)[#image("img/perfiles-perdida-espirales-1434.png")]

=== Fronteras de decisión para `(espirales_lo, 1434)`

#align(center)[#image("img/gbt-lr-espirales.png")]
#align(center)[#image("img/kn-espirales.png")]
#align(center)[#image("img/kdc-espirales.png")]
#align(center)[#image("img/gnb-svc-espirales.png")]

==== Efecto del ruido en la performance de clasificación
#columns(3)[
  #image("img/lunas_hi-overall.png")
  #colbreak()
  #image("img/circulos_hi-overall.png")
  #colbreak()
  #image("img/espirales_hi-overall.png")
]

=== 3D, 2 clases + piononos

#align(center)[#image("img/3d.png")]
#align(center)[#image("img/pionono.png")]
#columns(4)[
  #image("img/pionono_0-overall.png")
  #colbreak()
  #image("img/eslabones_0-overall.png")
  #colbreak()
  #image("img/helices_0-overall.png")
  #colbreak()
  #image("img/hueveras_0-overall.png")
]
#align(center)[#image("img/pionono-eslabones-r2.png")]
#align(center)[#image("img/helices-hueveras-r2.png")]

=== Parámetros óptimos para $"(F)KDC"$ en `helices_0`
#align(center)[#image("img/optimos-helices_0.png")]

=== Microindiferencia, macrodiferencia

- En zonas con muchas observaciones (por tener alta $f$ o alto $N$) sampleadas, la distancia de Fermat y la euclídea coinciden.
- "Localmente", siempre van a coincidir, aunque sea en un vecindario muy pequeño.
- Si el algoritmo de clasificación sólo depende de ese vencindario local para clasificar, no hay ganancia en la distancia de Fermat.
- ¡Pero tampoco hay pérdida si se elige mal `n_neighbors`! #emoji.person.shrug


=== $R^2$ por semilla para $"(F)KN"$ en `helices_0`
#align(center)[#image("img/r2-fkn-kn-helices_0.png")]

=== $R^2$ y $alpha^star$ para $"(F)KN"$ en `helices_0`, `n_neighbors` seleccionados
#align(center)[#image("img/r2-fkn-kn-n_neighbors-seleccionados.png")]

=== Mejor $R^2$ para $"(F)KN"$ en `helices_0`, en función de `n_neighbors`

#image("img/helices_0-fkn_kn-mean_test_score.png")


=== $R^2$ por semilla para $"(F)KN"$ en `eslabones_0`
#align(center)[#image("img/outputa.png")]

=== $R^2$ y $alpha^star$ para $"(F)KN"$ en `eslabones_0`, `n_neighbors` seleccionados
#align(center)[#image("img/Screenshot 2025-07-18 at 11.43.27 AM.png")]

=== Mejor $R^2$ para $"(F)KN"$ en `eslabones_0`, en función de `n_neighbors`

#image("img/outputb.png")


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


=== Conclusiones




==== Para Patu
Lo junan a @carpioFingerprintsCancerPersistent2019? "Fingerprints of cancer by persistent homology"
#quote[
  We have carried out a topological data analysis of gene expressions for diﬀerent databases based on the Fermat distance between the z scores of diﬀerent tissue samples. There is a critical value of the ﬁltration parameter at which all clusters collapse in a single one. This critical value for healthy samples is gapless and smaller than that for cancerous ones. After collapse in a single cluster, topological holes persist for larger ﬁltration parameter values in cancerous samples. Barcodes, persistence diagrams and Betti numbers as functions of the ﬁltration parameter are diﬀerent for diﬀerent types of cancer and constitute ﬁngerprints thereof.
]
= Glosario

/ clausura: ???
/ Riemanniana, métrica: sdfsdf
/ Lebesgue, medida de: ???
/ densidad, estimación de: cf. @berenfeldDensityEstimationUnknown2021
/ ventana: parámetro escalar que determina la "unidad" de distancia
/ núcleo, función: $K$


= Listados
#outline(target: figure.where(kind: image), title: "Listado de Figuras")
= Tablas
#outline(target: figure.where(kind: table), title: "Listado de Tablas")
= Código
#outline(target: figure.where(kind: raw), title: "Listado de código")

#bibliography("../bib/references.bib", style: "harvard-cite-them-right")
