#import "@preview/ctheorems:1.1.3": *

// ################
// # definiciones #
// ################
#let ind = $op(bb(1))$
#let iid = "i.i.d."
#let sop = $op("sop")$
#let Pr = $op("Pr")$
#let bu(x) = $bold(upright(#x))$
#let GG = $cal(G)$
#let MM = $cal(M)$
#let HH = $bu(H)$
#let XX = $bu(X)$
#let KH = $op(K_HH)$
#let dotp(x, y) = $lr(chevron.l #x, #y chevron.r)$
#let dg = $op(d_g)$
#let var = $op("Var")$
#let SS = $bu(Sigma)$
// nombres de clasificadores
#let fkdc = [$f$`-KDC`]
#let kdc = `KDC`
#let fkn = [$f$`-KN`]
#let kn = `KN`
#let gnb = `GNB` // $("GNB")$
#let logr = `LR`
#let slr = [$s$`-LR`]
#let svc = `SVC`
#let gbt = `GBT`
// clasificador genérico
#let clf = $op(hat(G))$
#let reps = 25

// Copetes flexibles para outline y texto, adaptado para 0.12 de
// https://github.com/typst/typst/issues/1295#issuecomment-1853762154
#let in-outline = state("in-outline", false)
#show outline: it => {
  in-outline.update(true)
  it
  in-outline.update(false)
}

#let flex-caption(long, short) = (
  context if in-outline.get() {
    short
  } else {
    long
  }
)
#let defn = thmbox("definition", "Definición", inset: (x: 1.2em, top: 1em), base_level: 2)
#let obs = thmplain("observation", "Observación").with(numbering: none)
#let thm = thmbox("theorem", "Teorema", inset: (x: 1.2em, top: 1em), base_level: 2)

// conveniencias
#let hfrac(num, denom) = math.frac(num, denom, style: "horizontal")

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
#set math.equation(numbering: "(1)")
#set quote(block: true)

#show: thmrules
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)
#show figure: set block(above: 1.2em, below: 1.2em)
#show link: it => underline(text(it, fill: blue))

// #############
// ### utils ###
// #############

#let tabla_csv(path) = {
  let data = csv(path)
  let eval_scope = (fkdc: fkdc, kn: kn, fkn: fkn, kdc: kdc, lr: logr, svc: svc, lsvc: `LSVC`, gnb: gnb, base: "base")
  table(columns: data.at(0).len(), ..data.flatten().map(eval.with(mode: "markup", scope: eval_scope)))
}


// ### TOC y listados
#outline(depth: 2)

#pagebreak()

= Vocabulario y Notación

A lo largo de esta monografía tomaremos como referencia enciclopédica el excelente _Elements of Statistical Learning_ @hastieElementsStatisticalLearning2009. En la medida de lo posible, basaremos nuestra notación en la suya.

Típicamente, denotaremos a las variables independientes #footnote[También conocidas como predictoras, o #emph[inputs]] con $X$. Si $X$ es un vector, accederemos a sus componentes con subíndices, $X_j$. En el contexto del problema de clasificación, la variable _cualitativa_ dependiente #footnote[También conocida como variable respuesta u #emph[output]] será $G$ (de "G"rupo). Usaremos letras mayúsculas como $X, G$ para referirnos a los aspectos genéricos de una variable. Los valores _observados_ se escribirán en minúscula, de manera que el i-ésimo valor observado de $X$ será $x_i$ (de nuevo, $x_i$ puede ser un escalar o un vector).

Representaremos a las matrices con letras mayúsculas en negrita, #XX; e.g.: el conjunto de $N$ vectores $p$-dimensionales ${x_i, i in {1, dots, N}}$ será representado por la matriz #XX de dimensión $N times p$.

En general, los vectores _no_ estarán en negrita, excepto cuando tengan $N$ componentes; esta convención distingue el $p-$vector de #emph[inputs] para la i-ésima observación,  $x_i$, del $N-$vector $bu(x)_j$ con todas las observaciones de la variable $X_j$. Como todos los vectores se asumen vectores columna, la i-ésima fila de #XX es $x_i^T$, la traspuesta de la i-ésima observación $x_i$.

#pagebreak()

A continuación, algunos símbolos y operadores utilizados a lo largo del texto:

#set terms(separator: h(2em, weak: true), spacing: 1em)

/ $RR$: los números reales; $RR_+$ denotará los reales estrictamente positivos.
/ $RR^p$: el espacio euclídeo de dimensión $p$
/ $[k]$: el conjunto de los enteros positivos del $1$ hasta $k$, ${1, 2, 3, dots, k}$
/ #MM: una variedad arbitraria #footnote[típicamente Riemanniana, compacta y sin frontera; oportunamente definiremos estos calificativos]
/ $d_x$: la dimensión "natural" #footnote[la dimensión de un elemento es su cantidad de componentes, la dimensión de un espacio es la dimensión de cualquiera de sus elementos] del elemento $x$
/ $h$: la ventana ($h in RR$) en un estimador de densidad por núcleos en $RR$
/ $bu(H)$: ídem $h$, para estimadores en $RR^p$ ($bu(H) in RR^(p times p)$)
/ $norm(dot)$: norma euclídea del elemento $x$
/ $bu(X)$: una muestra de $N$ elementos $p$-dimensionales ($XX in RR^(N times p)$)
/ $X_(i, j)$: la j-ésima dimensión del i-ésimo elemento de #XX
/ $ind(x)$: la función indicadora, $ind(x)=cases(1 "si" x "es verdadero", 0 "si no")$
/ $Pr(dot)$: función de probabilidad #footnote[en general no hará falta definir el espacio muestral ni la $sigma-$álgebra correspondientes; de hacer falta se indicarán con subíndices] <fn-pr>
/ $EE(dot)$: la función esperanza @fn-pr
/ $var(dot)$: la función varianza @fn-pr
/ $iid$: independientes e idénticamente distribuidos #footnote[típicamente los elementos aleatorios de #XX son $iid$]
/ $emptyset$: el conjunto vacío
/ $A slash B$: el complemento de $B$ en $A$, $A slash B = {a:a in A, a in.not B}$
/ $overline(S)$: la _clausura_ de S (la unión de S y sus puntos límites)
/ $lambda(x)$: la medida de Lebesgue de $x$ en $RR^d$
/ $a |-> b$: la función que "toma" $a$ y "devuelve" $b$  en notación de flechas
/ $y prop x$: "y es proporcional a x", existe una constante $c : y = c times x$
/ "c.s.": "casi seguramente", re. convergencia de elementos aleatorios
= Preliminares

== El problema de clasificación

=== Definición y vocabulario #footnote[adaptado de @hastieElementsStatisticalLearning2009[§2.4, "Statistical Decision Theory"]]
El _aprendizaje estadístico supervisado_ busca estimar (aprender) una variable _respuesta_ a partir de cierta(s) variable(s) _predictora(s)_. Cuando la _respuesta_ es una variable _cualitativa_, el problema de asignar cada observación $X$ a una clase $G in GG={GG_1, dots, GG_K}$ se denomina _de clasificación_. En general, reemplazaremos los nombres o "etiquetas" de clases $GG_i$ por los enteros correspondientes, $G in [K]$. En esta definición del problema las clases son

- _mutuamente excluyentes_: cada observación $X_i$ está asociada a lo sumo a una clase
- _conjuntamente exhaustivas_: cada observación $X_i$ está asociada al menos a una clase.

#defn("clasificador")[
  Un _clasificador_ es una función $hat(G)(X)$ que para cada observación intenta aproximar su verdadera clase $G$ por $hat(G)$ #footnote[pronunciado "ge sombrero"].
] <clasificador>

Para construir $hat(G)$, contaremos con una muestra o _conjunto de entrenamiento_ $XX, bu(g)$,  de pares $(x_i, g_i), i in {1, dots, N}$ conocidos.

Para discernir cuán bien se "ajusta" un clasificador a los datos, la teoría requiere de una "función de pérdida" $L(G, hat(G)(X))$ #footnote[_loss function_ en inglés. A veces también "función de riesgo" - _risk function_.]. Será de especial interés la función de clasificación $f$ que minimiza el "error de predicción esperado" $"EPE"$ #footnote[del inglés #emph[expected prediction error]]:

$
  hat(G) = arg min_f "EPE"(f) =arg min_f EE(L(G, f(X)))
$
donde la esperanza es contra la distribución conjunta $(X, G)$. Por la ley de la probabilidad total, podemos condicionar a X #footnote[Aquí "condicionar" implica factorizar la densidad conjunta $Pr(X, G) = Pr(G|X) Pr(X)$ donde $Pr(G|X) = hfrac(Pr(X, G), Pr(X))$, y repartir la integral bivariada de manera acorde.] y expresar el EPE como

$
  "EPE"(f) & = EE_(X,G)(L(G, hat(G)(X))) \
           & = EE_X EE_(G|X)(L(G, hat(G)(X))) \
           & = EE_X sum_(k in [K]) L(GG_k, hat(G)(X)) Pr(GG_k | X) \
$
Y basta con minimizar punto a punto para obtener una expresión computable de $hat(G)$:
$
  hat(G)(x) & = arg min_f EE(L(G, f(X))) \
            & = arg min_(g in GG) sum_(k in [K]) L(GG_k, g) Pr(GG_k | X = x)
$
Con la _pérdida 0-1_ #footnote[que no es otra cosa que la función indicadora de un error en la predicción, $bu(01)(hat(G), G) = ind(hat(G) != G)$], la expresión se simplifica a
$
  hat(G)(x) & = arg min_(g in GG) sum_(k in [K]) ind(cal(G)_k != g) Pr(GG_k|X=x) \
            & = arg min_(g in GG) [1-Pr(g|X=x)] \
            & = arg max_(g in GG) Pr(g | X = x)
$<clf-bayes>

Esta razonable solución se conoce como el _clasificador de Bayes_, y sugiere que clasifiquemos a cada observación según la clase modal #footnote[i.e., la de mayor probabilidad] condicional a su distribución conjunta $Pr(G|X)$.
Su error esperado de predicción $"EPE"$ se conoce como la _tasa de Bayes_. Un aproximador directo de este resultado es el clasificador de "k vecinos más cercanos" #footnote[del inglés _k-nearest-neighbors_]

#defn("clasificador de k-vecinos-más-cercanos")[
  Sean $x^((1)), dots, x^((k))$ los $k$ #footnote[que no guarda relación alguna con la cantidad $K$ del problema de clasificación] vecinos más cercanos a $x$, y $g^((1)), dots, g^((k))$ sus respectivas clases. El clasificador de k-vecinos-más-cercanos - que notaremos #kn - le asignará a $x$ la clase más frecuente entre $g^((1)), dots, g^((k))$. Más formalmente:
  $
    hat(G)_("kn")(x) & = g = arg max_(g in GG) sum_(i in [k]) ind(g^((i)) = g)
  $

] <kn-clf>

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
                & <=> Pr(GG_k|X=x) =max_(k in [K]) Pr(X=x|GG_k) times Pr(GG_k), \
$

donde el segundo $<=>$ vale siempre que $Pr(X = x) > 0$ --- y dado que _observamos_ $X=x$, el supuesto es razonable.

A las probabilidades "incondicionales" de clase $Pr(GG_k)$ se las suele llamar su "distribución a priori", y notarlas por $pi = (pi_1, dots, pi_K)^T, sum pi_k = 1$. Una aproximación razonable, si es que el conjunto de entrenamiento se obtuvo por muestreo aleatorio simple, es estimarlas a partir de las proporciones muestrales:
$
  forall k in [K], quad hat(pi)_k & = N^(-1) sum_(i in [N]) ind(g_i = GG_k) \
                                  & = \#{g_i : g_i = GG_k, i in [N]} / N
$


Resta hallar una aproximación $hat(Pr)(X=x|GG_k)$ a las probabilidades condicionales $X|GG_k$ para cada clase.

== Estimación de densidad por núcleos

Tal vez la metodología más estudiada a tales fines es la estimación de densidad por núcleos, reseñada en @hastieElementsStatisticalLearning2009[§6.6]. En el caso unidimensional, al estimador resultante se lo conoce por el nombre de Parzen-Rosenblatt, por sus contribuciones fundacionales en el área @parzenEstimationProbabilityDensity1962 @rosenblattRemarksNonparametricEstimates1956.

=== Estimación unidimensional


Para fijar ideas, asumamos que $X in RR$ y consideremos la estimación de densidad en una única clase para la que contamos con $N$ ejemplos ${x_1, dots, x_N}$. Una aproximación $hat(f)$ directa sería
$
  hat(f)(x_0) = \#{x_i in cal(N)(x_0)} / (N times h)
$ #label("eps-nn")


donde $cal(N)$ es un vecindario métrico de $x_0$ de diámetro $h$.

Esta estimación es irregular, con saltos discretos en el numerador, por lo que se prefiere el estimador "suavizado por núcleos" de Parzen-Rosenblatt. Pero primero: ¿qué es un núcleo?


#defn([función núcleo o _kernel_])[

  Se dice que $K(x) : RR -> RR$ es una _función núcleo_ si cumple que

  + toma valores reales no negativos: $K(u) >= 0$,
  + está "normalizada": $integral K(u) d u = 1$,
  + es simétrica en torno al cero: $K(u) = K(-u)$ y
  + alcanza su máximo en el centro: $max_u K(u) = K(0)$
] <kernel>

#obs[Todas las funciones de densidad simétricas centradas en 0 son núcleos; en particular, la densidad "normal estándar" $ phi.alt(x) = 1/sqrt(2 pi) exp(-x^2 / 2) $ lo es.]

#obs[Si $K(u)$ es un núcleo, entonces $K_h (u) = 1/h op(K)(u / h)$ también lo es.]

#defn("estimador de densidad por núcleos")[


  Sea $bu(x) = (x_1, dots, x_N)^T$ una muestra #iid de cierta variable aleatoria escalar $X in RR$ con función de densidad $f$. Su estimador de densidad por núcleos, KDE #footnote[de _Kernel Density Estimator_, por sus siglas en inglés] o estimador de Parzen-Rosenblatt es
  $
    hat(f)(x_0) = 1/N sum_(i=1)^N 1/ h K ((x_0 - x_i)/h) = 1/N sum_(i=1)^N K_h (x_0 - x_i)
  $

  donde $K_h$ es un núcleo según @kernel. Al parámetro $h$ se lo conoce como "ventana" de suavizado o _smoothing_.
] <parzen>

#obs[
  La densidad de la distribución uniforme centrada en 0 de diámetro 1, $U(x) = ind(1/2 < x <= 1/2)$ es un núcleo.  Luego, $ U_h (x) = 1/h ind(-h/2 < x < h/2) $ también es un núcleo válido, y por ende el estimador de @eps-nn resulta estrechamente emparentado al estimador de @parzen:
  $
    hat(f)(x_0) & = \#{x_i in cal(N)(x_0)} / (N times h) \
                & = 1 / N sum_(i in [N]) 1/h ind(-h/2 < x_i - x_0 < h/2) \
                & = 1 / N sum_(i in [N]) U_h (x_i - x_0)
  $
  con la diferencia de que el estimador de @eps-nn fija el _diámetro_ del vecindario a considerar, y el de @parzen fija la _cantidad_ de vecinos a tener en cuenta #footnote[Al primero se lo conoce como $epsilon$- nearest neighbors ($epsilon$-NN) con $epsilon$ denotando el _radio_ del vecindario; el segundo es el ya descrito $k$-NN.].
]
=== Clasificador de densidad por núcleos

Si $hat(f)_k, k in [K]$ son estimadores de densidad por núcleos de cada una de las $K$ densidades condicionales $X|GG_k$ según @parzen, podemos construir el siguiente clasificador

#defn(
  "clasificador de densidad por núcleos",
)[ Sean $hat(f)_1, dots, hat(f)_K$ estimadores de densidad por núcleos según @parzen. Sean además $hat(pi)_1, dots, hat(pi)_K$ las estimaciones de la probabilidad incondicional de pertenecer a cada grupo $GG_1, dots, GG_k$. Luego, la siguiente regla constituye un clasificador de densidad por núcleos - lo notaremos #kdc :
  $
    hat(G)_"KD" (x) = g & = arg max_(i in [K]) hat(Pr)(GG_i | X = x) \
                        & = arg max_(i in [K]) hat(Pr)(X=x|GG_i) times hat(Pr)(GG_i) \
                        & = arg max_(i in [K]) hat(f)_i (x) times hat(pi)_i \
  $] <kdc-duro>

=== Clasificadores duros y suaves

Un clasificador que asigna a cada observación _una clase_ - la más probable, se suele llamar _clasificador duro_. Un clasificador que asigna a cada observación _una distribución de probabilidades de clase_ $hat(gamma)$ #footnote[$hat(gamma)$ aproximará $gamma = (gamma_1, dots, gamma_K)^T$ con $gamma_i = Pr(G = GG_i), quad sum_(i in [K]) gamma_i = 1$.] se suele llamar _clasificador blando_. Dado un clasificador _blando_ $hat(G)_"Blando"$, es trivial construir el clasificador duro asociado $hat(G)_"Duro"$:
$
  hat(G)_"Duro" (x_0) = arg max_i hat(G)_"Blando" (x_0) = arg max_i hat(gamma)_i
$

#obs[
  El clasificador de @kdc-duro es la versión dura de un clasificador blando $hat(G)_"KD" (x) = hat(gamma)$, donde $ hat(gamma)_i = (hat(f)_i (x) times hat(pi)_i) / (sum_(i in [K]) hat(f)_i (x) times hat(pi)_i) $
]

#obs[
  Ciertos clasificadores solo pueden ser duros, como $hat(G)_"1-NN"$ (el clasificador de @kn-clf con $k=1$), o aquellos derivados de algoritmos clasifican sin estimar probabilidades condicionales, como los basados en SVMs #footnote["#link("https://es.wikipedia.org/wiki/M%C3%A1quina_de_vectores_de_soporte")[máquinas de vectores de soporte]", del inglés _support vector machines_].
]

Dos clasificadores _blandos_ pueden tener la misma pérdida $0-1$, pero "pintar" dos panoramas muy distintos respecto a cuán "seguros" están de cierta clasificación. Por caso, sea $epsilon > 0$ y arbitrariamente pequeño:
$
  hat(G)_"C(onfiado)" (x_0) &: hat(Pr)(GG_i | X = x_0) = cases(1 - epsilon &" si " i = 1, hfrac(epsilon, (K - 1)) &" si " i != 1) \
  hat(G)_"D(udoso)" (x_0) &: hat(Pr)(GG_i | X = x_0) = cases(1/K + epsilon &" si " i = 1, 1 / K - hfrac(epsilon, (K - 1)) &" si " i != 1)
$
$hat(G)_C$ está "casi seguro" de que la clase correcta es $GG_1$, mientras que $hat(G)_D$ está otorga casi la misma probabilidades a todas las clases. Para el entrenamiento y análisis de clasificadores blandos como el de densidad por núcleos, será relevante encontrar funciones de pérdida que recompensen la confianza de un clasificador _cuando ésta esté justificada_ #footnote[y lo penalicen cuando no - es decir, cuando la confianza está puesta en la clase errada. Más al respecto, más adelante.].

== Estimación de densidad multivariada
=== Naive Bayes
Una manera "ingenua" de adaptar el procedimiento de estimación de densidad ya mencionado a $X$ multivariadas, consiste en sostener el falso-pero-útil supuesto de que sus componentes $X_1, dots, X_p$ son independientes entre sí. De este modo, la estimación de densidad conjunta se reduce a la estimación de $p$ densidades marginales univariadas. Dada cierta clase $j$ #footnote[donde el entero $j in [K]$ es la etiqueta de la clase $GG_j$], podemos escribir la densidad condicional $X|j$ como
$
  f_j (X) = product_(k = 1)^p f_(j k) (X_k)
$ <naive-bayes>

A este procedimiento se lo conoce como "Naive Bayes" @hastieElementsStatisticalLearning2009[§6.6.3], y a pesar de su aparente ingenuidad es competitivo contra algoritmos mucho más sofisticados en un amplio rango de tareas. En términos de cómputo, permite resolver la estimación con $K times p$ KDE univariados. Además, permite que en $X$ se combinen variables cuantitativas y cualitativas: basta con reemplazar la estimación de densidad para las componenets $X_k$ cualitativos por su correspondiente histograma.

=== KDE multivariado
Consideremos un _dataset_ compuesto por observaciones muestradas de dos círculos concéntricos con algo de ruido:
#figure(
  caption: flex-caption(
    "Dos círculos concéntricos y sus KDE marginales por clase: a pesar de que la frontera entre ambos grupos de puntos es muy clara, es casi imposible distinguirlas a partir de sus densidades marginales.",
    "Dos círculos concéntricos",
  ),
  image("img/dos-circulos-jointplot.png", width: 75%),
)


En casos así, el procedimiento de Naive Bayes falla por completo, y será necesario adaptar el procedimiento de KDE unidimensional a $d >= 2$ sin basarnos en el supuesto de independencia de las $X_1, dots, X_k$. A lo largo de las cuatro décadas posteriores a las publicaciones de Parzen y Rosenblatt, el estudio de los estimadores de densidad por núcleos avanzó considerablemente, de manera que ya para mediados de los \'90 existían minuciosos libros de referencia como "Kernel Smoothing" @wandKernelSmoothing1995, que seguiremos en la presente sección.

#defn([KDE multivariada, @wandKernelSmoothing1995[§4]])[
  En su forma más general, estimador de densidad por núcleos #box[$d-$ variado] es

  $
    hat(f) (x; HH) = N^(-1) sum_(i=1)^N KH (x - x_i)
  $

  donde
  - $HH in RR^(d times d)$ es una matriz simétrica definida positiva análoga a la ventana $h in RR$ para $d=1$,
  - $KH(t) = abs(det HH)^(-1/2) K(HH^(-1/2) t)$
  - $K$ es una función núcleo $d$-variada tal que $integral K(bu(x)) d bu(x) = 1$
] <kde-mv>

Típicamente, K es la densidad normal multivariada
$
  Phi(x) : RR^d -> RR = (2 pi)^(-d/2) exp(- (||x||^2)/2)
$

=== La elección de $HH$
Sean las clases de matrices $RR^(d times d)$
- $cal(F)$, de matrices simétricas definidas positivas,
- $cal(D)$, de matrices diagonales definidas positivas ($cal(D) subset.eq cal(F)$) y
- $cal(S)$, de múltiplos escalares de la identidad: $cal(S) = {h^2 bu(I):h >0} subset.eq cal(D)$

Aún tomando una única $HH$ para _toda_ la muestra, la elección de $HH$ en $d$ dimensiones requiere ajustar
- $mat(d; 2) = (d^2 - d) slash 2$ parámetros si $HH in cal(F)$,
- $d$ parámetros si $HH in cal(D)$ y
- un único parámetro $h$ si $HH = h^2 bu(I)$.

La evaluación de la conveniencia relativa de cada parametrización se vuelve muy compleja, muy rápido. @wandComparisonSmoothingParameterizations1993 proveen un análisis detallado para el caso $d = 2$, y concluyen que aunque cada caso amerita su propio estudio, $HH in cal(D)$ suele un compromiso "adecuado" entre la complejidad de tomar $HH in cal(F)$ y la rigidez de $HH in cal(S)$. Sin embargo, este no es un gran consuelo para valores de $d$ verdaderamente altos, en cuyo caso existe aún un problema más fundamental.

=== La maldición de la dimensionalidad

Uno estaría perdonado por suponer que el problema de estimar densidades en alta dimensión se resuelve con una buena elección de $HH$, y una muestra "lo suficientemente grande". Considérese, sin embargo, el siguiente ejercicio ilustrativo de cuánto es "suficientemente grande":

#quote(attribution: [adaptado de @wandKernelSmoothing1995[§4.9 ej 4.1]])[
  Sean $X_i tilde.op^("iid")"Uniforme"([-1, 1]^d), thick i in [N]$, y consideremos la estimación de la densidad en el origen, $hat(f)(bu(0))$. Suponga que el núcleo $K_(HH)$ es un "núcleo producto" basado en la distribución univariada $"Uniforme"(-1, 1)$, y $HH = h^2 bu(I)$. Derive una expresión para la proporción esperada de puntos incluidos dentro del soporte del núcleo $KH$ para $(h, d)$ arbitrarios.
]

El "núcleo producto" multivariado basado en la ley $"Uniforme"(-1, 1)$ evaluado alrededor del origen es:
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
                     & = Pr(-h <= X_1 <= h)^d \
                     & = [(h - (-h))/(1-(-1))]^d = h^d quad square
$

#let h = 0.5
#let d = 20

Para $h =#h, d=#d, thick Pr(X in [-#h,#h]^#d) = #h^(-#d) approx #calc.round(calc.pow(h, d), digits: 8)$, ¡menos de uno en un millón! En general, la caída es muy rápida, aún para valores altos de $h$. Si $X$ representa un segundo de audio muestreado respetando el estándar _mínimo_ para llamadas telefónicas  #footnote[De Wikipedia: La tasa #link("https://en.wikipedia.org/wiki/Digital_Signal_0")[DS0], o _Digital Signal 0_, fue introducida para transportar una sola llamada de voz "digitizada". La típica llamada de audio se digitiza a $8 "kHz"$, o a razón de 8.000 veces por segundo.], tiene $d=8000$. En tal espacio ambiente, aún con $h=0.999$, $Pr(dot) approx #calc.round(calc.pow(0.999, 8000), digits: 6)$, o 1:3.000.

#figure(
  caption: flex-caption(
    [Proporción de $X_i tilde.op^("iid")"Uniforme"([-1, 1]^d)$ dentro de un $d$-cubo de lado $h$ para valore seleccionados de $h$.],
    [Proporción de $X$ dentro de un $d$-cubo de lado $h$],
  ),
  image("img/curse-dim.png"),
)
=== La hipótesis de la variedad (_manifold hypothesis_)

Ahora, si el espacio está _tan_, pero _tan_ vacío en alta dimensión, ¿cómo es que el aprendizaje supervisado _sirve de algo_? La reciente explosión en capacidades y herramientas de procesamiento (¡y generación!) de formatos de altísima dimensión #footnote[audio, video, texto y data genómica, por citar solo algunos] pareciera ser prueba fehaciente de que la tan mentada _maldición de la dimensionalidad_ no es más que una fábula para asustar estudiantes de estadística.

Pues bien, el ejemplo de un segundo segundo de audio antedicho _es_ sesgado: no es cierto que si $X$ representa un segundo de voz humana digitizada, su ley sea uniforme en 8000 dimensiones #footnote[El audio se digitiza usando 8 bits para cada muestra, así que más precisamente, si $B = [2^8] = {1, dots, 256}, sop X = B^8000$ o $64 "kbps"$, kilobits-por-segundo.]. Un segundo de audio generado siguiendo cualquier distribución en la que muestras consecutivas no tengan ninguna correlación, obtiene #link("https://es.wikipedia.org/wiki/Ruido_blanco")[_ruido blanco_]. La voz humana, por su parte, tiene _estructura_, y por ende correlación instante a instante. Cada voz tiene un _timbre_ característico, y las palabras enuncidas posibles están ceñidas por la _estructura fonológica_ de la lengua locutada.

Sin precisar detalles, podríamos postular que las realizaciones de la variable de interés $X$ (el habla), que registramos en un soporte $cal(S) subset.eq RR^d$ de alta dimensión, en realidad se concentran en cierta _variedad_ #footnote[Término que ya precisaremos. Por ahora, #MM es el _subespacio de realizaciones posibles_ de $X$] $MM subset.eq cal(S)$ potencialmente de mucha menor dimensión $dim MM = d_MM << d = dim cal(S)$, en la que noción de distancia entre observaciones aún conserva significado. A tal postulado se lo conoce como "la hipótesis de la variedad", o _manifold hypothesis_. <hipotesis-variedad>
#footnote[
  Para el lector curioso: @rifaiManifoldTangentClassifier2011 ofrece un desglose de la hipótesis de la variedad en tres aspectos complementarios, de los cuales el aquí presentado sería el segundo, la "hipótesis de la variedad no-supervisada". El tercero, "la hipótesis de la variedad para clasificación", dice que "puntos de distintas clases se concentrarán sobre variedades disjuntas separadas por regiones de muy baja densidad", y lo asumimos implícitamente a la hora de construir un clasificador.
]


La hipótesis de la variedad no es exactamente una hipótesis contrastable en el sentido tradicional del método científico; de hecho, ni siquiera resulta obvio que de existir, sean bien definibles las variedades en las que existen los elementos del mundo real: un dígito manuscrito, el canto de un pájaro, o una flor. Y de existir, es de esperar que sean altamente #box[no-lineales]. Más bien, corresponde entenderla como un modelo mental, que nos permite aventurar ciertas líneas prácticas de trabajo en alta dimensión.
#footnote[
  El concepto de "variedad" para denotar más o menos formalmente un espacio no-euclídeo con ciertas características intuitivas está extendido en literatura no estrictamente matemática. Para el lector ávido, mencionamos dos _papers_ interesantes al respecto de potenciales modelos "varietales" de fenómenos como la empatía y la conciencia.

  Uno es @galleseRootsEmpathyShared2003, _Las Raíces de la Empatía: La Hipótesis de la Variedad Compartida y las Bases Neuronales de la Intersubjetividad_: la hipótesis sostiene que existe un espacio intersubjetivo que compartimos con los demás. No somos mentes aisladas intentando descifrar a otras mentes aisladas; más bien, habitamos un espacio común de acción y emoción. Este "nosotros" (_we-centric space_) es la condición de posibilidad para la empatía. Reconocemos al otro no como un objeto, sino como otro "yo", porque cohabitamos la misma variedad corporal y neuronal.

  El otro es  @bengioConsciousnessPrior2019, _El Prior de la Conciencia_, en el que se postula que ante un espacio infinito de estímulos, la conciencia tiene una función evolutiva y computacional específica: actuar como un cuello de botella de información para facilitar el razonamiento y la generalización. La conciencia produce una representación rala y de baja dimensionalidad compuesta por los factores salientes de entre los estímulos recibidos y sus interconexiones - es decir, una cierta variedad de baja dimensión intrínsica.
]

#figure(caption: flex-caption(
  [Ejemplos de variedades en el mundo físico: una bandera flameando al viento, el pétalo de una flor. Ambas tienen dimensión $d_MM = 2$, están embedidas en $RR^3$, y no son lineales.],
  "Ejemplos de variedades en el mundo físico",
))[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 1em,
    image("img/hormiga-petalo.jpg", height: 14em), image("img/bandera-argentina.png", height: 14em),
  )
]

Antes de poder profundizar en esta línea, debemos plantearnos algunas preguntas básicas:
#align(center)[
  ¿Qué es, exactamente, una variedad? \ \
  ¿Es posible construir un KDE con soporte en cierta variedad _conocida_? \ \
  ¿Sirve de algo todo esto si _no conocemos_ la variedad en cuestión?
]

== Variedades de Riemann

Adelantando la respuesta a la segunda pregunta, resulta ser que si el soporte de $X$ es una "variedad de Riemann", bajo ciertas condiciones razonables sí es posible estimar su densidad por núcleos en la variedad @pelletierKernelDensityEstimation2005.

A continuación, damos un recorrido sumario e idiosincrático por ciertos conceptos básicos de topología y variedades que consideramos necesarios para motivar la definición de variedades Riemannianas, que de paso precisarán la respuesta a la primera pregunta - ¿qué es una variedad? - en el contexto que nos interesa. A tal fin, seguimos la exposición de la monografía _Estimación no paramétrica de la densidad en variedades Riemannianas_ @munozEstimacionNoParametrica2011, que a su vez sigue, entre otros, el clásico _Introduction to Riemannian Manifolds_ @leeIntroductionRiemannianManifolds2018.

=== Variedades Diferenciables

#v(-1em)

#defn([espacio topológico @wikipediaEspacioTopologico2025])[

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

#defn([entorno @wikipediaEntornoMatematica2025])[
  Si $(X,Τ)$ es un espacio topológico y $p$ es un punto perteneciente a X, un _entorno_ #footnote[ También se los conoce como "vecindarios" - por _neighborhoods_, su nombre en inglés.] del punto $p$ es un conjunto $V$ en el que está contenido un conjunto abierto $U$ que incluye al propio $p: p in U subset.eq V$.
]

#defn([espacio de Hausdorff @wikipediaEspacioHausdorff2024])[

  Sea $(X, T)$ un espacio topológico. Se dice que dos puntos $p, q in X$ cumplen la propiedad de Hausdorff si existen dos entornos $U_p$ de $p$ y $U_q$ de $q$ tales que $U_p inter U_q = emptyset$ (i.e., son disjuntos).

  Se dice que un espacio topológico es un espacio de Hausdorff #footnote[o que "verifica la propiedad de Hausdorff", o que "es separado o que es $bu(T_2)$"] si todo par de puntos distintos del espacio verifican la propiedad de Hausdorff.
]
En términos coloquiales, un espacio de Hausdorff es aquel donde todos sus puntos están "bien separados".

#defn(
  [variedad topológica @munozEstimacionNoParametrica2011[Def. 3.1.1], @leeIntroductionRiemannianManifolds2018[Apéndice A]],
)[
  Una variedad topológica de dimensión $d in NN$ es un espacio topológico $(MM, T)$ de Hausdorff, de base numerable, que es #strong[localmente homeomorfo a $RR^d$]. Es decir, para cada $p in MM$ existe un abierto $U in T$ y un abierto $A subset.eq RR^d$, tal que $p in U$ #footnote[de modo que $U$ es un entorno de $p$] y existe un homeomorfismo $phi : U -> A$.
]

#obs(
  "Sobre variedades con y sin frontera",
)[ Toda $n-$variedad #footnote[i.e. variedad de dimensión $n$] tiene puntos interiores, pero algunas además tienen una _frontera_; esta frontera es a su vez una variedad _sin_ frontera de dimensión $n - 1$. Por caso: un disco en el plano euclídeo $RR^2$ es una $2-$variedad _con_ frontera, cuya frontera es una variedad de dimensión $2 - 1 = 1$ sin frontera: el círculo $S^1$ #footnote[$S^n$ denota la $n-$esfera: la variedad de los puntos en $RR^(n+1)$ a distancia unitaria del origen. Así, $S^1$ es el círculo y $S^2$ es la superficie esférica.]; una pelota de tenis es una $3-$variedad con frontera dada por su superficie, que es (aproximadamente) la variedad sin frontera $S^2$. De aquí en más, cuando hablemos de variedades topológicas, nos referiremos a variedades _sin_ frontera.]


En una variedad topológica, cobra sentido cierto concepto de cercanía pero no necesariamente de _distancia_, y es posible definir funciones continuas y límites.

Un _homeomorfismo_ #footnote[del griego _homo-_: igual, _-morfo_: forma; de igual forma] es una función $phi$ entre dos espacios topológicos si es biyectiva y tanto ella como su inversa son continuas. El par ordenado $(U, phi)$ es una _carta #footnote[_chart_ en inglés] alrededor de $p$_.

A un conjunto numerable de tales cartas que cubran completamente la variedad se lo denomina "atlas". Simbólicamente, #box[$cal(A) = {(U_alpha, phi_alpha) : alpha in cal(I)}$] es un atlas sí y solo si $MM = union_alpha U_alpha$. Al conjunto de entornos ${U_alpha : (U_alpha, phi_alpha) in cal(A)}$ que componen un atlas se lo denomina "cobertura" de #MM.

Cuando un homeomorfismo - y su inversa - es $r-$veces diferenciable, se le llama _$C^r$-difeomorfismo_, o simplemente difeomorfismo #footnote[Luego, un homeomorfismo es un $C^0-$difeomorfismo]. En particular, un $C^oo-$difeomorfismo es un difeomorfismo _suave_.

#defn([cartas suavemente compatibles])[
  Sean $(MM, T)$ una variedad topológica de dimensión $d$ y sean $(U, phi), (V, psi)$ dos cartas. Diremos que son _suavemente compatibles_ #footnote[_smoothly compatible_ según @leeIntroductionRiemannianManifolds2018[ § "Smooth Manifolds and Smooth Maps"]. @munozEstimacionNoParametrica2011 lo denomina _compatible_ a secas.] si $U inter V = emptyset$ o bien si la función cambio de coordenadas restringida a $U inter V$ es un difeomorfismo.]

La compatibilidad requiere que la transición entre cartas no sea solo continua, sino también _suave_. El motivo de esta condición es asegurar que el concepto de _suavidad_ esté bien definido en toda la variedad $MM$, independientemente de qué carta se use: si una función es diferenciable vista a través de una carta, también lo será al analizarla desde cualquier carta compatible.

#defn([estructura diferenciable @munozEstimacionNoParametrica2011[Def. 3.1.3]])[
  Un atlas $cal(A) = {(U_alpha, phi_alpha) : alpha in cal(I)}$ es diferenciable si sus cartas son compatibles entre sí. Si un atlas diferenciable $cal(D)$ es _maximal_ lo llamaremos una _estructura diferenciable de la variedad $MM$ _. Con maximal queremos decir lo siguiente: Si $(U, phi)$ es una carta de $MM$ que es compatible con todas las cartas de $cal(D)$, entonces $(U, phi) in cal(D)$ #footnote[i.e., no existe otro atlas diferenciable que contenga propiamente a $cal(D)$, lo cual desambigua la referencia.]
]
#defn([variedad diferenciable @munozEstimacionNoParametrica2011[Def. 3.1.4]])[
  Una variedad diferenciable de dimensión $d$ es una terna $(MM, tau, cal(D))$ donde $(MM, tau)$ es una variedad topológica de dimensión $d$ y $cal(D)$ una estructura diferenciable.
]

Una variedad diferenciable entonces, es aquella en la que la operación de diferenciación tiene sentido no solo punto a punto, sino globalmente. De no poder diferenciar, tampoco podremos tomar integrales, y definir funciones de densidad - ni hablar de estimarlas - resulta imposible.

Sobre una variedad diferenciable, cobra sentido plantear el concepto de _métrica_. En particular, toda variedad diferenciable admite una "métrica de Riemann" @docarmoRiemannianGeometry1992[§1, Proposición 2.10].

#defn(["métrica Riemanniana" @docarmoRiemannianGeometry1992[§1, Def. 2.1]])[
  Sea $T_p MM$ el _espacio tangente_ a un punto $p in MM$. Una métrica Riemanniana -  o estructura Riemanniana  - en una variedad diferenciable $MM$ es una correspondencia que asocia a cada punto $p in MM$ un producto interno $dotp(dot, dot)$ (i.e., una forma bilinear simétrica definida positiva) en el espacio tangente $T_p MM$ que "varía diferenciablemente" #footnote[para el lector riguroso, el texto original define precisamente el sentido de esta expresión] en el entorno de $p$.

  A dicho producto interno se lo denomina $g_p$ e induce naturalmente una norma: $norm(v)_p= sqrt(op(g_p)(v, v)) = sqrt(dotp(v, v))$. Decimos entonces que $g_p$ es una métrica Riemanniana y el par $(MM, g)$ es una variedad de Riemann.
] <metrica-riemanniana>

#figure(image("img/Tangent_plane_to_sphere_with_vectors.svg", height: 12em), caption: flex-caption(
  [Espacio tangente  $T_p MM$ a una esfera $MM = S^2$ por $p$. Nótese que el espacio tangente varía con $p$, pero siempre mantiene la misma dimensión ($d=2$) que $MM$],
  [Espacio tangente en $S^2$],
))

#obs(
  [según @docarmoRiemannianGeometry1992[Prop. 2.10]],
)[
  *Toda variedad diferenciable admite una métrica Riemanniana*, que se puede construir componiendo las métricas Riemannianas locales a cada carta de su estructura diferenciable según la "partición de la unidad"
  #footnote[
    La definición formal de "partición de la unidad" se da sin prueba de existencia en @docarmoRiemannianGeometry1992[§0.5, p. 30]. A cada entorno $U_alpha$ de la cobertura de #MM se le asigna una función $f_alpha$ de manera que $sum_alpha f_alpha (p) = 1 forall p in MM$. Intuitivamente, da una base funcional de #MM, que al ser evaluadas en cualquier punto ponderan con pesos que suman 1 las métricas locales a cada carta para obtener un resultado global coherente.
  ]
  ${f_alpha : alpha in cal(I)}$ subordinada a su cobertura.

  Es claro que podemos definir una métrica Riemanniana $dotp(dot, dot)^alpha$ en cada entorno $U_alpha$ de la cobertura: la métrica inducida por el sistema de coordenadas locales. Sea entonces:
  $
    dotp(u, v)_p = sum_alpha f_alpha (p) dotp(u, v)_p^alpha quad forall p in MM, thick u,v in T_p MM
  $
  es posible verificar que esta construcción define una métrica Riemanniana en todo #MM.
]

#obs[ Cuando $MM=RR^d$, el espacio es constante e idéntico a la variedad: $forall p in RR^d, thick T_p RR^d = RR^d$. La base canónica de $T_p RR^d = RR^d$ formada por las columnas de $bu(I)_d$ es una matriz positiva definida que da lugar al producto interno "clásico" $dotp(u, v) = u^T bu(I)_d v = sum_(i=1)^d u_i v_i$. $dotp(u, v)$ es una métrica Riemanniana que induce la norma euclídea $norm(v) = sqrt(v^T v)$ y la distancia $d(x, y) = norm(x-y)$.]

=== Geodésicas y mapa exponencial
Con las definiciones previas podemos definir algunos conceptos fundamentales como longitud, distancia y geodésica en variedades de Riemann.

#defn("longitud de una curva")[
  Sea $gamma : [a, b] -> MM$ una _curva diferenciable_ en #MM, y $gamma'$ su derivada. La _longitud_ de $gamma$ está dada por
  $
    L(gamma) = integral_a^b norm(gamma'(t)) dif t = integral_a^b sqrt(op(g_(gamma(t)))(gamma'(t), gamma'(t))) dif t
  $] <longitud>
#defn("distancia en variedades de Riemann")[
  Sea $(MM, g)$ una variedad de Riemann, y $p, q in MM$ dos puntos. Definimos la distancia entre ellos inducida por la métrica $g$ como
  $
    dg(p, q) = inf_(gamma) thick {L(gamma) : thick thick gamma: [0, 1] -> MM, thick gamma(0)=p,thick gamma(1)=q}
  $
]
Una _geodésica_ es una generalización de la "línea recta" en la geometría euclídea. Considérese la siguiente analogía #footnote[Este párrafo y el que sigue están adaptados de "El Flujo Geodésico" @docarmoRiemannianGeometry1992[§3.2]]: en la física clásica, un objeto que no es sujeto a ninguna fuerza (no recibe _aceleración_ alguna) estará o quieto (con velocidad nula) o en movimiento _rectilíneo_ uniforme ("MRU"). En variedades diferenciables, las geodésicas son exactamente eso: curvas sin aceleración, $gamma''(t) = 0 forall t$. En particular, una geodésica es localmente minimizante de longitud: la curva $gamma$ que realiza la distancia $dg(p, q)$ es necesariamente una geodésica.

Sea $p in MM$ y $v in T_p MM$ un vector tangente en $p$, que interpretamos como una _velocidad inicial_: su dirección $v slash norm(v)$ indica hacia dónde partir y su magnitud $norm(v)$ indica cuán rápido. Por existencia y unicidad de soluciones de ecuaciones diferenciales, existe una única geodésica $gamma$ con $gamma(0) = p$ y $gamma'(0) = v$. Como $gamma''(t) = 0 forall t$, la rapidez a lo largo de $gamma$ es constante: $norm(gamma'(t)) = norm(v) forall t$, de modo que $L(gamma) = integral_0^1 norm(gamma'(t)) dif t = norm(v)$. Tras una unidad de tiempo, la geodésica alcanza el punto $gamma(1) in MM$, habiendo recorrido una longitud $norm(v)$.

Esta relación, entre vectores de $T_p MM$ y geodésicas de $MM$ con origen en $p$, nos permite relacionar una "bola" en $T_p MM$ con su análogo en $MM$.

#defn("mapa exponencial")[
  Sean $p in MM, v in T_p MM$. Se conoce como _mapa exponencial_ a la función
  $ exp_p (v) : T_p MM -> MM = gamma_(p,v)(1) $
  donde $gamma_(p,v)(t)$ es la única geodésica que en el instante $t=0$ pasa por $p$ con velocidad $v$.
]

#defn("bola normal")[
  Sea $B_epsilon (x) subset RR^d$ la bola cerrada de radio $epsilon$ centrada en $x$:
  $ B_epsilon (x) = {y in RR^d : dg(x, y) = norm(x - y) <= epsilon} $
  Si $exp_p$ es un difeomorfismo  en un vecindario (entorno) $V$ del origen en $T_p MM$, su imagen $U = exp_p (V)$ es un "vecindario normal" de $p$.
  Si $B_epsilon (0)$ es tal que $overline(B_epsilon (0)) subset V$, llamamos a $exp_p B_epsilon (0) = B_epsilon (p)$ la _bola normal_ – o "bola geodésica" - con centro $p$ y radio $epsilon$.
]
La frontera de $B_epsilon (p)$ es una "subvariedad" de #MM ortogonal a las geodésicas que irradian desde $p$. Una concepción intuitiva de qué es una bola normal, es "un entorno de $p$ en el que las geodésicas que pasan por $p$ son minimizadoras de distancias". El siguiente concepto es útil para entender "cuán lejos vale" la aproximación local a un espacio euclídeo en la variedad.

#defn(
  [radio de inyectividad #footnote[Basado en @munozEstimacionNoParametrica2011[Def. 3.3.16] Una definición a mi entender más esclarecedora se encuentra en @docarmoRiemannianGeometry1992[§13.2, _The cut locus_], que introducimos aquí informalmente. El _cut locus_ o _ligne de partage_ $C_m (p)$ - algo así como la línea de corte - de un punto $p$ es la unión de todos los puntos de corte: los puntos a lo largo de las geodésicas que irradian de $p$ donde éstas dejan de ser minizadoras de distancia. El ínfimo de la distancia entre $p$ y su línea de corte, es el radio de inyectividad de #MM en $p$, de modo podemos escribir $ "iny" MM = inf_(p in MM) d(p, C_m (p)) $
      donde la distancia de un punto a una variedad es el ínfimo de la distancia a todos los puntos de la variedad.]],
)[
  Sea $(MM, g)$ una $d-$variedad Riemanniana. Llamamos "radio de inyectividad en $p$" a
  $
    "iny"_p MM = sup{s in RR > 0 : B_s (p) " es una bola normal"}
  $
  El ínfimo de los radios de inyectividad "puntuales", es el radio de inyectividad de la variedad #MM.
  $
    "iny"MM = inf_(p in MM) "iny"_p MM
  $
]

#obs[Si $MM = RR^d$ con la métrica canónica entonces $"iny" MM = oo$. Si $MM = RR^d - {p}$, con la métrica usual, entonces existe un punto arbitrariamente cerca de $p$ en el que la geodésica que irradia en dirección a $p$ se corta inmediatamente: entonces el radio de inyectividad es cero. Si $MM = S^1$ con radio unitario y la métrica inducida de $RR^2$, el radio de inyectividad es $pi$, puesto que si tomamos "el polo norte" $p_N$ como origen de un espacio tangente $T_p_N S^1$, todas las geodésicas que salen de él llegan al polo sur $p_S$ "al mismo tiempo" $pi$, y perdemos la inyectividad.]

#figure(caption: flex-caption(
  [Espacio tangente y mapa exponencial para $p_N in S^1$. Nótese que $"iny" S^1 = pi$. Prolongando una geodésica  $gamma(t)$ más allá de $t = pi$, ya no se obtiene un camino mínimo, pues hubiese sido más corto llegar por $-gamma(s), thick s = t mod pi$.],
  [Espacio tangente y mapa exponencial para $p_N in S^1$],
))[#image("img/mapa-exponencial-s1.svg")]


Agregamos una última definición para restringir la clase de variedades de Riemann que nos interesará:

#defn(
  "punto límite",
)[Un punto $x$ es límite del conjunto $S$ si toda vecindad abierta de $x$ contiene puntos de $S$ distintos de $x$.]

#v(-1em)

#defn("variedad compacta")[
  Decimos que una variedad es _acotada_ cuando $sup_((p, q) in MM^2) dg(p, q) = overline(d) < oo$ --- i.e., no posee elementos distanciados infinitamente entre sí.
  Una variedad que incluya todos sus "puntos límite" es una variedad _cerrada_. Una variedad cerrada y acotada se denomina _compacta_.
]

#obs[
  Un círculo en el plano, $S^1 subset RR^2 = {(x, y) : x^2 + y^2 = 1}$ es una variedad compacta: es acotada --- ninguna distancia es mayor a medio gran círculo, $pi$ --- y cerrada. $RR^2$ es una variedad cerrada pero no acotada. El "disco sin borde" ${(x, y) in RR^2 : x^2 + y^2 < 1}$ es acotado pero no cerrado - pues no incluye su frontera $S^1$. El "cilindro infinito" ${(x, y, z) in RR^3 : x^2 + y^2 < 1}$ no es ni acotado ni cerrado.
]

Ahora sí, hemos arribado a un objeto lo suficientemente "bien portado" para soportar funciones diferenciables, una noción de distancia y todo aquello que precisamos para definir elementos aleatorios: la _variedad de Riemann compacta sin frontera_. Cuando hablemos de una variedad de Riemann sin calificarla, nos referiremos a esta.


=== Probabilidad en Variedades
Hemos definido una clase bastante general de variedades - las variedades de Riemann - capaces de soportar funciones de densidad y sus estimaciones @pelletierKernelDensityEstimation2005. Estos desarrollos relativamente modernos no constituyen el origen de la probabilidad en variedades. Mucho antes de su sistematización, ciertos casos particulares fueron ya bien estudiados y allanaron el camino para el interés en variedades más generales.

Probablemente la referencia más antigua a un elemento aleatorio en una variedad distinta a $RR^d$, se deba a Richard von Mises, en _Sobre la naturaleza entera del peso atómico y cuestiones relacionadas_ @vonmisesUberGanzzahligkeitAtomgewicht1918 #footnote["Über die 'ganzzahligkeit der' atomgewichte und verwandte fragen", en el alemán original]. En él, von Mises se plantea si los pesos atómicos - que empíricamente se observan siempre muy cercanos a la unidad para los elementos más livianos - son enteros con un cierto error de medición, y argumenta que para tal tratamiento, el "error gaussiano" clásico es inadecuado:

#quote(attribution: [traducido de @vonmisesUberGanzzahligkeitAtomgewicht1918])[
  ($dots$) Pues no es evidente desde el principio que, por ejemplo, para un peso atómico de $35,46$ (Cl), el error sea de $+0,46$ y no de $-0,54$: es muy posible que se logre una mejor concordancia con ciertos supuestos con la segunda determinación. A continuación, se desarrollan los elementos — esencialmente muy simples — de una "teoría del error cíclico", que se complementa con la teoría gaussiana o "lineal" y permite un tratamiento completamente inequívoco del problema de la "enteridad" y cuestiones similares.
]

#figure(
  image("img/von-mises-s1.png", height: 16em),
  caption: flex-caption(
    [Pretendido "error" - diferencia módulo 1 - de los pesos atómicos medidos para ciertos elementos sobre $S^1$. Nótese como la mayoría de las mediciones se agrupan en torno al $0.0$. Fuente: @vonmisesUberGanzzahligkeitAtomgewicht1918],
    [Pesos atómicos "módulo 1" sobre $S^1$],
  ),
)
Motivado también por un problema del mundo físico, Ronald Fisher escribe "Dispersiones en la esfera" @fisherDispersionSphere1957, donde desarrolla una teoría apropiada para mediciones de posición en una esfera #footnote[y como era de esperar del padre del test de hipótesis, también su correspondiente test de significancia, análogo al "t de Student".] y la ilustra a partir de mediciones de la dirección de la "magnetización termorremanenteremanente" de flujos de lava  en Islandia.
#footnote[
  Los datos que Fisher usa en la Sección 4 son mediciones de magnetismo remanenteen muestras de roca de flujos de lava islandeses, recolectadas por J. Hospers en Pembroke College, Cambridge. Cuando la lava se enfría y solidifica, los minerales ferromagnéticos (como la magnetita) se alinean con el campo magnético terrestre del momento y quedan "congelados" en esa orientación. Esto se llama magnetización termorremanente. Siglos o milenios después, se puede tomar una muestra de esa roca y medir en qué dirección apunta su magnetización residual, hecho que Fisher utiliza para "testear" si entre "su presente" y el período Cuaternario el campo magnético terrestre se invirtió --- cosa que efectivamente sucedió.
]


Dos décadas más tarde, los casos particulares de von Mises ($S^1$) y Fisher ($S^2$) fueron integrados al caso más general $S^n$ en lo que se conocería como "estadística direccional" #footnote[la $n-$ esfera $S^n$ de radio $1$ con centro en $0$ contiene exactamente a todos los vectores unitarios --- i.e., todas las _direcciones_ posibles de un vector --- en su espacio ambiente $RR^(n+1)$]. En 1975 se habla ya de _teoría de la distribución_ para la distribución von Mises - Fisher @mardiaDistributionTheoryMisesFisher1975, la "más importante en el análisis de datos direccionales". A fines de los '80 Jupp y Mardia plantean "una visión unificada de la teoría de de la estadística direccional" @juppUnifiedViewTheory1989, adaptando conceptos claves del "caso euclídeo" como las familias exponenciales y el teorema central del límite, entre otros.

Aunque el caso particular de la $n-$esfera sí fue bien desarrollado a lo largo del siglo XX, no se alcanzó un tratamiento más general de la estadística en variedades riemannianas conocidas pero arbitrarias.

=== KDE en variedades de Riemann

Un trabajo sumamente interesante a principios del siglo XXI es el de Bruno Pelletier, que se propone una adaptación directa del estimador de densidad por núcleos de @kde-mv en variedades de Riemann compactas sin frontera @pelletierKernelDensityEstimation2005. Lo presentamos directamente y ampliamos los detalles a continuación


#defn([KDE en variedades de Riemann @pelletierKernelDensityEstimation2005[Ecuación 1]])[
  Sean
  - $(MM, g)$ una variedad de Riemann compacta y sin frontera de dimensión intrínseca $d$, y $dg$ la distancia de Riemann, #footnote[mantenemos la notación del original; $d$ es un entero y #dg un operador, lo que debería evitar la confusión]
  - $K$ un _núcleo isotrópico_ en #MM soportado en la bola unitaria en $RR^d$
  - dados $p, q in MM$, $theta_p (q)$ la _función de densidad de volumen en_ #MM
  Sea #XX una muestra de $N$ observaciones de una variable aleatoria $X$ con densidad $f$ soportada en #MM
  Luego, el estimador de densidad por núcleos para $X$ es la #box[$hat(f) :MM ->RR$] que a cada $p in MM$ le asocia el valor
  $
    hat(f) (p) & = N^(-1) sum_(i=1)^N K_h (p,X_i) \
               & = N^(-1) sum_(i=1)^N 1/h^d 1/(theta_X_i (p))K((dg(p, X_i))/h)
  $
] <kde-variedad>
con la restricción de que la ventana $h <= h_0 <= "iny" MM$, el radio de inyectividad de #MM. #footnote[
  Esta restricción no es catastrófica. Para toda variedad compacta, el radio de inyectividad será estrictamente positivo @munozEstimacionNoParametrica2011[Prop. 3.3.18]. Como además $h$ es en realidad una sucesión ${h_n}_(n=1)^N$ decreciente como función del tamaño muestral, siempre existirá un cierto tamaño muestral a partir del cual $h_n < "iny" MM$.
].
El autor prueba la convergencia en $L^2(MM)$:

#thm([convergencia de $hat(f)$ en $L^2$ @pelletierKernelDensityEstimation2005[§3 Teorema 5]])[
  Sea $f$ una densidad de probabilidad dos veces diferenciable en #MM con segunda derivada covariante acotada. Sea $hat(f)_n$ el estimador de densidad definido en @kde-variedad con ventana $h_n < h_0 < "iny" MM$. Luego, existe una constante $C_f$ tal que
  $
    EE norm(hat(f)_n - f)_(L^2(MM))^2 <= C_f (1/ (n h^d)+ r^4).
  $
  En consecuencia, para $h tilde n^(-1/(d+4))$, tenemos $ EE norm(hat(f)_n - f)_(L^2(MM))^2 = O(n^(-4/(d+4))) $
]
Nótese que esta formulación sugiere en qué orden comenzar la búsqueda de un $h$ "óptimo". Guillermo Henry y Daniela Rodríguez prueban la consistencia fuerte de $hat(f)$ @henryKernelDensityEstimation2009[Teorema 3.2]: bajo los mismos supuestos de @pelletierKernelDensityEstimation2005, obtienen que
$
  sup_(p in MM) abs(hat(f)_n(p) - f(p)) attach(->, t: "c.s.") 0
$

#defn("núcleo isotrópico")[ Sea $K: RR_+ -> RR$ un mapa no-negativo tal que:
  #table(
    align: (left, right),
    stroke: none,
    columns: 2,
    $integral_(RR^d) K(norm(x)) dif lambda(x) = 1$, [$K$ es función de densidad en $RR^d$],
    $integral_(RR^d) x K(norm(x)) dif lambda(x) = 0$, [Si $Y~K, thick EE Y = 0$],
    $integral_(RR^d) norm(x)^2 K(norm(x)) dif lambda(x) < oo$, [Si $Y~K, thick var Y < oo$],
    $sop K = [0, 1]$, "",
    $sup_x K(x) = K(0)$, [$K$ se maximiza en el origen],
  )

  Decimos entonces que el mapa $RR^d in.rev x |-> K(norm(x)) in RR$ es un _núcleo isotrópico_ en $RR^d$ soportado en la bola unitaria.
]

#obs[Todo núcleo válido en @kde-mv también es un núcleo isotrópico. A nuestros fines, continuaremos utilizando el núcleo normal.]
#defn(
  [función de densidad de volumen @besseManifoldsAllWhose1978[§6.2]],
)[
  Sean $p, q in MM$; le llamaremos _función de densidad de volumen_ en #MM a $theta_p (q)$ definida como
  $
    theta_p (q) : q |-> theta_p (q) = mu_(exp_p^*g) / mu_g_p (exp_p^(-1)(q))
  $
  es decir, el cociente de la medida canónica de la métrica  Riemanniana $exp_p^*$ sobre $T_p MM$ (la métrica _pullback_ que resulta de transferir $g$ de $MM$ a $T_p MM$ a través del mapa exponencial $exp_p$), por la medida de Lebesgue de la estructura euclídea en $T_p MM$.
] <vol-dens>

#obs[

  $theta_p (q)$ está bien definida "cerca" de $p$: por ejemplo, es idénticamente igual a $1$ en el entorno $U$ localmente "plano" de $p$ donde las geodésicas $gamma subset MM$ coinciden con sus representaciones en $T_p MM$. Ciertamente está definida para todo $q$ dentro del radio de inyectividad de $p$, $dg(p, q) < "iny"_p MM$ #footnote[ su definición global es compleja y escapa al tema de esta monografía. Besse y Pelletier consideran factible extenderla a todo #MM utilizando _campos de Jacobi_ @besseManifoldsAllWhose1978[§6.3] @pelletierKernelDensityEstimation2005[§2].]. Con $N$ "suficientemente grande", siempre podremos elegir $h_N < "iny"_p MM$  que mapee "suficientes" observaciones al soporte de K, $[0, 1]$  en las que el cálculo de $theta_p (q)$ sea factible, y las más lejanas queden por fuera, de modo que su cálculo _no sea necesario_.
]


El mapa exponencial alrededor de $p, thick exp_p : T_p MM -> MM$ es un difeomorfismo en cierta bola normal alrededor de $p$, así que admite una inversa continua y biyectiva al menos en tal bola; lo llamaremos $exp_p^(-1) : MM -> T_p MM$. Así, $exp_p^(-1) (q) in T_p MM$ es la representación de $q$ en las coordenadas localmente euclídeas del espacio tangente a $p$ (o sencillamente "locales a $p$"). De esta cantidad $x = exp_p^(-1) (q)$, queremos conocer el cociente entre dos medidas:
- la métrica _pullback_ de $g$:  la métrica inducida en $T_p MM$ por la métrica riemanniana $g$ en #MM
- la medida de Lebesgue en la estructura euclídea de $T_p MM$.

En otras palabras, $theta_p (q)$ representa cuánto se infla/encoge el espacio en la variedad #MM alrededor de $p$, relativo al volumen "natural" del espacio tangente. En general, su cómputo resulta sumamente complejo, salvo en casos particulares como las variedades "planas" o de curvatura constante.

=== Densidad de volumen en la esfera

#obs(
  [@besseManifoldsAllWhose1978[§6.2]],
)[En una variedad plana, $theta_p (q)$ es idénticamente igual a 1 para todo $p, q in MM$.]

Una variedad plana tiene _curvatura_ #footnote[la _curvatura_ de un espacio es una de las propiedades fundamentales que estudia la geometría riemanniana; en este contexto, basta con la comprensión intuitiva de que una variedad no-plana tiene _cierta_ curvatura] nula en todo punto. De entre las variedades curvas, las $n-$ esferas son de las más sencillas, y tienen curvatura _positiva y constante_. Esta estructura vuelve posible el cómputo de $theta_p (q)$ en $S^n$.

En _Kernel Density Estimation on Riemannian Manifolds: Asymptotic Results_ @henryKernelDensityEstimation2009, Guillermo Henry y Daniela Rodriguez estudian algunas propiedades asintóticas del estimador de @kde-variedad, y las ejemplifican con datos de sitios volcánicos en la superficie terrestre. Para ello, calculan $theta_p (q)$ en $S^2$ y llegan a que #footnote[Recordemos que la antípoda de $p, -p$ cae justo fuera de $"iny"_p S^d$]

#v(1em)
$
  theta_p (q) = cases(
    R abs(sin(dg(p, q) slash R)) / dg(p, q) & "si" q != p\, -p,
    1 & "si" q = p
  )
$

#v(1em)

#figure(caption: flex-caption(
  [Densidad estimada de sitios volcánicos en la superficie terrestre ($approx S^2$) para distintos valores de $h$. Fuente: @henryKernelDensityEstimation2009],
  [Densidad estimada en $S^2$ para distintos valores de $h$],
))[#image("img/henry-rodriguez-bolas.png", height: 22em)]

Para variedades de curvatura variable, el cálculo es mucho más complejo. En un trabajo reciente, por ejemplo, se reseña:

#quote(
  attribution: [@berenfeldDensityEstimationUnknown2021[§1.2, "Resultados Principales"]],
)[
  Un problema restante a esta altura es el de entender cómo la _regularidad_ #footnote[En este contexto, se entiende que una variedad es más regular mientras menos varíe su densidad de volumen punto a punto.] de #MM afecta las tasas de convergencia de funciones suaves (...).
  Luego, en el caso especial en que la dimensión de #MM es conocida e igual a $1$, podemos construir un estimador que alcanza la tasa [propuesta anteriormente]. Así, se establece que en dimensión $1$ al menos, la regularidad de la variedad #MM no afecta la tasa para estimar $f$ aún cuando #MM es desconocida. Sin embargo, la función de densidad de volumen $theta_p (q)$ _no_ es constante tan pronto como $d >= 2$ y obtener un panorama global en mayores dimensiones es todavía un problema abierto y presumiblemente muy desafiante.
]

== Clasificación en variedades

Un desarrollo directo del estimador de @kde-variedad consta en _A kernel based classifier on a Riemannian manifold_ @loubesKernelbasedClassifierRiemannian2008,
donde los autores construyen un clasificador para un objetivo de dos clases $GG in {0, 1}$ con #emph[inputs] $X$ soportadas sobre una variedad de Riemann. A tal fin, minimizan la pérdida $0-1$ y siguen la regla de Bayes, de manera que su clasificador _duro_ resulta:

$
  hat(G)(X) = cases(1 "si" hat(Pr)(G=1|X) > hat(Pr)(G=0|X), 0 "si no")
$
que está de acuerdo con el estimador del clasificador de Bayes basado en densidad por núcleos para $K$ clases propuesto @kdc-duro.

Una notación simplificada surge de estudiar la expresión que el clasificador intenta maximizar. Para todo $k in [K]$,
$
  hat(Pr)(G=k|X) &= (hat(f)_k (x) times hat(pi)_k) / underbrace((sum_(k in [K]) hat(f)_k (x) times hat(pi)_k), =c) = c^(-1) times hat(f)_k (x) times hat(pi)_k
$
de modo que la tarea es equivalente a maximizar $hat(f)_k (x) times hat(pi)_k$ sobre $k in [K]$. Si $N_k$ es la cantidad de observaciones en la clase $k$ y  $sum_k N_k = N$, podemos reescribir el estimador de densidad de la clase $k$ como:
$
  hat(f)_k (x) & = N_k^(-1) sum_(i=1)^N_k K_h (x,X_i) \
               & = (sum_(i=1)^N ind(G_i = k) K_h (x,X_i)) / (sum_(i=1)^N ind(G_i = k)) \
$
como además $hat(pi)_k = N_k slash N =N^(-1) sum_(i=1)^N ind(G_i = k)$, resulta que
$
  hat(f)_k (x) times hat(pi)_k& = (sum_(i=1)^N ind(G_i = k) K_h (x,X_i)) / (sum_(i=1)^N ind(G_i = k)) times (sum_(i=1)^N ind(G_i = k)) / N \
  & = N^(-1) sum_(i=1)^N ind(G_i = k) K_h (x,X_i)
$
Y suprimiendo la constante $N$ concluimos que la regla de clasificación resulta equivalente a:
$
  hat(G)(p) = arg max_(k in [K]) sum_(i=1)^N ind(G_i = k) K_h (p,X_i)
$ <clf-kde-variedad>
para todo $p in MM$ con $K_h_n$ un núcleo isotrópico con sucesión de ventanas $h_n$ @loubesKernelbasedClassifierRiemannian2008[Ecuación 3.1].

La belleza de esta regla, es que combina "sin costuras" el peso de los _priors_ $hat(pi)_i$ - a través de los elementos no nulos de la suma cuando $ind(G_i = k) = 1$) - con el peso de la "evidencia" - vía su cercanía "suavizada" al punto de interés $K_h (p, X_i)$.

Los autores toman de @devroyeProbabilisticTheoryPattern1996 la siguiente definición de _consistencia_:

#defn([consistencia de un clasificador @devroyeProbabilisticTheoryPattern1996[§6.1]])[
  Sea $hat(G)_1, dots, hat(G)_n$ una secuencia de clasificadores #footnote[A veces también llama una _regla_ de clasificación] de modo que el $n-$ésimo clasificador está construido con las primeras $n$ observaciones de la muestra $XX, bu(g)$. Sea $L_n$ la pérdida $0-1$ que alcanza $hat(G)_n$, y $L^*$ la pérdida que alcanza el clasificador de Bayes de @clf-bayes.

  Diremos que la regla ${hat(G)_i}_(i=1)^n$ es (débilmente) consistente --- o asintóticamente eficiente en el sentido del riesgo de Bayes --- para cierta distribución $(X, G)$ si cuando $n-> oo$
  $
    EE L_n = Pr(hat(G)_n (X) != G) -> L^*
  $
  y fuertemente consistente si
  $
    lim_(n -> oo) L_n = L^* "con probabilidad 1"
  $
]

En el trabajo, se prueba que el clasificador de @clf-kde-variedad es fuertemente consistente para $K=2$.

== Aprendizaje de distancias

La hipótesis de la variedad nos ofrece un marco teórico en el que abordar la clasificación en alta dimensión, y encontramos en la literatura que la estimación de densidad por núcleos en variedades de Riemann compactas sin frontera está estudiada y tiene buenas garantías de convergencia. Por alentador que resulte, nos resta un problema fundamental: _no suele ser conocida la variedad que soporta las $X$_. Salvo que los datasets estén generados sintéticamente o el objeto de estudio cuente con un dominio bien entendido y formalizable, tendremos problemas tanto para definir adecuadamente la dimensión intrínseca $d_MM$ y la distancia $d_g$ en #MM.

#figure(caption: flex-caption(
  [Data espacial con dimensiones bien definidas. (izq.) Los datos geoespaciales están sobre la corteza terrestre, que es aproximadamente la $2-$esfera $S^2 in RR^3$ que representa la frontera de nuestra "canica azul" , una $3-$bola. (der.) La clasificación clásica de Hubble distingue literalmente _variedades_ "elípticas","espirales" e "irregulares" de galaxias .#footnote[La categoriación completa es más compleja, con _outliers_ cuando #link("https://astronomy.stackexchange.com/questions/32947/what-decides-the-shape-of-a-galaxy")[distintas galaxias interactúan entre sí], como las #link("https://es.wikipedia.org/wiki/Galaxias_Antennae")[Antennae]. La #link("https://en.wikipedia.org/wiki/Spacetime_topology")[topología del espacio-tiempo] es un tópico de estudio clave en la relatividad general.]],
  "Data espacial con dimensiones bien definidas.",
))[
  #grid(
    columns: (auto, auto),
    column-gutter: 1em,
    image("img/blue-marble.jpg", height: 20%), image("img/tipos-de-galaxia-secuencia-hubble.png"),
  )
]

#let reddot = math.class("normal", circle(radius: 2.5pt, fill: red.lighten(60%), stroke: 1pt + red))
#let greendot = math.class("normal", circle(radius: 2.5pt, fill: green.lighten(60%), stroke: 1pt + green))
#let yellowdot = math.class("normal", circle(radius: 2.5pt, fill: yellow.lighten(60%), stroke: 1pt + yellow))

Considere, por caso, el diagrama de @variedad-u, la curva $cal(U) subset RR^2, d_cal(U) =1$.

#figure(
  caption: flex-caption[La curva $cal(U), thin d_(cal(U)) = 1$ embebida en $RR^2$. En el espacio ambiente, $d(greendot, reddot thin |RR^2) < d(greendot, yellowdot thin | RR^2)$. Trasladándose _sobre_ $cal(U)$, #box[$thin d(greendot, reddot thin |cal(U)) > d(greendot, yellowdot thin | cal(U)) approx 1/2 d(greendot, reddot thin |cal(U))$] .][Variedad $cal(U)$ embebida en $RR^2$],
)[#image("img/variedad-u.svg", height: 18em)] <variedad-u>

A los fines de estimar la densidad de $X$ soportada en cierta variedad #MM, resulta clave es contar con una noción de _distancia_ apropiada en #MM, que raramente coincidirá con  la propia del espacio ambiente.

La distancia entre dos puntos $p$ y $q$ de #MM es la longitud de la curva geodésica que los une; la longitud de una curva se obtiene integrándola en toda su extensión; integrarla implica conocer el espacio tangente y la métrica $g$ en toda su extensión. Por ende, "conocer la variedad" $(MM, g) = sop X$ y "computar la distancia $dg$ inducida por su métrica $g$" son esencialmente la misma tarea.

En el ejemplo de @variedad-u, con  tan solo $n=3$ observaciones es imposible distinguir $cal(U)$, pero con una muestra #XX "suficientemente grande", es de esperar que los propios datos revelen la forma de la variedad; encima de esta observación es que se edifica la teoría de "aprendizaje de distancias" a partir de la propia muestra.

La distancia nos da entonces una _representación_ útil de cuán similares son dos puntos: a menor distancia, mayor similitud. Por ello, la estimación de variedades es fundamental al _aprendizaje de representaciones_. En una extensa reseña de dicho campo, @bengioRepresentationLearningReview2014 así lo explican:


#quote(attribution: [ @bengioRepresentationLearningReview2014[§8]])[
  (...) [L]a principal tarea del aprendizaje no-supervisado se considera entonces como el modelado de la estructura de la variedad que sustenta los datos. La representación asociada que se aprende puede asociarse con un sistema de coordenadas intrínseco en la variedad embebida.
]



=== El ejemplo canónico: Análisis de Componentes Principales (PCA)

El término "hipótesis de la variedad" es moderno, pero el concepto está presente hace más de un siglo en la teoría estadística #footnote[estas referencias vienen del mismo Bengio #link("https://www.reddit.com/r/MachineLearning/comments/mzjshl/comment/gwq8szw/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button")[comentando en Reddit sobre el origen del término]].

El algoritmo arquetípico de modelado de variedades es, como era de esperar, también el algoritmo arquetípico de aprendizaje de representaciones de baja dimensión: el Análisis de Componentes Principales, PCA @pearsonLIIILinesPlanes1901, que dada $XX in RR^(N times p)$, devuelve en orden decreciente las "direcciones de mayor variabilidad" en los datos, $bu(U)_p = (u_1, u_2, dots, u_p) in RR^(p times p)$. Proyectar $XX$ sobre las primeras $k <= p$ direcciones, $ hat(XX) = XX bu(U)_k in RR^(n times k), thick hat(X)_i = (hat(X)_(i 1), dots, hat(X)_(i k))^T $
nos devuelve la "mejor" #footnote[cuya definición precisa obviamos.] representación lineal de dimensión $k$.
#figure(
  image("img/pca.png"),
  caption: flex-caption(
    [$XX in RR^2$ y sus componentes principales. Fuente: _"On lines and planes of closest fit to systems of points in space."_ @pearsonLIIILinesPlanes1901],
    [$XX in RR^2$ y sus componentes principales.],
  ),
)

Ya se dijo que las variedades que soporten muestras #XX "silvestres" seguramente sean fuertemente no-lineales. Sin embargo, todavía hay lugar para PCA en esta aventura: cuando el dataset tiene dimensión verdaderamente muy alta, un proceso razonable consistirá en primero disminuir la dimensión a un subespacio lineal en que las distancias relativas sean casi idénticas a las del espacio original usando PCA, y recién en este subespacio aplicar técnicas más complejas de aprendizaje de distancias

Aprovechando que al menos las observaciones de entrenamiento son puntos conocidos de la variedad #footnote[_módulo_ el error de medición y/o el efecto de covariables no medidas], y que en la variedad el espacio es _localmente euclídeo_, los autores de @vincentManifoldParzenWindows2002 parten del estimador de de densidad por núcleos multivariado de @kde-mv pero en lugar de utilizar un núcleo $KH$ fijo en cada observación $x_i$, se proponen primero hacer análisis de componentes principales de la matriz de covarianza _pesada_ estimada en cada punto,
$
  hat(SS)_cal(K)_i = hat(SS)_cal(K)(x_i) = (sum_(j in [N] - i) cal(K)(x_i, x_j) (x_j - x_i) (x_j - x_i)^T )/(sum_(j in [N] - i) cal(K)(x_i, x_j))
$
donde $cal(K)$ es alguna medida de cercanía en el espacio ambiente (e.g. la densidad normal multivariada $Phi$ ya mencionada), con lo cual la estimación de densidad resulta:
$
  hat(f) (x) & = N^(-1) sum_(i=1)^N K_( hat(SS)_cal(K)_i) (x, x_i) \
             & = N^(-1) sum_(i=1)^N abs(det hat(SS)_cal(K)_i)^(-1/2) K( hat(SS)_(cal(K)_i)^(-1/2) (x - x_i))
$
Ahora bien, computar una $hat(SS)_cal(K)_i$  para cada una de las $N$ observaciones, más su inversa y la "raíz cuadrada" de ésta última es computacionalemnte muy costoso, por lo que los autores agregan un refinamiento: si la variedad en cuestión es $d-$dimensional, es de esperar que las direcciones principales a partir de la $d+1$ --- ésima sean "negligibles" #footnote[la sugerente metáfora que usan en el trabajo, es que en lugar de ubicar una "bola" de densidad alrededor de cada observación $x_i$, quieren ubicar un "panqueque" tangente a la variedad] en lugar computar las componentes principales de $hat(SS)_cal(K)_i$, simplemente:
+ fijan de antemano la dimensión $d$ esperada para la variedad,
+ se quedan con las $d$ direcciones principales #footnote[en la práctica, las obtienen usando SVD - descomposición en valores singulares @hastieElementsStatisticalLearning2009[§3, Eq. 45, p. 64]],
+ "ponen en cero" el resto y
+ "completan" la aproximación con un poco de "ruido" $sigma^2 bu(I)$.

#v(1em)

La aproximación resultante #box[$hat(SS)_i = f(hat(SS)_cal(K)_i) + sigma^2 bu(I)$] es mucho menos costosa de invertir, y tiene una interpretación geométrica bastante intuitiva en cada punto.
Usando el mismo clasificador basado en la regla de Bayes de @clf-bayes que ya mencionamos, obtienen así resultados superadores a los de @kde-mv con $HH = h^2 bu(I)$. Hemos de notar, sin embargo, dos dificultades:
- no es obvio cuál debería ser la dimensión intrínseca $d$ cuando la variedad es desconocida, y
- no es suficiente para computar KDE en variedades según @kde-variedad, pues $hat(SS)_i$ solo aproxima el tensor métrico $g$ en cada $x_i$, y para computar $theta_p (q)$ necesitamos conocer $g$ _en todo punto_. #footnote[El grupo de investigación de Bengio, Vincent, Rifai et ales continuó trabajando estos estimadores, con especial énfasis en la necesidad de aprender una geometría _global_ de la variedad para evitar el crecimiento exponencial de tamaño muestral que exigen los métodos locales como KDE en alta dimensión o variedades muy "rugosas", pero a partir de aquí su camino se desvía del de esta monografía. Una brevísima reseña de lo que _no_ cubriremos: en @bengioNonLocalManifoldParzen2005 se agregan restricciones globales a la estimación de los núcleos punto a punto y los computan simultáneamente usando redes neuronales; en @rifaiManifoldTangentClassifier2011 se aprende explícitamente un atlas que luego usan para clasificación con TangentProp @simardTangentPropFormalism1991. Este último propone una modificación del algoritmo de _backpropagation_ típico de redes neuronales, para aprender una representación que conserve las "direcciones tangentes" a las observaciones de #XX.]

En un trabajo contemporáneo a @vincentManifoldParzenWindows2002, "Charting a Manifold" @brandChartingManifold2002, el autor encara frontalmente las limitaciones recién mencionadas, en tres etapas:
+ estimar la dimensión intrínseca de la variedad $d_MM$; luego
+ definir un conjunto de cartas centradas en cada observación $x_i in MM$ que minimicen una _divergencia_ global, y finalmente
+ "coser" las cartas a través de una _conexión_ global sobre la variedad.

#v(1em)

El procedimiento para estimar $d_MM$ es ingenioso, pero costoso. Sean $XX = (x_1^T, dots, x_N^T)$ $N$ observaciones $p-$dimensionales muestreadas de una distribución en $(MM, g), d_MM < p$ con algo de ruido _isotrópico_ #footnote[Del griego _iso-_, "igual" y _-tropos_, "dirección"; "igual en todas las direcciones"] $p-$dimensional. Dada una bola $B_r (q)$ centrada en un punto cualquiera $q in #MM$, consideremos la tasa $t(r)$ a la que incorpora observaciones vecinas en a medida que crece $r$. Cuando $r$ está en la escala del ruido isotrópico, la bola incorpora puntos rápidamente, pues los hay en todas las direcciones. A medida que $r$ alcanza la escala en la que el espacio es localmente análogo a $RR^p$, la incorporación de nuevos puntos disminuye, pues solo habrá nuevas observaciones en las $d_MM$ direcciones tangentes a $q$. Si $r$ sigue creciendo la bola $B_r (q)$ eventualmente alcanzará la escala de la _curvatura_ de la variedad, momento en el que comenzará a acelerarse nuevamente la incorporación de puntos. Analizando $arg min_r t(r)$ podemos identificar la dimensión intrínseca de la variedad. #footnote[Más precisamente, el autor utiliza otra función de $r$, $c(r)$ que se _maximiza_ cuando $r approx 1/d$, y evalúa las bondades y dificultades de estimar $d_MM$ tanto punto a punto como globalmente en toda la variedad.]

#v(1em)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  image("img/scale-behavior-1d-curve-w-noise.png"),
  align(horizon, figure([], caption: flex-caption(
    [
      Una bola de radio creciente centrada en un punto de una $1-$variedad muestreada con ruido en $RR^2$ _minimiza_ la tasa a la que incorpora observaciones cuando $r$ está en la escala "localmente lineal" de la variedad.
      #v(.5em)
      Fuente: @brandChartingManifold2002[Fig. 1]
    ],
    [
      Comportamiento de escala de una $1-$variedad en $RR^2$
    ],
  ))),
)

Estimada $d_MM$, los pasos siguientes no son menos complejos. Por un lado, se plantea un sistema ecuaciones para obtener _al mismo tiempo_ todos los entornos coordenados centrados en cada observación minimizando la _divergencia_ entre $SS_j$ vecinos
#footnote[
  A tal fin, modela la muestra como una "mezcla de $N$ Gaussianas"  --- _gaussian mixture modelling_, GMM por sus siglas en inglés --, con $mu_i = x_i forall i in [N]$, y resuelve simultáneamente $SS_i forall i in [N]$. Aquí "divergencia" tiene un significado preciso que obviamos, pero intuitivamente, representa el "costo" --- la variación --- que uno encuentra cuando quiere representar un punto $a$ del vecindario $U$ de $x_i$, en las coordenadas correspondientes a un vecindario $V$ de $x_j$. Se puede mostrar que el cociente entre las densidad de $a$ en ambos sistemas coordenados --- la #link("https://en.wikipedia.org/wiki/Cross-entropy")[entropía cruzada] entre $cal(N)(x_i, SS_i)$ y $cal(N)(x_j, SS_j)$ --- es la divergencia que se busca minizar.]. Finalmente, han de encontrar una _conexión_ entre los entornos coordenados de cada observación, de manera que se puedan definir coordenadas para _cualquier_ punto de la variedad y con ellas formar un atlas diferenciable.

Una #link("https://es.wikipedia.org/wiki/Conexi%C3%B3n_af%C3%ADn")[_conexión_] es otro término de significado muy preciso en geometría riemanniana que aquí usamos coloquialmente. Es un _objeto geométrico_ que _conecta_ espacios tangentes cercanos, describiendo precisamente cómo éstos varían a medida que uno se desplaza sobre la variedad, y permite entonces _diferenciarlos_ para computar $g_p$ y la métrica inducida en cualquier punto $p in MM$. Desde ya que con tal estructura es posible calcular $theta_p (q) forall p, q in MM$, pero a esta altura, hemos reemplazado el de-por-sí difícil problema original --- encontrar una buena representación de baja dimensión #MM --- por uno _muy difícil_: encontrar la dimensión intrínseca, un atlas diferenciable y su conexión global para una variedad desconocida. El proceso es sumamente interesante, pero complejiza en lugar de simplificar nuestro desafío inicial.

=== Isomap

Recordemos que toda esta aventura comenzó cuando identificamos que
+ en alta dimensión, la distancia euclídea deja de proveer información útil sobre la similitud entre observaciones de #XX y
+ de haber una estructura de menor dimensión que represente mejor las observaciones, ésta casi seguro sea fuertemente no-lineal.

#v(1em)

En rigor, no es necesario conocer #MM para estimar densidades en ella; bastaría con conocer una aproximación a la distancia geodésica en #MM que sirva de sustituto a la distancia euclídea en el espacio ambiente. Probablemente el algoritmo más conocido a tal fin, sea Isomap - por "mapeo isométrico de _features_".

Desarrollado a fines del siglo XX por Joshua Tenenbaum et al.  @tenenbaumMappingManifoldPerceptual1997 @tenenbaumGlobalGeometricFramework2000, el algoritmo consta de tres pasos:

#defn("algoritmo Isomap")[
  Sean $XX = (x_1, dots, x_N), x_i in RR^p$ $N$ observaciones $p-$dimensionales.
  El mapeo isómetrico de _features_ es el resultado de:
  + Construir el grafo pesado de vecinos más cercanos $bu(N N) = (XX, E, W)$, donde cada observación $x_i$ es un vértice y la arista #footnote[_edge_ en inglés] $e_i = a ~ b$ que une $a$ con $b$ está presente con peso $w_i = norm(a - b)$ sí y solo si
    - ($epsilon-$Isomap): la distancia euclídea entre $a, b$ en el espacio ambiente es menor o igual a épsilon, $norm(a - b) <= epsilon$.
    - ($k-$Isomap): $b$ es uno de los $k$ vecinos más cercanos de $a$ #footnote[o viceversa, pues en un grafo no-dirigido la relación de vecinos más cercanos es mutua]
  + Computar la distancia geodésica en el grafo $bu(N N)$ --- el "costo" de los caminos mínimos --- entre todo par de observaciones, $d_bu(N N)(a, b) forall a, b in XX$ #footnote[A tal fin, se puede utilizar segón convenga el algoritmo de #link("https://es.wikipedia.org/wiki/Algoritmo_de_Floyd-Warshall")[Floyd-Warshall] o #link("https://es.wikipedia.org/wiki/Algoritmo_de_Dijkstra")[Dijkstra]].
  + Construir la representación - $d-$dimensional utilizando MDS #footnote["Multi Dimensional Scaling", o #link("https://es.wikipedia.org/wiki/Escalamiento_multidimensional")[_escalamiento multidimensional_], un algoritmo de reducción de dimensionalidad] en el espacio euclídeo $RR^d$ que minimice una métrica de discrepancia denominada "estrés", entre las distancias $d_bu(N N)$ de (2) y la norma euclídea en la representación. Para elegir el valor óptimo de $d$ #footnote[que debería coincidir con la dimensión intrínseca de los datos], búsquese el "codo" en el gráfico de estrés en función de la dimensión de MDS.
]
#figure(
  image("img/isomap-2.png", height: 16em),
  caption: flex-caption(
    [Isomap aplicado a 1.000 dígitos "2" manuscritos del dataset _MNIST_ con $d=2$ Nótese que las dos direcciones se corresponden fuertemente con características de los dígitos: el rulo inferior en el eje $X$, y el arco superior en el eje $Y$. Fuente: @tenenbaumGlobalGeometricFramework2000.],
    [Isomap ($d=2$) aplicado 1.000 dígitos "2" manuscritos],
  ),
)

La pieza clave del algoritmo, es la estimación de la distancia geodésica en #MM a través de la distancia en el grafo de vecinos más cercanos. Si la muestra disponible es "suficientemente grande", es razonable esperar que en el entorno de una observación $x_0$ las distancias euclídeas aproximen bien las distancias geodésicas, y por ende un "paseo" por el grafo $bu(N N)$ debería describir una curva prácticamente contenida en #MM. Isomap resultó ser un algoritmo sumamente efectivo que avivó el interés por el aprendizaje de distancias, pero todavía cuenta con un talón de Aquiles: la elección del parámetro de cercanía, $epsilon$ ó $k$:
- valores demasiado pequeños pueden "partir" $bu(N N)$ en más de una componente conexa, otorgando distancia "infinita" a puntos en componentes disjuntas, mientras que
- valores demasiado grandes pueden "cortocircuitar" la representación - en particular en variedades con muchos pliegues -, uniendo secciones de la variedad subyacente a través del espacio ambiente.

=== Distancias basadas en densidad

Algoritmos como Isomap aprenden la _geometría_ de los datos, reemplazando la distancia euclídea ambiente por la distancia geodésica en el grafo pesado $bu(N N)_k$ #footnote[donde el subíndice representa la cantidad de vecinos considerados - o el diámetro $epsilon$ de la vecindad, de corresponder.], que con $n -> oo$ converge a la distancia $dg$ en $MM$. En estadística, conocer la geometría del soporte no es suficiente para tener un panorama completo. Por caso: sean $X'$ y $X^*$ dos distribuciones aleatorias soportadas en la esfera $S^2$:
- $X'$ surgida de _samplear_ uniformemente "coordenadas polares" en el rectángulo $[0, pi] times [0, 2 pi]$, y
- $X^*$ surgida de samplear uniformemente directamente en $S^2$.
Ambas distribuciones tienen la misma geometría, pero distintas densidades: $X'$ se concentra en los polos y es mínimamente densa en el ecuador; $X^*$ es efectivamente igual de densa en todo $S^2$.

Un ejemplo aún más concreto: sea $Omega$ la población de alumnos de nuestra facultad, y tomemos $X(Omega) = (X_1, X_2)$ con
$
  X_1(omega) & = "edad de " omega \
  X_2(omega) & = "cantidad de cabellos de " omega \
$
Es cierto que $sop(X) = RR^2$, pero resulta patente que la tasa de variación en ambas dimensiones _no es_ la misma: una decena de años es una diferencia de edad significativa, mientras que una decena de cabellos faltantes es invisible a cualquiera #footnote[salvo, seguramente, a quien los haya perdido].

Conocer la _densidad_ de los datos en la geometría es crucial para obtener una noción de distancia verdaderamente útil: de esta necesidad surge el estudio de las _distancias basadas en densidad_ #footnote[o "DBD", por sus siglas en inglés: _density-based distance (metric)_]: su premisa básica es computar la longitud de una curva $gamma$ integrando una función de costo inversamente proporcional a la densidad $f_X$ en #MM --- más "costosa" en regiones menos densas. Esta área del aprendizaje de distancias vio considerables avances durante el siglo XXI --- luego del éxito empírico de Isomap --, y pavimentó el camino para técnicas de reducción de dimensionalidad basales en el "aprendizaje profundo" #footnote[  O "deep learning" en inglés. Llamamos genéricamente de tal modo a la plétora de arquitecturas de redes neuronales con múltiples capas que dominan hoy el procesamiento de información de alta dimensión @AprendizajeProfundo2025] como los "autocodificadores" #footnote[#emph[autoencoders] en inglés, algoritmo que dada #XX, aprende un codificador $c(x): RR^D -> RR^d, d << D$ y un decodificador $d(x) : RR^d -> RR^D$ tal que $d(c(x)) approx x$.
]. Yoshua Bengio --- uno de los "padres de la IA" cuyo trabajo ya mencionamos en esta monografía --, menciona #link("https://www.reddit.com/r/MachineLearning/comments/mzjshl/d_who_first_advanced_the_manifold_hypothesis_to/", "en Reddit") cómo su grupo de investigación en la Universidad de Montréal trabajando en estas ideas: aprendizaje de variedades primero, y autocodificadores posteriormente.

#quote(attribution: "Y. Bengio")[
  El término hipótesis de la variedad es en efecto más antiguo que la revolución del aprendizaje profundo, aunque el concepto ya estaba presente en los primeros días de los autoencoders en los primeros años de los 90 (no bajo ese nombre, pero la misma idea) y los mapas autoorganizados en los 80, por no mencionar PCA aún antes (aunque eso estaba limitado a variedades lineales). Y el grupo a mi alrededor en la U. de Montreal en la década del 2000 y principios de la del 2010 trabajó bastante sobre el concepto, en el contexto de modelar distribuciones que se concentran cerca de un conjunto de menor dimensión (es decir, una variedad), por ejemplo, con denoising auto-encoders (trabajo liderado por Pascal Vincent) y contractive auto-encoders (liderado por Salah Rifai). También trabajamos en cómo la hipótesis de la variedad impactaba los modelos generativos y la dificultad de muestrear (y cómo muestrear) cuando hay múltiples variedades alejadas entre sí (el problema de mezcla en MCMC).
]
Aprender una DBD nos permite saltearnos el problema ya harto descrito de aprender la variedad desconocida #MM, e ir directamente a lo único estrictamente necesario para tener un algoritmo de clasificación funcional: una noción de distancia adecuada.

@vincentDensitySensitiveMetrics2003 proveen una de las primeras heurísticas para una DBD: al igual que Isomap, toma las distancias de caminos mínimos pesados en un grafo con vértices #XX, pero
- considera el grafo completo $bu(C)$ en lugar del de $k-$vecinos $bu(N N)_k$ y
- pesa las aristas del grafo por la distancia euclídea en el espacio ambiente entre sus extremos _al cuadrado_.

Esta noción de "distancia de arista-al-cuadrado" #footnote["edge-squared distance" en inglés] tiene el efecto de desalentar grandes saltos entre observaciones lejanas, que es una manera  de "asignar un costo alto a trayectos por regiones de baja densidad", por lo cual ya califica como una DBD  rudimentaria.

#figure(image("img/distancia-cuadrada.svg", height: 16em), caption: flex-caption(
  [En este grafo completo de 3 vértices "isósceles", hay solo dos caminos entre $a$ y $c$: $zeta = a -> b -> c$, y $gamma = a -> c$],
  [Grafo completo de 3 vértices en disposición "isósceles"],
)) <grafo-completo-3-vertices>

Consideremos el grafo _planar_ de @grafo-completo-3-vertices. Bajo la con la norma euclídea, $L(gamma) = 3 < 4 = 2 + 2 = L(zeta)$ de modo que $d(a, c) = 3$ con geodésica $gamma$. Con la distancia de arista-al-cuadrado, $L(zeta) = 2^2 + 2^2 = 8 < 3^2 = L(gamma)$, y por lo tanto $d(a, c) = 8$ con geodésica $zeta$. La distancia de arista-al-cuadrado cambia las geodésicas, y también cambia la escala en que se miden las distancias.


En las dos últimas décadas han surgido numersos algoritmos para calcular DBDs y hasta algunos _surveys_ comparando las bondades relativas de cada una, nos detendremos arbitrariamente en algunos. En @caytonAlgorithmsManifoldLearning2005 el autor provee un resumen de los algoritmos de aprendizaje de variedades más relevantes (hasta entonces). En sus reflexiones finales #footnote[cf. @caytonAlgorithmsManifoldLearning2005[§5, "¿Qué queda por hacer?"], la lectura de cuyas conclusiones al menos --- si no el trabajo entero --- recomiendo. ], el autor considera que es tan amplio el espectro de variedades subyacentes y de representaciones "útiles" que se pueden concebir, que (a) en el plano teórico resulta muy difícil de obtener garantías de eficiencia y performance, y (b) en el plano experimental, quedamos reducidos a elegir un conjunto representativo de variedades y observar si los resultados obtenidos son "intuitivamente agradables". Más aún, las evaluaciones experimentales requieren _conocer_ la variedad subyacente para luego evaluar si el algoritmo de aprendizaje preserva información útil. Determinar si un dataset del mundo real efectivamente yace sobre cierta variedad es tan difícil como aprender la variedad; usar datos artificiales puede no rendir resultados realistas. Veintiún años más tarde, en esta monografía nos topamos con las mismas dificultades de antaño.

A nuestro entender, @bijralSemisupervisedLearningDensity2012 ofrece una de las primeras formalizaciones de qué constituye una DBD. Para abordarla, revisaremos una definición previa. En @longitud definimos la longitud de una curva $gamma$ parametrizada y diferenciable sobre una variedad de Riemann compacta y sin frontera $(MM, g)$.

#defn(
  "curva rectificable",
)[Una _curva rectificable_ es una curva que tiene longitud finita. Más formalmente, sea $gamma: [a,b] -> MM$ una curva parametrizada. La curva es rectificable si su longitud de arco es finita:

  $ L(gamma) = sup sum_(i=1)^n |gamma(t_i) - gamma(t_(i-1))| < infinity $

  donde el supremo se toma sobre todas las particiones posibles $a = t_0 < t_1 < ... < t_n = b$ del intervalo $[a,b]$.

  Equivalentemente, si $gamma$ es diferenciable por tramos, entonces es rectificable si y solo si:

  $ L(gamma) = integral_a^b |gamma'(t)| dif t < infinity $
]

Las curvas rectificables son importantes porque permiten definir conceptos como la longitud de arco y la parametrización por longitud de arco, que son fundamentales en geometría diferencial y análisis. En particular, sea $gamma: [a,b] -> RR^n$ una curva rectificable parametrizada y diferenciable por tramos y $f: RR^n -> RR$ una función diferenciable. La "integral de línea" #footnote[_line integral_ en inglés] de $f$ sobre $gamma$ se define como:

$ integral_gamma f dif s = integral_a^b f(gamma(t)) |gamma'(t)| dif t $

donde $dif s$ representa el elemento de longitud de arco.

Si $gamma$ tiene longitud finita y $f$ es continua --- como en nuestro caso de uso --, el resultado de la integral *existe y es independiente de la parametrización*.

Sea entonces $X ~ f, thick f : MM -> RR_+$ un elemento aleatorio distribuido según $f$ sobre una variedad de Riemann compacta y sin frontera --- potencialmente desconocida --- #MM. Sea además $g(t) : RR_+ -> RR$ una función _monótonicamente decreciente_ en su parámetro. Consideraremos el _costo_$J_f$  de un camino $gamma : [0, 1] -> MM, gamma(0)=p, gamma(1)=q$ entre $p, q$ como la integral de $g compose f$ a lo largo de $gamma$:

$
  op(J_(g compose f))(gamma) = integral_0^1 op(g) lr(( f(gamma(t)) ), size: #140%) norm(gamma'(t))_p dif t
$

Y la distancia basada en la densidad $f$ pesada por $g$ entre dos puntos cualesquiera $p, q in MM$ como

$
  D_(g compose f) (p, q) = inf_gamma op(J_(g compose f))(gamma),
$
donde la minimización es con respecto a todos los senderos rectificables con extremos en $p, q$, y $norm(dot)_p$ es la $p-$norma o distancia de Minkowski con parámetro $p$.


#defn([norma $p$])[
  Sea $p >= 1$. Para $x, y in RR^d$, la norma $ell_p$ #footnote[También conocida como "$p-$norma" o "distancia de Minkowski"] se define como:

  $
    norm(x)_p = (sum_(i=1)^d abs(x_i)^p)^(1/p)
  $
]
#obs[La longitud de @longitud es equivalente a tomar $g(t) = 1$ y $p=2$, salvo que $g(t) = 1$ no es monontónicamente decreciente, y por ende la definición previa es insensible a la densidad.]
#obs[Cada $p-$norma induce su propia distancia $d_p$. Algunas son muy conocidas:
  - $p=1$ da la distancia "taxi" o "de Manhattan" #footnote[Llamada así porque representa la distancia que recorrería un taxi en una grilla urbana. Una traducción localizada razonable sería "distancia de San Telmo"]:
  $ d_1(x, y) = norm(x - y)_1 = sum_(i=1)^d abs(x_i - y_i) thin , $
  - $p=2$ da la distancia euclídea que ya hemos usado, omitiendo el subíndice $2$:
  $ d_2(x, y) = norm(x - y) = sqrt(sum_(i=1)^d (x_i-y_i)^2) thin , $
  - $p -> oo$ da la distancia de Chebyshev:
  $ norm(x)_(p->oo) = max_(1 <= i <= d) |x_i - y_i| $
] <lp-metric>

¿Es posible estimar $D_(g compose f)$ de manera consistente? Intuitivamente, consideremos dos puntos $a, b in U subset MM, thick dim MM = d$ #footnote[reemplazamos la notación habitual de $p, q in MM$ por $a, b in MM$ y $d_MM$ por $d$ como en @bijralSemisupervisedLearningDensity2012[§3], y así evitar confusiones con la $p-$norma.] en un vecindario $U$ de $a$ lo "suficientemente pequeño" como para que $f$ sea esencialmente uniforme en él, y en particular en el segmento $gamma_(a b) = overline(a b)$ y tomemos $g = 1 slash f^r$:

$J_(r)(gamma_(a b)) = D_r (a, b) & approx g lr((f("alrededor de " a " y " b)), size: #140%) norm(b - a)_p \
& prop g(norm(b -a)_p^(-d)) norm(b-a)_p \
& = norm(b -a)_p^(r d + 1) = norm(b-a)_p^q thin,$

donde $q = r times d+1$.

Nótese que como ya mencionamos, tomar $q=1$ (o $r = 0$) devuelve la distancia de Minkowski.

Sea $Pi = (pi_0, pi_1, dots, pi_k)$ una serie de índices identificando $k + 1$ observaciones de $XX$. Luego, el costo de un paseo de $k$ pasos por el grafo completo de #XX, $x_(pi_0)-> x_(pi_1) -> dots -> x_(pi_k)$ se puede computar con una simple suma:
$
  J_r (x_(pi_0)-> dots -> x_(pi_k)) & = sum_(j=1)^k D_r (x_(pi_(j-1)), x_(pi_(j))) \
                                    & approx prop sum_(j=1)^k norm(x_(pi_(j)) - x_(pi_(j-1)))_p^q
$

Finalmente, si #XX es una muestra "duficientemente densa", podemos estimar las distancias geodésicas $D_r$ como los "caminos mínimos" en el grafo completo de $XX$ con aristas pesadas por $norm(b - a)_p^q), thick a, b^T in XX$.

Esta estimación es particularmente atractiva, en tanto no depende para nada de la dimensión ambiente $D$, y solo depende de la dimensión intrínseca $d$ de #MM a través de $q=r d+1$. De hecho, los autores mencionan que "casi cualquier par de valores $(p, q)$ funciona", y en particular encuentran que en sus experimento, $p=2, q=8$ "anda bien en general" @bijralSemisupervisedLearningDensity2012[5.1] #footnote[tendremos más para decir al respecto en la @resultados, "Resultados"].

#v(1em)
#align(center)[$quad * quad * quad *$]
#v(1em)

Un resultado interesante por lo exacto, aparece en @chuExactComputationManifold2019. Dado un conjunto de puntos $P = {p_1, dots, p_N}, p_i in MM forall i in [N]$, Considérese la "métrica de vecino más cercano"

$ r_P(q) = 4 min_(p in P) norm(q - p) thin , $

donde $P subset MM$ es un _subconjunto_ de la variedad #footnote[a nuestros fines, $P = XX$, pero no tiene por qué serlo: el argumento de Chu et al admite cualquier conjunto _finito_ $P$, cuyos elementos pueden ser regiones enteras de $MM$, y ni siquiera deben ser convexas] que da lugar a la función de costo

$ J_(r_P) (gamma) = integral_0^1 r_P (gamma(t)) norm(gamma'(t)) dif t thin , $

que a su vez define la distancia

$
  D_(r_P) = inf_gamma J_(r_P) (gamma) thin ,
$
que los autores llaman "distancia de vecino más cercano", $d_bu(N) = D_(r_P)$.

Considérese además la distancia de arista-al-cuadrado #footnote[cuando $P = XX$, ésta es la misma que @vincentDensitySensitiveMetrics2003 propusieran dieciséis años antes]:
$
  d_bu(2)(a, b) = inf_((p_0, dots, p_k)) sum_(i=1)^k norm(p_i - p_(i-1))^2
$
donde el ínfimo se toma sobre toda posible secuencia de puntos $p_0, dots, p_k in P, p_0 = a, p_k = b$. Resulta entonces que la distancia de vecino más cercano $d_bu(N)$ y la métrica de arista cuadrada $d_bu(2)$ son equivalentes para todo conjunto de puntos $P$ en dimensión arbitraria. @chuExactComputationManifold2019[Teorema 1.1] #footnote[De hecho, la prueba que ofrecen es un poco más general: los elementos de $P$ no tienen por qué ser puntos en #MM, sino que pueden ser conjuntos compactos, con costo cero al atravesarlos, cf. @chuExactComputationManifold2019[Figura 2]].

Probar la equivalencia para el caso trivial con $P = {a, b} subset RR^D$ se convierte en un ejercicio de análisis muy sencillo, que cementa la intuición y explica el factor de $4$ en $r_P$:

#v(-3em)
#figure(
  image("img/equivalencia-d2-dN.svg"),
  caption: flex-caption(
    [Ejemplo trivial de la equivalencia $d_bu(N) equiv d_bu(2)$ para $P = {a, b}$],
    [Ejemplo de la equivalencia $d_bu(N) equiv d_bu(2)$],
  ),
) <equiv-d2-dn>

Por la desigualdad triangular, cualquier camino entre $a$ y $b$ ha de ser al menos tan largo como $overline(a b)$. En la mitad del segmento $overline(a b)$ más cercana a $a$ (región azul), $d_bu(N)$ es $norm(z - a)^2$; análogamente, en la región naranja $d_bu(N) = norm(z - b)^2$. Parametricemos $overline(a b):$
$
  gamma(t) &: [0, 1] -> RR^D, quad
  gamma(t) &= a + (b - a) t, quad
  gamma'(t) &= b - a
$
$
  d_bu(N)(a, b) & =  D_(r_P) = inf_gamma J_(r_P) (gamma) = J_(r_P) (overline(a b)) \= integral_0^1 r_{a, b} (gamma(t)) times norm(gamma'(t)) dif t \
  & = integral_0^1 4 min_(p in {a, b}) norm((a + (b -a)t) - p) norm(b-a) dif t \
  & = 4 norm(b-a) (integral_0^(1/2) norm(a + (b -a)t - a) dif t + integral_(1/2)^1 norm(a + (b -a)t - b) dif t )\
  & = 4 norm(b-a) (integral_0^(1/2) norm((b -a)t) dif t + integral_(1/2)^1 norm((a-b)(1-t)) dif t )\
  & = 4 norm(b-a)^2 (integral_0^(1/2) t dif t + integral_(1/2)^1 (1-t) dif t ) \
  &= 4 norm(b-a)^2 [( t^2 |^(1 slash 2)_0) + (t - t^2 |^1_(1 slash 2))] \
  &= 4 norm(b-a)^2 (1/2 - 1/4) \
  & = norm(b-a)^2 \
  & = d_bu(2)(a, b) quad square
$
#v(1em)

El grueso del trabajo de Chu et al consiste en una prueba general de esta igualdad, que se desarrolla en tres partes:
1. Para toda colección finita de puntos $P = {p_i : p_i in RR^D}$,

  1.a. $d_bu(N) <= d_bu(2)$

  1.b. $d_bu(N) >= d_bu(2)$
2. (1) también es válido para toda colección de compactos $P$ de $RR^D$.

Una utilidad de este resultado, es que permite calcular con precisión qué valores de $k$ estimar $d_bu(N)$ sobre el grafo pesado por aristas cuadradas $bu(N N)_k (XX)$  es un "suficientemente buen reemplazo" del cálculo equivalente --- pero mucho más costoso --- sobre $bu(C)(XX)$. En @chuExactComputationManifold2019[Theorema 1.3], observan que con tomar $k = O(2^d ln n)$ basta.

Lo que Chu et al llaman $d_bu(2)$ y figura en @chuExactComputationManifold2019 @vincentDensitySensitiveMetrics2003 como "distancia de arista-cuadrada", es la misma distancia $D_r$ que @bijralSemisupervisedLearningDensity2012 consideran con $p = 2$ (norma euclídea) y $r = 1/d$ --- de modo que $q=r d+1=2$. 

=== Distancia de Fermat

No conocemos pruebas de equivalencia para valores arbitrarios de $p, q$, ni se deriva de la prueba mencionada que deban de existir. Sin embargo, sí existe en la literatura una familia de DBDs  para la cual se conocen tasas de convergencia asintótica de la aproximación muestral en el grafo completo a la distancia propiamente dicha, sobre una variedad Riemanniana compacta sin frontera --- la familia de _Distancia(s) de Fermat_.

El trabajo de @groismanNonhomogeneousEuclideanFirstpassage2019 considera la misma familia de distancias basadas en funciones monótonamente decrecientes de la densidad que @bijralSemisupervisedLearningDensity2012, $g = 1 / f^r$, salvo que en Groisman et al,
$
  p = 2; quad q = alpha; quad r = beta = (alpha - 1) / d
$

y no se limitan a sugerir que la distancia en el espacio ambiente se puede aproximar a través de la distancia basada en el grafo completo con aristas pesadas como en Bijral et al , sino que precisan en qué sentido la una converge a la otra, y a qué tasa.#footnote[Con respecto a fijar $p=2$, en la "Observación 2.6" los autores mencionan que es posible y hasta sería interesante reemplazar la norma euclídea --- $2-$norma --- por otra distancia --- otra $p-$norma, por ejemplo --, reemplazando las integrales con respecto a la longitud de arco, por integrales con respecto a la distancia involucrada. Entendemos de ello que no es una condición _necesaria_ para el desarrolo del trabajo, sino solo _conveniente_. Omitiremos el subíndice en la $2-$norma de aquí en más.]

#defn([Distancia "macroscrópica" de Fermat @groismanNonhomogeneousEuclideanFirstpassage2019[Definición 2.2]])[

  Sea $f$ una función continua y positiva, $beta >=0$
  y $x, y in S subset.eq RR^D$. Definimos la _Distancia de Fermat_ $cal(D)_(f, beta)(x, y)$ como:

  $
    cal(T)_(f, beta)(gamma) = integral_gamma f^(-beta) dif s, quad cal(D)_(f, beta)(x, y) = inf_gamma cal(T)_(f, beta)(gamma) thin ,
  $

donde el ínfimo se toma sobre el conjunto de todos los "senderos" o curvas rectificables entre $x$ e $y$ contenidos en $overline(S)$ --- la clausura de $S$ --, y la integral se entiende con respecto a la longitud de arco $dif s$ dada por la distancia euclídea. Omitiremos la dependencia en $beta$ y $f$ cuando no sea estrictamente necesaria. #footnote[
  En palabras de los autores, el nombre deriva de que "esta definición coincide con el Principio de Fermat en óptica para determinar el sendero recorrido por un haz de luz en un medio no homogéneo cuando el índice de refracción está dado por $f^(-beta)$"
]
]

Este objeto "macroscópico" se puede aproximar a partir de una versión "microscópica" del mismo, que en límite converge a $cal(D)_(f, beta)$:

#let sfd = $D_(Q, alpha)$

#defn([Distancia muestral o "microscópica" de Fermat])[

  Sea $Q$ un conjunto no-vacío, _localmente finito_ #footnote[Es decir, que para todo compacto $U subset RR^D$, la cardinalidad de $Q inter U$ es finita, $abs(Q inter U) < oo$.] de $RR^D$. Para $alpha >=1$ y $x, y in RR^d$, la _Distancia Muestral de Fermat_ se define como


  $
    sfd = inf { & sum_(j=1)^(K-1) ||q_(j+1) - q_j||^alpha : (q_1, dots, q_K) \
                         & "es un camino de x a y", K>=1}
  $

  donde los $q_j in Q thin forall j in [K]$. Nótese que #sfd satisface la desigualdad triangular, define una métrica sobre $Q$ y una pseudo-métrica #footnote[una métrica tal que la distancia puede ser nula entre puntos no-idénticos:  $ exists a != b : d(a, b) = 0 $] sobre $RR^d$.
] <sample-fermat-distance>

Antes de presentar en qué sentido  #sfd converge a $cal(D)_(f, beta)$, una definición más:
#defn([variedad isométrica])[
  Diremos que #MM es una variedad $d-$dimensional $C^1$ _isométrica_ embebida en $RR^D$ si existe un conjunto abierto y conexo $S subset RR^D$ y $phi : S -> RR^D$ una transformación isométrica #footnote[Que preserva las métricas o distancias; del griego "isos" (igual) y "metron" (medida)] tal que $phi(overline(S)) = MM$. Como se mencionó con anterioridad, se espera que $d << D$, pero no es necesario.
]

#defn([Convergencia de $D_(Q, alpha)$, @groismanNonhomogeneousEuclideanFirstpassage2019[Teorema 2.7]])[

  Asuma que #MM es una variedad $C^1$ $d$-dimensional isométrica embebida en $RR^D$ y $f: MM -> R_+$ es una función de densidad de probabilidad continua. Sea $Q_n = {q_1, ..., q_n}$ un conjunto de elementos aleatorios independientes con densidad común $f$. Entonces, para $alpha > 1$ y $x,y in M$ tenemos:

  $ lim_(n->oo) n^beta D_(Q_n,alpha)(x,y) = mu D_(f,beta)(x,y) " casi seguramente." $

  Aquí,
  - $beta = (alpha-1) slash d$,
  - $mu$ es una constante que depende únicamente de $alpha$ y $d$ y
  - la minimización se realiza sobre todas las curvas rectificables $gamma subset MM$ que comienzan en $x$ y terminan en $y$.
]

#obs[
  El factor de escala $beta = (alpha-1)/d$ depende de la dimensión intrínseca $d$ de la variedad, y no de la dimensión $D$ del espacio ambiente.
]

La distancia muestral de Fermat $D_(Q, alpha)$ se puede aproximar a partir de una muestra "lo suficientemente grande _sin conocer ni la variedad #MM ni su dimensión intrínseca_. Además, tiene garantías de convergencia a una distancia basada en densidad (DBD) --- la distancia de Fermat "macroscópica" $cal(D)_(f, beta)$ --- para todo $beta,$. ¡Hemos encontrado la pieza faltante para nuestro clasificador en variedades _desconocidas_! Estamos finalmente en condiciones de proponer un algoritmo de clasificación que reúna todos los cabos del tejido teórico hasta aquí desplegado.

Los trabajos de @littleBalancingGeometryDensity2021 @mckenziePowerWeightedShortest2019 --- contemporáneos a Groisman et al --- consideran lo que ellos llaman "distancias de caminos mínimos pesadas por potencias" #footnote["power-weighted shortest-path distances" o PWSPDs por sus siglas en inglés], aplicándoles no a problemas de clasificación, sino de _clustering_ #footnote[i.e., de identificación de grupos en datos no etiquetados]. Las definiciones de ambos grupos son muy similares en espíritu, con una diferencia menor: la distancia microscópica que plantean Little et al no es la suma de las aristas pesadas por $q=alpha$ como en Bijral et al y Groisman et al, sino la raíz $alpha$-ésima de tal suma, en una especia de reversión de la distancia de Minkowski. Siendo la sustancia de ambos trabajos en esencia la misma, pasaremos directamente a la próxima sección --- nuestra propuesta original.

= Propuesta Original

Al comienzo de este sendero teórico nos preguntamos: ¿es posible mejorar un algoritmo de clasificación reemplazando la distancia euclídea por una aprendida de los datos? Habiendo explorado el área en profundidad, entendemos que sí pareciera ser posible, y en particular la distancia muestral de Fermat es un buen candidato de reemplazo.

Para saldar la cuestión, nos propusimos:
1. Implementar un clasificador basado en estimación de densidad por núcleos como el de @kde-variedad @loubesKernelbasedClassifierRiemannian2008, al que llamaremos "KDC".
2. Implementar un estimador de densidad por núcleos basado en la distancia de Fermat, a fines de comparar la _performance_ de KDC con distancia euclídea y con distancia de Fermat.

Nótese que el clasificador de $k-$vecinos más cercanos de @kn-clf (k-NN, @eps-nn), tiene un pariente cercano, $epsilon-upright("NN")$
#defn([clasificador de $epsilon-$vecinos-más-cercanos])[
  Sean $B_epsilon(x)$ una bola normal de radio $epsilon$ centrada en $x$, y $cal(N)_epsilon (x) = XX inter B_epsilon(x)$ el $epsilon-$vecindario de $x$. El clasificador de $epsilon-$vecinos-más-cercanos $epsilon-N N$ le asignará a $x$ la clase más frecuente entre la de sus vecinos $y in cal(N)_epsilon (x)$
] <epsnn-clf>

@eps-nn es esencialmente equivalente a KDC con un núcleo "rectangular", $k(t) = ind(d(x, t) < epsilon) / epsilon$, pero su implementación es considerablemente más sencilla. Para comprender más cabalmente el efecto de la distancia de Fermat en _la tarea de clasificación_, y no solamente en _cierto_ algoritmo de clasificación, nos propusimos también

3. Implementar un clasificador cual @kn-clf, pero con distancia muestral de Fermat en lugar de euclídea.

=== Estimación de distancia _out-of-sample_

Entrenar el clasificador por validación cruzada no presenta inconvenientes: como $XX_"train" subset.eq XX$ y $XX_"test" subset.eq XX$, se sigue que $forall (a, b) in {XX_"train" times XX_"test"} subset.eq {XX times XX}$, y $D_(XX, alpha) (a, b)$ está bien definida. Ahora bien, ¿cómo calculamos la distancia _muestral_ de una _nueva_ observación $x_0$ a los elementos de cada clase?

Para cada una de las $g_i in GG$ clases, definimos el conjunto $ Q_i= {x_0} union {x_j : x_j in XX, g_j = g_i, j in {1, dots, N}} $
y calculamos $D_(Q_i, alpha) (x_0, dot)$.

=== Adaptación a variedades disjuntas, elección de $h$ por clase

El clasificador de Loubes & Pelletier asume que todas las clases están soportadas en la misma variedad #MM. ¿Quién dice que ello vale para las diferentes clases? ¡Nadie! Sin embargo:
1. No hace falta dicho supuesto: en el peor de los casos, podemos asumir que la unión de las clases está soportada en _cierta_ variedad de Riemann que resulta de (¿la clausura de?) la unión de sus soportes individuales.
2. Si las variedades (y las densidades que soportan) difieren, tanto el $alpha_i^*$ como el $h_i^*$ óptimos para los estimadores de densidad individuales no tienen por qué coincidir.
3. Aunque las densidades individuales $f_i$ estén bien estimadas, el clasificador resultante puede ser malo si no diferencia bien en las fronteras. Por simplicidad, decidimos parametrizar el clasificador con dos únicos hiperparámetros globales: $alpha, h$.

// TODO: integrar referencia @hallBandwidthChoiceNonparametric2005 sobre h óptimo para clasificación con KDEs
== Evaluación

Nos interesa conocer en qué circunstancias, si es que hay alguna, la distancia muestral de Fermat provee ventajas a la hora de clasificar por sobre la distancia euclídea. Además, en caso de existir, quisiéramos en la medida de lo posible comprender por qué (o por qué no) es que tal ventaja existe.
A nuestro entender resulta imposible hacer declaraciones demasiado generales al respecto de la capacidad del clasificador: la cantidad de _datasets_ posibles, junto con sus _configuraciones de evaluación_, es tan densamente infinita como lo permita la imaginación del evaluador. Con un ánimo exploratorio, nos proponemos evaluar la _performance_ de nuestros clasificadores basados en distancia muestral de Fermat en algunas _tareas_ puntuales.

=== Métricas de _performance_

En tareas de clasificación, la métrica más habitual es la _exactitud_ #footnote([Más conocida por su nombre en inglés, _accuracy_.])


#defn(
  "exactitud",
)[Sean $(XX, bu(g)) in RR^(n times p) times RR^n$ una matriz de $n$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(g)) = hat(G)(XX)$ las predicciones de clase resultado de una regla de clasificación $hat(G)$. La _exactitud_ ($"exac"$) de $hat(G)$ en #XX se define como la proporción de coincidencias con las clases verdaderas $bu(g)$:
  $ op("exac")(hat(G) | XX) = n^(-1) sum_(i=1)^n ind(hat(g)_i = g_i) $
] <exactitud>

La exactitud está bien definida para cualquier clasificador que provea una regla _dura_ de clasificación. Ahora bien, cuando un clasificador provee una regla suave, la exactitud como métrica "pierde información": dos clasificadores binarios que asignen respectivamente 0.51 y 1.0 de probabilidad de pertenecer a la clase correcta a todas las observaciones tendrán la misma exactitud, $100%$, aunque el segundo es a las claras mejor. A la inversa, cuando un clasificador erra al asignar la clase: ¿lo hace con absoluta confianza, asignando una alta probabilidad a la clase equivocada, o con cierta incertidumbre, repartiendo la masa de probabilidad entre varias clases que considera factibles?

Una métrica natural para evaluar una regla de clasificación suave es la _verosimilitud_ (y su logaritmo) de las predicciones.

#defn(
  "verosimilitud",
)[Sean $bu(("X, y")) in RR^(n times p) times RR^n$ una matriz de $n$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(Y)) = clf(XX) in RR^(n times k)$ la matriz de probabilidades de clase resultado de una regla suave de clasificación #clf. La _verosimilitud_ ($"vero"$) de #clf en #bu("X") se define como la probabilidad conjunta que asigna #clf a las clases verdaderas #bu("y"):
  $
    op(L)(clf) = op("vero")(
      clf | XX
    ) = Pr(hat(bu(y)) = bu(y)) = product_(i=1)^n Pr(hat(y)_i =y_i) = product_(i=1)^n hat(bu(Y))_((i, y_i))
  $

  Por conveniencia, se suele considerar la _log-verosimilitud promedio_,
  $ op(cal(l))(clf) = n^(-1) log(op("L")(clf)) = n^(-1)sum_(i=1)^n log(hat(bu(Y))_((i, y_i))) $
] <vero>

La verosimilitud de una muestra varía en $[0, 1]$ y su log-verosimilitud, en $(-oo, 0]$, pero, como métrica, esta solo se vuelve comprensible _relativa a otros clasificadores_. Una forma de "normalizar" la log-verosimilitud se debe a @mcfaddenConditionalLogitAnalysis1974.

#defn(
  [$R^2$ de McFadden],
)[Sea $clf_0$ el clasificador "nulo", que asigna a cada observación y posible clase, la frecuencia empírica de clase encontrada en la muestra de entrenamiento $XX_("train")$. Para todo clasificador suave $clf$, definimos el $R^2$ de McFadden como
  $ op(R^2)(clf | XX) = 1 - (op(cal(l))(clf)) / (op(cal(l))(clf_0)) $
] <R2-mcf>

#obs[ $op(R^2)(clf_0) = 0$. A su vez, para un clasificador perfecto $clf^star$ que otorgue toda la masa de probabilidad a la clase correcta, tendrá $op(L)(clf^star) = 1$ y log-verosimilitud igual a 0, de manera que $op(R^2)(clf^star) = 1 - 0 = 1$.

  Sin embargo, un clasificador _peor_ que $clf_0$ en tanto asigne bajas probabilidades a las clases correctas, puede tener un $R^2$ infinitamente negativo.
]

Visto y considerando que tanto #fkdc como #fkn son clasificadores suaves, evaluaremos su comportamiento en comparación con ambas métricas, la exactitud y el $R^2$ de McFadden #footnote[de aquí en más, $R^2$ para abreviar].

=== Algoritmos de referencia

Además de medir qué (des)ventajas otorga el uso de una distancia aprendida de los datos en la tarea de clasificación, quisiéramos entender (a) por qué sucede, y (b) si tal (des)ventaja es significativa en el amplio abanico de algoritmos disponibles. Pírrica victoria sería mejorar con la distancia de Fermat la _performance_ de cierto algoritmo, para encontrar que aún con la mejora, el algoritmo no es competitivo en la tarea de referencia.

Consideraremos a modo de referencia los siguientes algoritmos:
- Naive Bayes Gaussiano (#gnb),
- Regresión Logistica (#logr) y
- Clasificador de Soporte Vectorial (#svc)

Esta elección no pretende ser exhaustiva, sino que responde a un "capricho informado" del investigador. #gnb es una elección natural, ya que es la simplificación que surge de asumir independencia en las dimensiones de $X$ para KDE multivariado (@kde-mv), y se puede computar para grandes conjuntos de datos en muy poco tiempo. #logr es "el" método para clasificación binaria, y su extensión a múltiples clases no es particularmente compleja: para que sea mínimamente valioso un nuevo algoritmo, necesita ser al menos tan bueno como #logr, que tiene ya más de 65 años en el campo @blissCALCULATIONDOSAGEMORTALITYCURVE1935, @coxRegressionAnalysisBinary1958. Por último, fue nuestro deseo incorporar algunos métodos más cercanos al estado del arte: un método de _boosting_ (@GradientBoosting2025) y el antedicho clasificador de soporte vectorial, #svc #footnote[en dos variantes: con núcleos (_kernels_) lineales y RBF - #emph[radial basis functions]]. Por conocerlo en profundidad y en virtud de su sencillez de uso, la implementación se realizó utilizando `scikit-learn` @JMLR:v12:pedregosa11a, un poderoso y extensible paquete para tareas de aprendizaje automático en Python.

=== Metodología

La unidad de evaluación de los algoritmos a considerar es una `Tarea`, que se compone de:
- un _diccionario de algoritmos_ a evaluar en condiciones idénticas, definidas por
- un _dataset_ con el conjunto de $N$ observaciones en $D$ dimensiones repartidas en $K$ clases, $(XX, bu(g))$,
- un _split de evaluación_ $r in (0, 1)$, que determina las proporciones de los datos a usar durante el entrenamiento ($1 - r$) y la evaluación ($r$), junto con
- una _semilla_ $s in [2^32]$ que alimenta el generador de números aleatorios y define determinísticamente cómo realizar la división antedicha.

=== Entrenamiento de los algoritmos
La especificación completa de un clasificador requiere, además de la elección del algoritmo, la especificación de sus _hiperparámetros_, de manera tal de optimizar su rendimiento bajo ciertas condiciones de evaluación. Para ello, se definió de antemano para cada clasificador una _grilla_ de hiperparámetros: durante el proceso de entrenamiento, la elección de los "mejores" hiperparámetros se efectuó maximizando la log-verosimilitud @vero para los clasificadores suaves, y la exactitud @exactitud para los duros #footnote[Entre los mencionados, el único clasificador duro es #svc. Técnicamente es posible entrenar un clasificador suave a partir de uno duro con un _segundo_ estimador que toma como _input_ el resultado "crudo" del clasificador duro y da como _output_ una probabilidad calibrada (cf. #link("https://scikit-learn.org/stable/modules/calibration.html")[Calibración] en la documentación de `scikit-learn`  @buitinckAPIDesignMachine2013), pero es un proceso computacionalmente costoso.] con una búsqueda exhaustiva por validación cruzada de 5 pliegos #footnote[Conocida en inglés como #emph[Grid Search 5-fold Cross-Validation]] sobre la grilla entera.

=== Estimación de la variabilidad en la _performance_ reportada
En última instancia, cualquier métrica evaluada no es otra cosa que un _estadístico_ que representa la "calidad" del clasificador en la tarea a mano. A fines de conocer no solo su estimación puntual sino también darnos una idea de la variabilidad de su _performance_, para cada dataset y colección de algoritmos, se entrenaron y evaluaron #reps tareas idénticas salvo por la semilla $s$, que luego se usaron para estimar la varianza y el desvío estándar en la exactitud (@exactitud) y el pseudo-$R^2$ (@R2-mcf).

Cuando el conjunto de datos proviene del mundo real y por lo tanto _preexiste a nuestro trabajo_, las #reps semillas $s_1, dots, s_#reps$ fueron utilizadas para definir el split de entrenamiento/evaluación. Por el contrario, cuando el conjunto de datos fue generado sintéticamente, las semillas se utilizaron para generar #reps versiones distintas pero perfectamente replicables del dataset, y en todas se utilizó una misma semilla maestra $s^star$ para definir el split de evaluación.


=== Regla de Parsimonia

La estrategia de validación cruzada intenta evitar que los algoritmos sobreajusten durante el entrenamiento, evaluando su comportamiento en $XX_"test"$, disjunto de $XX_"train"$.
No todas las parametrizaciones son equivalentes: en general, para cada hiperparámetro se puede establecer una dirección en la que el modelo se complejiza, en tanto se ajusta más y más a los datos de entrenamiento: un estimador #kn entrenado con _menos vecinos_  cambia sus predicciones más seguido que uno con _más vecinos_ - considere $50-"NN"$ y $1-"NN"$.

#obs(link("https://es.wikipedia.org/wiki/Navaja_de_Ockham")[Navaja de Occam])[
  "cuando dos teorías en igualdad de condiciones tienen las mismas consecuencias, la teoría más simple tiene más probabilidades de ser correcta que la compleja"
]
Reformulando, diremos que sujeto a la implementación de _cierto_ algoritmo, cuando dos hiperparametrizaciones $h_0, h_1$ tienen _casi_ las mismas consecuencias --- alcanzan $R^2$ tales que $abs(R^2(h_0) - R^2(h_1)) <= epsilon$ --- preferiremos la más sencilla: la de menor _complejidad_ $C(h)$, para cierta función $C$ a definir.

La validación cruzada de $k$ pliegos nos provee naturalmente de $k$ realizaciones de la métrica a optimizar para cada hiperparametrización. Sea $h^"opt"$ la que minimiza la pérdida de evaluación y $hat(s^2)(L(h^"opt"))$ la varianza estimada de dicha pérdida. Sobre esta base, implementamos la siguiente regla:
#defn([regla de $1 sigma$])[
  Sea $hat(s^2)(L(h))$ una estimación razonable de la varianza de la pérdida $L(h)$ del modelo parametrizado en $h$, y $h^"opt"$ la que alcanza la mínima pérdida. De entre todas las hiperparametrizaciones, elíjase _la más sencilla_: $h^star = arg min_(h in cal(H)) C(h), \ cal(H) = {h : L(h) <= L(h^"opt") + sqrt(hat(s^2)(L(h^"opt"))) }$.
] <r1sd>

Para definir $C$ en modelos con $dim(h) > 1$, definimos el orden de complejidad creciente _para cada clasificador_ como una lista ordenada de 2-tuplas con el nombre de cada hiperparámetro y una dirección de crecimiento. Para #fkdc, por ejemplo, $C(h) = [(alpha, "ascendente"), (h, "descendente")]$. La decisión de ordenar así los parámetros, con $alpha$ primero y $C$ ascendente en $alpha$, hace que la evaluación "prefiera" naturalmente a #kdc por sobre #fkdc#footnote[$#kdc = op(#fkdc)(alpha = 1)$], ya que el mínimo $alpha = 1$ estudiado resulta siempre preferido. En consecuencia, cuando veamos que #fkdc elige un $alpha != 1$, sabremos que no es por pura casualidad.

// TODO: considerar renombrar hiperparametrización para evitar confusión con la ventana $h$.
#obs([complejidad en $h$])[
  La complejidad es _descendente_ en el tamaño de la ventana $h$: a mayor $h$, tanto más grande se vuelve el vecindario donde $K_h (d(x, x_i)) >> 0$ y por ende pesa en la asignación. Análogamente, $k-"NN"$ y su primo $epsilon- "NN"$ tienen complejidad descendente en $k, epsilon$.
]

=== Medidas de locación y dispersión no-paramétricas
Siendo el _setting_ --- estimación de densidad basada en distancias en una variedad de Riemann desconocida --- tan poco ortodoxo, parece razonable comparar la _performance_ con medidas de locación robustas. Por eso comparamos la _performance_ mediana (y no media) por semilla de cada clasificador, y la visualizamos con un _boxplot_ en lugar de un intervalo de confianza $mu plus.minus n times sigma$.
= Resultados <resultados>

== In Totis

En total, ejecutamos unas 4,500 tareas, producto de #reps repeticiones por dataset y clasificador, sobre un total de 20 datasets y 9 clasificadores diferentes. Recordemos que todos los estimadores se entrenaron con _score_ `neg_log_loss` (para optimizar por $R^2$), salvo #svc, que al ser un clasificador duro se entrenó con `accuracy`. Así, entre los clasificadores blandos la distancia de Fermat rindió frutos, con el máximo $R^2$ mediano en 10 de los 20 experimentos: 7 preseas fueron para #fkdc y 3 para #fkn.

#gbt "ganó" en 5 datasets, entre ellos en varios con mucho ruido (`_hi` y `_12`). #kdc resultó óptimo en 2 datasets, consolidando la técnica del @kde-variedad como competitiva de por sí. Por último, tanto #kn como #logr (en su versión escalada, #slr) resultaron medianamente mejores que todos los demás en ciertos datasets, y solo #gnb no consiguió ningún podio - aunque resultó competitivo en casi todo el tablero.
La amplia distribución de algoritmos óptimos según las condiciones del dataset pone de relieve la existencia de ventajas relativas en todos ellos.

#let data = csv("data/mejor-clf-por-dataset-segun-r2-mediano.csv")

#let headers = data.at(0)
#let rows = data.slice(1, count: data.len() - 1)
#table(columns: headers.len(), table.header(..headers), ..rows.flatten())

El mismo análisis con métrica de exactitud es, desde luego, menos favorable a nuestros métodos entrenados para otra cosa. #svc, entrenado a tono, resulta un algoritmo casi imbatible, con sólidos números en todo tipo de datasets y máximos en 6 datasets. #gbt vuelve a brillar en datasets con mucho ruido y siguen figurando como competitivos un amplio abanico de estimadores: hasta #fkdc retiene su título en 1 dataset, `espirales_lo`.

#let data = csv("data/mejor-clf-por-dataset-segun-accuracy-mediano.csv")
#let headers = data.at(0)
#let rows = data.slice(1, count: data.len() - 1)
#table(columns: headers.len(), table.header(..headers), ..rows.flatten())


Solo considerar la _performance_ de #fkdc y #fkn en los 20 datasets daría unas 40 unidades de análisis, y en el espíritu de indagación curiosa que guía esta tesis, existen aún más tendencias y patrones interesantes en los 4,500 experimentos realizados. No es nuestra intención matar de aburrimiento al lector, con lo cual a continuación haremos un paneo arbitrario por algunos de los resultados que (a) nos resultaron más llamativos o (b) se acercan lo suficiente a alguno de la literatura previa como para merecer un comentario aparte. Quien desee corroborar que no hice un uso injustificado de la discrecionalidad para elegir resultados, puede referirse al @apendice-a[Apéndice A2 - Hojas de resultados por experimento] y darse una panzada de tablas y gráficos.
== Lunas, círculos y espirales ($D=2, d=1, k=2$)

Para comenzar, consideramos el caso no trivial más sencillo con $D>d$: $D=2, d=1, k=2$, y exploramos tres curvas sampleadas con un poco de "ruido blanco"/* TODO: citar paper sobre sampleo en el tubo de radio $r$ alrededor de la variedad #MM. */:
#let plotting_seed = 1075
#figure(
  columns(3)[
    #image("img/lunas_lo-scatter.svg")
    #colbreak()
    #image("img/circulos_lo-scatter.svg")
    #colbreak()
    #image("img/espirales_lo-scatter.svg")
  ],
  caption: flex-caption["Lunas", "Círculos" y "Espirales", con $d_x = 2, d_(MM) = 1$ y $s=#plotting_seed$][ "Lunas", "Círculos" y "Espirales" ],
) <fig-2>

#defn("ruido blanco")[Sea $X = (X_1, dots, X_d) in RR^d$ una variable aleatoria tal que $"E"(X_i)=0, "Var"(X_i)=SS thick forall i in [d]$. Llamaremos "ruido blanco con escala $SS$" a toda realización de $X$.] <ruido-blanco>.



En una primera variación con "bajo ruido" (y sufijada "`_lo`") #footnote[en inglés, _low_ y _high_ - baja y alta - son casi homófonos de _lo_ y _hi_], a las observaciones #XX sobre la variedad #MM/* TODO: agregar nota al pie sobre cómo se generaron los datasets (sampleo uniforme en espacio euclídeo homeomorfo, carta exponencial + ruido blanco). Cf. Apéndice Datasets. */ se les añadió ruido blanco de una normal estándar bivariada escalada por un parámetro de ruido $sigma$, $epsilon ~ cal(N)_2(0, sigma^2 bu(I))$ ajustado a cada dataset para resultar "poco" relativo a la escala de los datos.
$ sigma_"lunas" = 0.25 quad sigma_"circulos" = 0.08 quad sigma_"espirales" = 0.1 $.

En los tres datasets, el resultado es muy similar: #fkdc es el estimador que mejor $R^2$ reporta, y en todos tiene una exactitud comparable a la del mejor para el dataset. En ninguno de los tres datasets #fkdc tiene una exactitud muy distinta a la de #kdc, pero saca ventaja en $R^2$ para `lunas_lo` y `espirales_lo`.

Entre el resto de los algoritmos, los no paramétricos son competitivos: #kn, #fkn y #gbt, mientras que #gnb, #slr, #logr rinden mal pues las _fronteras de decisión_ que pueden representar no cortan bien a los datos.


#let highlights_figure(dataset) = {
  let highlights = json("data/" + dataset + "-r2-highlights.json")
  let csv_string = highlights.at("summary")
  let lines = csv_string.split("\n")
  let data = ()
  for line in lines {
    let fields = line.split(",")
    data.push(fields)
  }
  let headers = data.at(0)
  let rows = data.slice(1, count: data.len() - 2)
  // TODO: pintar de color fermat, negrita best acc, grisar mal R2
  let tabla_resumen = table(columns: headers.len(), stroke: 0.5pt, table.header(..headers), ..rows.flatten())


  figure(
    table(
      columns: 2,
      rows: 2,
      stroke: 0pt,
      image("img/" + dataset + "-scatter.svg"), text(size: 8pt)[#tabla_resumen],
      image("img/" + dataset + "-r2-boxplot.svg"), image("img/" + dataset + "-accuracy-boxplot.svg"),
    ),
    caption: flex-caption[Resumen para #dataset][],
  )
}


#highlights_figure("lunas_lo")
#pagebreak()

#highlights_figure("circulos_lo")
#pagebreak()

#highlights_figure("espirales_lo")
#pagebreak()

#let euc = $norm(thin dot thin)_2$
#let sfd = $D_(Q, alpha)$

#obs("riesgos computacionales")[
  Una dificultad de entrenar un clasificador _original_ es que hay que definir las rutinas numéricas "a mano" #footnote[Usando librerías estándares como `numpy` y `scipy`, sí, pero nada más.], y _debugear_ errores en rutinas numéricas es particularmente difícil, porque las operaciones casi siempre retornan, salvo que retornan valores irrisorios #footnote[Hubo montones de estos, cuya resolución progresiva dio lugar a la pequeña librería que acompaña esta tesis. Todo error de cálculo que pueda persistir en el producto final depende exclusivamente de mí, pero tan mal no parecen haber dado los experimentos.].

  A ello se le suma que el cómputo de #sfd es realmente caro. /* TODO: precisar orden $O$. */ Aun siguiendo "buenas prácticas computacionales" #footnote[Como sumar logaritmos en lugar de multiplicar valores "crudos" siempre que sea posible], implementaciones ingenuas pueden resultar impracticables hasta en datasets de pequeño $n$.

  Por otra parte, es cierto que cuando $alpha = 1$ y $n->oo, quad sfd -> cal(D)_(f, beta) = euc$, pero esa es una afirmación asintótica y aquí estamos tomando $k=5$ pliegos de entre $n = 800$ observaciones, con $n_"train" = n_"eval" = n slash 2$ observaciones para un tamaño muestral efectivo de $(k-1)/k n/2 = 360$. ¿Es 360 un tamaño muestral "lo suficientemente grande" para que sea válida?

  Por todo ello, que la bondad de los clasificadores _no empeore_ con el uso de #sfd en lugar de #euc es de por sí un hito importante.
]

#pagebreak()
==== Fronteras de decisión
Una inspección ocular a las fronteras de decisión revela las limitaciones de distintos algoritmos.

#logr y #slr solo pueden dibujar fronteras "lineales", y como ninguna frontera lineal que corte la muestra logra dividirla en dos regiones con densidades de clase realmente diferentes, el algoritmo falla. #gnb falla de manera análoga, aunque su problema es otro - no lidia bien con distribuciones con densidades marginales muy similares.

#let clfs = ("kdc", "fkdc", "svc", "kn", "fkn", "gbt", "slr", "lr", "gnb")
#align(center)[#box(width: 160%, figure(table(columns: 3, stroke: 0pt, ..clfs.map(clf => image(
    "img/espirales_lo-" + clf + "-decision_boundary.svg",
  )))))]

Aun con esas limitaciones, #logr tiene un rendimiento decente en `lunas_lo`:

#figure(
  image("img/lunas_lo-lr-decision_boundary.svg"),
  caption: [Frontera de decisión para #slr en `lunas_lo`, semilla #plotting_seed],
)
Nótese que la frontera _lineal_ entre clases (al centro de la banda gris) aprendida por #logr separa _bastante_ bien la muestra: pasa por el punto del segmento que une los "focos" de cada luna, y de todas las direcciones con origen allí, es la que mejor separa las clases. _Grosso modo_, en el tercio de la muestra más cercano a la frontera, alcanza una exactitud de $~50%$, pero en los tercios al interior de cada región está virtualmente en 100%, que da un promedio global de $1/3 50% + 2/3 100% = 86.7%$, casi exactamente la exactitud observada.

También resulta llamativa la "creatividad" de #gbt para aproximar unas fronteras naturalmente curvas como una serie de preguntas binarias, que solo permiten dibujar regiones rectangulares #footnote[Quien haya pasado alguna clase no particularmente emocionante pintando espirales en hoja cuadriculada reconocerá este patrón rápidamente.].

Entre #kn y #fkn casi no observamos diferencias, asunto en el que ahondaremos más adelante. Por lo pronto, sí se nota que se adaptan bastante bien a los datos, con algunas regiones "claras" de incertidumbre que resultan onerosas en términos de $R^2$: a primera vista los mapas de decisión recién expuestos se ven muy similares, pero las pequeñas diferencias de probabilidades resultaron en una diferencia de $0.19$ en $R^2$ _en contra_ del modelo más complejo para esta semilla.

#kdc ofrece una frontera aún más regular que #kn, sin perder en $R^2$ y hasta mejorando la exactitud. Y por encima de esta ya destacable _performance_, el uso de la distancia de Fermat _incrementa_ la confianza en estas regiones -nótese cómo se afinan las áreas grises y aumenta la superficie de rojo/azul sólido, mejorando otro poco el $R^2$.



#figure(columns(2)[
  #image("img/espirales_lo-fkdc-decision_boundary.svg")
  #colbreak()
  #image("img/espirales_lo-svc-decision_boundary.svg")
])

Por último, observamos las fronteras de #svc, que no tienen gradiente de color sino solo una frontera lineal #footnote[Como aprendimos: la frontera de una variedad riemanniana de dimensión intrínseca $d$ es una variedad sin frontera de dimensión intrínseca $d-1$; la frontera de estas regiones en $R^2$ es una curva parametrizable en $RR^1$ embebida en $RR^2$]. Es sorprendente la flexibilidad del algoritmo, que consigue dibujar una única frontera sumamente no-lineal que separa los datos con altísima exactitud. La ventaja que #fkdc pareciera tener sobre #svc aquí, es que la frontera que dibuja pasa "más lejos" de las observaciones de clase, mientras que la #svc parece estar muy pegada a los brazos de la espiral, particularmente en el giro más interno.

=== Estudio de ablación #footnote[Según la #link("https://dle.rae.es/ablaci%C3%B3n")[RAE], "Del lat. tardío ablatio, -ōnis 'acción de quitar'."; ¿qué se pierde en términos de $R^2$ al _no_ usar #sfd en estos algoritmos?]: $R^2$ para #kdc/ #kn con y sin distancia de Fermat.

Sirvan como panorama para concentrar la atención en esta diferencia, los gráficos de dispersión del $R^2$ alcanzado en $XX_"test"$ para #kn y #kdc con y sin distancia de Fermat, en las #reps repeticiones de cada Tarea.

#let curvas = ("lunas", "circulos", "espirales")
#figure(
  columns(2)[
    #for c in curvas {
      image("img/" + c + "_lo-kdc-fkdc-r2-scatter.svg")
    }
    #colbreak()
    #for c in curvas {
      image("img/" + c + "_lo-kn-fkn-r2-scatter.svg")
    }
  ],
  caption: [Gráficos de dispersión (_scatterplots_) de $R^2$ para #kdc (izq.) y #kn (der.) con (eje $y$) y sin (eje $x$) distancia de Fermat.],
) <fig-17>

Para #kn y #fkn, los resultados son casi exactamente iguales para todas las semillas; con ciertas semillas saca ventaja #fkn en `espirales_lo`, pero también tiene dos muy malos resultados con $R^2 approx 0$ que #kn evita.

Para #fkdc, pareciera evidenciarse alguna ventaja para varias semillas en `lunas_lo, espirales_lo`, menos así para `circulos_lo`.

Veamos primero qué sucede durante el entrenamiento para `circulos_lo`: ¿es que no hay ninguna ventaja en usar #sfd? Consideremos la _superficie de pérdida_ que resulta de graficar en 2D la pérdida $L$ usada _durante el entrenamiento_ para cada hiperparametrización considerada:

#obs(
  "unidades de la pérdida",
)[Si bien consideramos como _score_ (a más, mejor) $R^2$, el entrenamiento se realizó con `neg_log_loss`, que aunque tiene la misma monotonicidad que $R^2$, está en otras unidades: entre $(-oo, 0]$.]

#figure(
  image("img/circulos_lo-8527-fkdc-bandwidth-alpha-loss_contour.svg"),
  caption: [Superficie de _score_: para cada valor de $alpha$ considerado, una cruz roja marca el valor de $h$ que maximizó el _score_.],
)
Nótese que la región amarilla, que representa los máximos puntajes durante el entrenamiento, se extiende diagonalmente a través de todos los valores de $alpha$. Es decir, no hay un _par_ de hiperparámetros óptimos $(alpha^star, h^star)$, sino que fijando $alpha$, siempre pareciera existir un(os) $h^star (alpha)$ que alcanza (o aproxima) la máxima exactitud _posible_ con el método en el dataset. En este ejemplo en particular, hasta pareciera ser que una relación log-lineal captura bastante bien el fenómeno, $log(h^star) prop alpha$. En particular, entonces, $"exac"(h^star (1), 1) approx "exac"(h^star, alpha^star)$, y se entiende que el algoritmo #fkdc, que agrega el hiperparámetro $alpha$ a #kdc no mejore significativamente su exactitud.

// TODO: agregar referencia al paper que dice que "todo alfa da OK", que tomaba p=2 q=8 (bijral?)
// TODO: aplicar q=8 a ver qué resulta

Ahora bien, esto es solo en _un_ dataset, con _una_ semilla específica. ¿Se replicará el fenómeno en los otros datasets?

// #let mejores_semillas = (7837, 5640, 4286)
// #let peores_semillas = (1075, 1434, 9975)
#let mejores_semillas = (9975, 1434, 7837)
#let peores_semillas = (7354, 8527, 1188) //, 4286)
#let semillas = peores_semillas // + peores_semillas

#let imgs = (curvas.map(c => semillas.map(s => (c, s))).sum()).map(tup => image(
  "img/" + tup.at(0) + "_lo-" + str(tup.at(1)) + "-fkdc-bandwidth-alpha-loss_contour.svg",
))


#align(center)[
  #box(width: 150%)[
    #figure(
      grid(columns: semillas.len(), stroke: 0pt, ..imgs),
      caption: [Superficies de pérdida para tres semillas y cada uno de los tres datasets. El patrón log-lineal previamente observado se replica casi perfectamente en todos los casos.],
    ) <fig-19>
  ]
]

¡Pues sí replica! Podemos observar también en datasets como `(circulos_lo, 7354)`, cómo la regla de parsimonia nos ayuda a elegir, dentro de la gran "meseta color lima" en que todas las hiperparametrizaciones alcanzan resultados similares, para cada $h$ el mínimo $alpha$ que no "cae" hacia la región azul de menores scores.

Estamos ahora frente a una contradicción: en la @fig-17 vimos que por ejemplo, para `lunas_lo`, #fkdc alcanzaba un $R^2$ consistentemente mejor que #kdc; mientras que de los paneles superiores de la @fig-19 observamos que los score que se alcanzan limitados a $alpha = 1$ son tan altos como los de $alpha > 1$. Es cierto que los resultados de @fig-17 son a través de _todas_ las semillas, y en el conjunto de _evaluación_, mientras que en la @fig-19 observamos _algunas semillas_ y sobre los datos de entrenamiento, pero la pregunta es válida: ¿de dónde proviene la ventaja de #fkdc en estos datasets?

==== Hiperparámetros óptimos en `lunas_lo` para #kdc, #fkdc

Hacemos entonces una comprobación fundamental: ¿qué parametrizaciones están siendo elegidas en el esquema de validación cruzada con regla de parsimonia? Hete aquí el detalle para las #reps repeticiones de `lunas_lo`:

#tabla_csv("data/lunas_lo-best_params.csv")

#obs(
  "mejores corridas de _test_",
)[En _test_, a veces el mejor puntaje lo obtienen _otros_ $alpha$, pero la diferencia no es lo suficientemente grande para descartar alguna opción con $alpha = 1$.
  #tabla_csv("data/lunas_lo-best_test_params.csv")
]
Resulta ser que
- al entrenar #fkdc se está eligiendo $alpha=1$ para _todas_ las semillas, y
- el ancho de banda seleccionado es ligera pero consistentemente _menor_ que el que toma #kdc.

Veamos cómo se comparan los valores de $R^2$ que alcanza cada algoritmo en cada semilla:
#figure(
  columns(2)[
    #image("img/lunas_lo-[f]kdc-score-vs-bandwidth.png")
    #colbreak()
    #image("img/lunas_lo-[f]kdc-delta_r2-vs-delta_h.png")],
  caption: [
    (izq.) Dispersión de $R^2$ en función de $h$ por clasificador y semilla en lunas_lo, para #fkdc, #kdc;
    (der.) Dispersión de $Delta_(R^2) = R^2_#kdc - R^2_#fkdc$ en función de $Delta_h = h^star_#fkdc - h^star_#kdc$ para cada semilla.],
)
En el panel izquierdo se observa una clara tendencia a mejorar ligeramente el $R^2$ a medida que disminuye el ancho de la ventana $h$ (en el rango en cuestión). En el panel derecho, para confirmar que la tendencia sucede _en cada repetición del experimento_, comparamos no los valores absolutos sino las diferencias _relativas_ en $R^2, h$ para #fkdc, #kdc, y vemos que a mayor diferencia en $h$, peor es la caída en $R^2$.

Cabe aquí una crítica al diseño experimental: si #fkdc está tomando siempre $alpha =1$, por qué #kdc no puede elegir el mismo $h$ que #fkdc y así "empatar" su performance? ¿Se exploró una grilla de hiperparámetros a propósito desfavorable para #kdc? Pues no, todo lo contrario #footnote[La definición exacta está en `fkdc/config.py`, y es `np.logspace(-5, 6, 45)` para #fkdc y `np.logspace(-5, 6, 136)` para #kdc]: las grillas de $h$ para #kdc y #fkdc
- cubren de manera "logarítmicamente equidistante" el mismo rango de $h: [10^(-5), 10^6]$ y
- la grilla de #kdc cuenta con $approx$ el triple de puntos de #fkdc ($136 "vs." 45$).

Como en el entrenamiento de #fkdc se gastaron 13 veces más recursos evaluando 13 valores distintos de $alpha$ #footnote[$alpha in {1 + 0.25 i, thick i in [13]} subset [1, 4]$], consideramos oportuno permitirle a #kdc explorar más valores de $h$, y la cantidad se eligió para que la grilla de #kdc coincida en lo posible con la de #fkdc, y tenga además otros dos valores "entre medio" de dos valores cualesquiera de la grilla de #fkdc. En efecto, en el rango de interés, las grillas contaban con los valores:
$
  #fkdc: & [0.1, 0.178, 0.316, 0.562] \
   #kdc: & [0.119, 0.143, 0.173, 0.208, 0.251, 0.303, 0.366, 0.441, 0.532] \
$
con lo cual #kdc _podría_ haber encontrado el ligeramente más conveniente $h^star approx 0.17$, pero la convalidación cruzada se inclinó por valores concentrados en el rango $[0.25, 0.3]$. De repetir el experimento tomando una grilla más fina en este rango crucial, es posible que $Delta_h^star approx 0$ y por ende $Delta_(R^2)$ también, aunque por el mismo argumento de tomar una grilla más fina para $alpha approx 1$ terminaríamos tal vez encontrando un $alpha^star > 1$ para #fkdc #footnote[Hete aquí la dificultad de enunciar propiedades generales a partir de experimentos particulares: siempre hay _un experimento más_ para hacer, pero lamentablemente, en algún momento había que culminar la etapa experimental.]. En cualquier caso, hemos de aceptar que la ventaja de #fkdc en `lunas_lo` (y también en `espirales_lo`, aunque de menor magnitud) sobre #kdc _no_ se debe a la inclusión del hiperparámetro $alpha$, sino a una validación cruzada aleatoriamente favorable.

=== Efectos de aumentar el ruido

Consideremos ahora los mismos datasets que hasta ahora, pero sampleando las observaciones sobre la variedad con "más ruido"; i.e., aumentando el valor de $sigma$ en el ruido blanco (@ruido-blanco) que le agregamos a los $X in MM$ según

$ sigma_"lunas" = 0.5 quad sigma_"circulos" = 0.2 quad sigma_"espirales" = 0.2 $.

#figure(
  columns(3)[
    #image("img/lunas_hi-scatter.svg")
    #colbreak()
    #image("img/circulos_hi-scatter.svg")
    #colbreak()
    #image("img/espirales_hi-scatter.svg")
  ],
  caption: flex-caption["Lunas", "Círculos" y "Espirales" con "alto ruido"][ "Lunas", "Círculos" y "Espirales", alto ruido ],
) <fig-22>

En general, #fkdc y #fkn siguen siendo competitivos, pero el "terreno de juego" se ha nivelado considerablemente, y las ventajas antes vistas disminuyen. En particular, en `lunas_hi, circulos_hi` observamos que #gbt alcanza un $R^2$ marginalmente mejor que el #fkdc, y en el segundo también lo supera ligeramente en exactitud. En `espirales_hi` todos los métodos basados en densidad por núcleos (#fkdc, #kdc, #fkn, #kn) alcanzan un $R^2$ muy similar mientras todos los demás quedan largamente atrás (#gbt) o no se distinguen del $0$, pero #svc obtiene la mejor exactitud. Las ventajas de #fkdc por sobre #kdc son casi nulas en este contexto.


#highlights_figure("lunas_hi")
#pagebreak()

#highlights_figure("circulos_hi")
#pagebreak()

#highlights_figure("espirales_hi")



El aumento en la cantidad de ruido hace la tarea más difícil para _todos_ los estimadores, pero los métodos basados en densidad por núcleos parecen sufrirlo particularmente, aunque solo sea porque "caen desde más alto", a un nivel de _performance_ similar al de otros métodos.

#figure(
  columns(3)[
    #image("img/lunas-caida_r2.svg")
    #colbreak()
    #image("img/circulos-caida_r2.svg")
    #colbreak()
    #image("img/espirales-caida_r2.svg")
  ],
  caption: [
    $R^2$ mediano por clasificador y dataset, comparado entre la variante con bajo (`_lo`) y alto (`_hi`) ruido en el sampleo.
  ],
)


Por último, veamos las fronteras de decisión que resultan para nuestro método, #fkdc, y los más competitivos en términos de $R^2$ (#gbt) y exactitud (#svc).
#align(center)[#box(width: 160%)[
  #figure(
    columns(3)[
      #image("img/lunas_hi-fkdc-decision_boundary.svg")
      #image("img/circulos_hi-fkdc-decision_boundary.svg")
      #image("img/espirales_hi-fkdc-decision_boundary.svg")
      #colbreak()
      #image("img/lunas_hi-gbt-decision_boundary.svg")
      #image("img/circulos_hi-gbt-decision_boundary.svg")
      #image("img/espirales_hi-gbt-decision_boundary.svg")
      #colbreak()
      #image("img/lunas_hi-svc-decision_boundary.svg")
      #image("img/circulos_hi-svc-decision_boundary.svg")
      #image("img/espirales_hi-svc-decision_boundary.svg")
    ],
    caption: [Fronteras de decisión para #fkdc, #gbt, #svc en regímenes de alto ruido, $s = #plotting_seed$],
  )]]

Al ojo humano, queda claro que las fronteras y regiones de confianza que "dibuja" #fkdc se alinean "en espíritu" con la forma de las variedades que buscamos descubrir: la "región de indiferencia" gris en `lunas_hi` es una especie de curva casi-cúbica que efectivamente separa las lunas, el "huevo frito" de `circulos_hi` efectivamente tiene máxima confianza a favor de la clase interna en el centro de ambos círculos (y se va deformando progresivamente a medida que nos alejamos de él), y en `espirales_hi` casi logra dibujar la espiral. Sin embargo, esta deseable propiedad no es fácilmente reducible a una métrica en $RR$, y se desdibuja en las comparaciones puramente numéricas.

== Pionono, Eslabones, Hélices y Hueveras ($D=3$)

Consideraremos a continuación datasets sintéticos embebidos en 3 dimensiones ($D = 3$), con variedades de dimensión intrínseca  $1$ (`eslabones, helices`) y $2$ (`pionono, hueveras`).

=== Eslabones
#image("img/eslabones-scatter-3d.svg")

// TODO: poner scatter 3D en highlight por dataset para $D=3$
#highlights_figure("eslabones_0")

Toda la familia de estimadores de densidad por núcleos alcanza un $R^2 approx 1$, y aun Naive Bayes tiene una _performance_ aceptable: con este nivel de ruido blanco en el sampleo, el "margen de separación" entre ambos anillos es tan amplio que el problema resulta sencillo. Dicho esto, este dataset resulta ser particularmente fácil para (casi) todos los clasificadores.

Un punto en contra de #fkdc aquí es que el _boxplot_ de $R^2$ - no así el de exactitud - revela un fuerte outlier de $approx 0.65$ para la semilla $2411$, que no corresponde a una parametrización particularmente extraña.

#tabla_csv("data/eslabones_0-params-2411.csv")

==== Hélices
#image("img/helices-scatter-3d.svg")
Este dataset consiste en dos hélices del mismo diámetro y "enroscadas" en la misma dirección, una de ellas empezando a "media altura" entre dos brazos consecutivos de la otra. El dataset es particularmente desafiante para #slr, #logr, y Naive Bayes, que no logran diferenciarse en nada de un clasificador trivial que prediga siempre la misma clase.

#highlights_figure("helices_0")
Que a #gnb le resulte complejo no es sorprendente, ya que las distribuciones marginales son prácticamente idénticas:

#image("img/helices-pairplot.svg")

La clasificación dura con estimación de densidad por núcleos --- con distancia de Fermat o sin ella --- resulta ser superior a todas las alternativas en términos de exactitud, pero encima de ello, #fkdc saca una diferencia significativa en términos de $R^2$, que ya es visible en un boxplot pero se ve aún más claramente en el gráfico de dispersión de los $R^2$ alcanzados por semilla:

#image("img/helices_0-r2-fkdc-vs-kdc.svg")

En prácticamente todas las semillas el $R^2$ de #fkdc es estrictamente mejor al "control" de #kdc. ¿Con qué parámetros sucede?

#tabla_csv("data/helices_0-parametros_comparados-kdc.csv")

Ordenados por $Delta_(R^2) = R^2_#fkdc - R^2_#kdc$, la semilla con mayor diferencia a favor del resultado con distancia de Fermat corresponde a un no-trivial $(alpha = 1.25, h = 0.006)$ que resulta en un $Delta_(R^2) = 0.237 (= 0.953 - 0.716)$ puntos _en términos absolutos_#footnote[I.e., "un montón".] por encima de #kdc con $h = 0.208$, usando una ventana unas 35 veces más ancha.
Salta a la vista también que tales parametrizaciones tienen muy variada _performance_ _out-of-sample_, pues para $s = 8096$ _también_ se eligió $(alpha = 1.25, h = 0.006)$ contra $h_#kdc = 0.143 approx 25 h_#fkdc$ y se dio la segunda diferencia _negativa_ más amplia en contra de #fkdc ($Delta_(R^2) = -0.098$).

Gracias a la regla de parsimonia sabemos, por ejemplo, que para $s = 1188$ --- con el segundo mayor $Delta_(R^2) = 0.227$ --- no hay parametrizaciones con $alpha < 2.5$ a menos de 1SD de la mejor parametrización en _test_, cuyo $alpha = 3$.

#image("img/helices_0-1188-fkdc-bandwidth-alpha-loss_contour.svg")
Nótese la mínima isla alrededor de $alpha=3; h = 0,000562$.


Lo tercero, es que en unos cuantos casos en que $alpha_#fkdc = alpha_#kdc = 1$, #fkdc todavía performa un poco mejor que #kdc al elegir anchos de banda mucho más pequeños. Ya hemos visto que aún ligeras diferencias en la ventana $h$ podían llevar a mejoras en $R^2$ a favor de #fkdc por el detalle fino de la búsqueda en grilla que se definió. Sin embargo, aquí se encuentran sustanciales diferencias de $R^2$ como la tercera más alta ($Delta_R^2=0.111, alpha_#fkdc = alpha_#kdc = 1; h_#fkdc / h_#kdc approx 14)$, o la séptima ($Delta_R^2=0.111, h_#fkdc / h_#kdc approx 17$), que cuesta explicar como una ligera discrepancia en la grilla de $h$. Nuestra hipótesis es que el dominio ampliado de hiperparámetros de #fkdc junto con la regla de parsimonia trabajan en tándem:

#image("img/r1sd+alpha.svg")

Nuestro control, #kdc encuentra durante su entrenamiento y posterior testeo con R1SD la solución $h=0.143$ (cf. posición $(1)$ del diagrama). Presumiblemente, la varianza de la performance en testeo para dicha solución fue tal que ningún punto en el entorno de $h=0.01$ (cf. $(3)$) estaba a menos de 1SD de $(1)$. Cuando entrenamos #fkdc y ampliamos el dominio de la parametrización a $RR^2$ con $(h, alpha)$, la validación cruzada alcanza un máximo en $alpha=3; h = 0,000562$ (cf. $(2)$). Esta nueva solución, potencialmente "sobreparametrizada" con un $R^2_"train" = ; R^2_"test" = 0.988$, también tiene más varianza en sus resultados a través de cada pliego de CV, por lo que de repente ahora la cota inferior para ser considerada dentro de $cal(h)$ de @r1sd se vuelve más permisiva, en tanto se contrae menos por el aumento en el $R^2$ óptimo que lo que se relaja por el incremento en su varianza.
En ese rango ampliado de parametrizaciones "suficientemente buenas", ahora sí se encuentra $alpha=1; h=0.01$, y la CV "se mueve" de $(2) " a " (3)$, encontrando un óptimo en el espacio reducido de #kdc que este no llegó a considerar.

De hecho, es este fenómeno --- que se repite con las semillas `4286, 1182, 6610, 2411, 8527, 7060, 8591`, todas con $alpha <= 1.25$ --- el que termina desplazando la mediana de $R^2$ hasta $0.97$, por fuera de la "caja"#footnote[El rango intercuartil en el _boxplot_.] de $R^2_#kdc$.

#image("img/helices_0-boxplot-r2-zoomed.svg")

==== Efecto de #sfd en las vecindades óptimas de #kn

Si recordamos $hat(f)_(K,N)$ según Loubes & Pelletier, al núcleo $K$ se lo evalúa sobre
$
  (d (x_0, X_i)) / h, quad d = D_(Q_i, alpha)
$

Lo que $alpha$ afecta a $hat(f)$ vía $d$, también se puede conseguir vía $h$.

Si $D_(Q_i, alpha) prop ||dot||$ (la distancia de Fermat es proporcional a la euclídea), podemos escribir

$
  D_(Q_i, alpha) (x_0, X_i)) approx (c_alpha norm(dot))/ h = norm(x_0 - X_i) / h'
$
con $h' = h slash c_alpha$ y efectivamente los parámetros se solapan en sus funciones. Lamentablemente, sabemos que localmente esto _es_ cierto. Por ejemplo, la serie $k_n$ que minimiza el error cuadrátrico medio cuando $n -> oo$ es $k prop n^(d/(d+4))$, que para nuestro problema resulta en $(400 * 4/5 )^(3/(3+4)) =320^(3/7) approx 12$. Pues bien, cuando miramos el mejor rendimiento en test por `n_neighbors` para #kdc y #fkdc, vemos que elegir $alpha$ le permite a #fkdc mantener una óptima performance en términos de "score" ($-cal(l)$) para _cualquier_ valor de $k$ #footnote[`n_neighbors` en la parametrización de `scikit-learn`.]


#image("img/test-score-n_neighbors-fkn-kn.svg")

Por otra dirección, llegamos a la misma conclusión que antes: si un clasificador depende de distancias extremadamente locales, salvo que la muestra esté muy escasamente sampleada, el efecto de la distancia de Fermat aprendida de los datos no será muy notorio. // TODO: en trabajos posteriores estudiar efecto de alpha con $n$ fijo.

=== Pionono

#image("img/pionono-scatter-3d.svg")
#highlights_figure("pionono_0")

Este dataset es "clásico" para testear algoritmos de _clustering_ no-lineales @sapienzaWeightedGeodesicDistance2018, así que decidimos incluirlo en la serie experimental como _benchmark_. El trabajo citado no hace _clasificación_ con densidad por núcleos, sino _clustering_ basado en el algoritmo $k-$medoides, pero provee un gráfico de exactitud #footnote[presuntamente fijando $k=4$ y comparando las asignaciones contra los clusters verdaderos] que compara con la obtenida por otro "primo" algorítmico ya citado, Isomap. Los autores encuentran que
#quote[existe un amplio rango de $d$ #footnote[$alpha$ en nuestra notación] para los que la $d-$ distancia se porta significativamente mejor que Isomap. [...] para la exactitud esta región está limitada a $1.7 <= d <= 2.2$
]

Por nuestra parte, en un ambiente ligeramente distinto, no encontramos diferencia significativa con la _performance_ "cruda" de #kdc, que a su vez no se distingue de los métodos estado-del-arte en exactitud (#svc) ni $R^2$ (#gbt).

=== Hueveras ($D=3, d=2, k=2$)

Este dataset sumamente sintético consiste de dos clases con idénticas distribuciones pero signo opuesto en la dirección de la coordenada vertical#footnote[Cf. `fkdc/datasets.py` para ver el detalle de las fórmulas.], pero que se puede conceptualizar aproximadamente bien imaginando dos cartones de maple de huevos, uno invertido respecto al otro, intentando ocupar el mismo lugar en el espacio.
#image("img/hueveras-scatter-3d.svg")

La exactitud de la familia $cal(K)={#fkdc, #kdc, #fkn, #kn}$ es competitiva contra la de #svc, que parece ser ligera y significativamente mejor. En términos de $R^2$, la familia $cal(K)$ es la única en alcanzar valores no-nulos, y #sfd parece resultar en mejoras significativas, sobre todo para #fkn.
#highlights_figure("hueveras_0")


En efecto, observando los parámetros comparados de #fkdc v. #kdc, se repite que
- la mejor hiperparametrización $(alpha_"opt", h_"opt")$ en la grilla de CV tiene $alpha > 1$,
- hay una parametrización $(alpha_0, h_0)$ con $alpha_0 =1$ que cumple la regla de un desvío estándar,
- con $h_0$ "significativamente distinto" a $h_"opt"$#footnote[ Por ello nos referimos a que durante el entrenamiento de #kdc existió un $h_"alt" approx h_0$, que la R1SD + #kdc _no_ eligió, y la R1SD + #fkdc sí.].
#tabla_csv("data/hueveras_0-parametros_comparados-kdc.csv")

Esta "sinergia" virtuosa no alcanza para explicar lo que observamos del efecto de la distancia de Fermat en #kn:

#tabla_csv("data/hueveras_0-parametros_comparados-kn.csv")

A primera vista, se observan unas cuantas semillas para las cuales la elección de un $alpha > 1$ resultó en una diferencia de $R^2$ bastante positiva. Pero mejor aún, en 5 de 25 semillas ($s in {7074, 7060, 8443, 1434, 1193}$), #fkn y #kn maximizaron el objetivo con la _misma_ cantidad de vecinos, ¡y sin embargo #fkn eligió un $alpha > 1$!

= Conclusiones

A priori, nuestras tres propuestas de estimación:
- una implementación de la propuesta de  @loubesKernelbasedClassifierRiemannian2008 para clasificación suave en variedades con @clf-bayes y @kde-variedad;
- el reemplazo de la distancia euclídea por distancia de Fermat muestral en algoritmos de clasificación por núcleos (#kn, #kdc) y
- extender la estimación de la distancia de Fermat "microscópica" $cal(D)_(f,beta)$ a partir de la distancia de Fermat macroscópica #sfd, a puntos por fuera de la muestra
_funcionaron_, por separado y en conjunto, a la par de métodos de primera línea, paramétricos (#svc) y no paramétricos (#gbt). Al evaluarlos por "exactitud", a pesar de estar entrenados para maximizar la log-verosimilitud, los métodos resultaron competitivos aunque sin mejoras significativas. Al evaluarlos por $R^2$, sí se observaron excelentes rendimientos para toda la familia de métodos basados en densidad por núcleos $cal(K)$, y en ciertas ocasiones la distancia de Fermat se destaca por encima de la euclídea.

Ya existía una implementación previa de la Distancia de Fermat #link("https://www.aristas.com.ar/fermat/fermat.html")[como librería de Python] orientada a "clustering" @sapienzaWeightedGeodesicDistance2018, tarea que tiene la particularidad de entrenar y predecir sobre los mismos datos. El problema de clasificación se evalúa, para ser justos, en observaciones que _no_ se usaron para entrenar, lo cual nos llevó a escribir una librería nueva, con menos opciones de parametrización, pero capacidad de estimación "out-of-sample" y una implementación mínimamente performante sobre métodos bien optimizados que nos permitan ejecutar una suite extensa de experimentos que pudiésemos refinar iterativamente.

Poner a legos a implementar algoritmos numéricos complejos no suele terminar bien, pero milagrosamente llevamos el invento a buen puerto. También podía ser que el método tuviese una performance _decente_ pero no _competitiva_ con el estado del arte; no fue el caso.

// TODO: integrar referencia al paper que argumenta que basta con aprender alpha, que junta $d$ y $beta$ en uno solo.

En ninguno de los datasets estudiados (casos con bajo $D in {2, 3}$) se vieron modos "catastróficos" donde la _performance_ de #fkdc/#fkn fuese muchísimo peor que la de sus pares euclídeos. En los datasets en que se comprueba una ventaja sistemática de #fkdc (resp. #fkn) sobre #kdc (resp. #kn), se puede explicar por dos efectos:
- En todos los casos examinados, una parte importante de la ventaja se da por una "simbiosis" positiva entre el mecanismo de selección de modelos de @r1sd, y el espacio de parámetros ampliado por la dimensión de $alpha$. Esta resulta en parametrizaciones de #fkdc (resp. #fkn) con $alpha=1$ y ligeramente mejor $R^2$ que #kdc (resp. #kn) ignora.
- En ciertos casos (como #fkn en `hueveras_0`), acontece que parte de la mejora se debe a la elección de parametrizaciones de #fkn que coinciden en el $k$ elegido con #kn, pero además registran un $alpha > 1$ - i.e., una mejora _netamente gracias al_ uso de la distancia de Fermat muestral.

En el "caso general", en el que #fkdc anda tan bien o mal como #kdc, observamos una relación log-lineal, $ log(h) prop alpha $ que se discierne en la _superficie de pérdida_ de entrenamiento como un "risco" de parametrizaciones equivalentes en bondad. Entendemos que esto sucede porque
- los datasets están "bien sampleados" y
- para todo $p in MM$ una variedad de Riemann, siempre existe un vecindario dentro del radio de inyectividad de $"iny"_p MM$ en el que $cal(D)_(f,beta) prop norm(dot)$
En estas circunstancias existe un $h <= "iny"_p MM$ tal que el efecto de $alpha$ "(des)inflando" la distancia euclídea puede ser sustituido completamente por una parametrización con distinto $h$, y no hay ventaja alguna que obtener usando distancia de Fermat (#fkn o #fkdc) en lugar de euclídea.

Los métodos de estimación por densidad de núcleos son "altamente locales", y por ende solo vemos mejoras no-triviales de $R^2$ en circunstancias extraordinarias, como en los datasets de `espirales`, `helices` o `hueveras` en que aun los vecindarios locales son altamente no-euclídeos.

Con respecto a los tiempos de cómputo, no se hizo un análisis exhaustivo esencialmente porque no hizo falta: corrimos 25 repeticiones de 20 datasets para 9 clasificadores en unas 12 horas en mi computadora personal #footnote[Macbook Air M1 2020, 8GB RAM, 256GB SSD] media docena de veces hasta tenerlo todo a punto, y en general ni siquiera fue #sfd el algoritmo más problemático. Es cierto que el cómputo de #sfd - que implica calcular geodésicas en grafos completos - puede requerir varios órdenes de magnitud mayores recursos que el de la distancia euclídea, pero
- para datasets "moderados" (en el desarrollo se consideraron $n_k <= 1000, p <= 90$) el tiempo de cómputo de base es pequeñísimo, y aunque crezca por órdenes de magnitud no afecta significativamente la experiencia del científico; además
- con estrategias básicas de "cacheo" #footnote[Confer el uso del decorador `joblib.Memory.cache` en `fkdc/fermat.py`], se puede computar una única vez las distancias de Fermat, y reutilizarlas en todas las evaluaciones posteriores de distancias de entrenamiento.


== Trabajo Futuro
En el presente trabajo hemos desarrollado una librería y un marco teórico sumamente riguroso para intentar identificar condiciones en las cuales estimadores de densidad entrenados con distancia de Fermat muestral son estrictamente mejores que sus versiones euclídeas.

Es _infinita_ la cantidad de circunstancias en las que podemos poner a prueba una técnica de clasificación, y en los experimentos ejecutados y presentados no hemos hecho más que rascar la superficie. Así y todo, pareciera ser que en espacios ralamente sampleados o altamente curvos, donde "no quede otra" que tomar una ventana $h > "iny" MM$ para tener una densidad "viable", el uso de la distancia de Fermat mejora, si no la exactitud de los algoritmos, sí su $R^2$ y por ende la capacidad de discernimiento "relativo" de estos estimadores.

Sería interesante entonces investigar si existen condiciones reales en las que sepamos "a priori" que las variedades intrínsecas son altamente no-euclídeas, y en ese contexto probar si en ciertos tamaños muestrales $n$ (y por cada clase, $n_1, dots, n_k$) pequeños relativos a la dimensión ambiente es particularmente conveniente el uso de la distancia de Fermat.

== A incorporar
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




= Listados

#outline(target: figure.where(kind: image), title: "Listado de Figuras")
#outline(target: figure.where(kind: table), title: "Listado de Tablas")
#bibliography("../bib/references.bib", style: "harvard-cite-them-right")


= Apéndice A: Fichas de resultados por dataset <apendice-a>


== Arenero

// TODO: revisar forma de citar bibliografía, corregir autores y formato en textos enciclopédicos y otros
#tabla_csv("data/2-blobs.csv")
#let best(..contents) = {
  contents.pos().map(content => table.cell(fill: rgb("#7cff9dc9"), content))
}

#let bad(..contents) = {
  contents.pos().map(content => text(fill: black.transparentize(70%), content))
}

#show table.cell.where(y: 0): set text(weight: "bold")
#let na = align(center)[--]
#table(
  columns: 3,
  stroke: none,
  align: (x, y) => if y == 0 { center } else { if x == 0 { right } else { top } },
  table.header[clf][$R^2$][exac],
  table.hline(stroke: 1pt),
  table.vline(x: 1, start: 1, stroke: .5pt),
  [#fkdc], [1.0], [1.0],
  [#kdc], [1.0], [1.0],
  [#gnb], [1.0], [1.0],
  [#kn], [1.0], [1.0],
  [#fkn], [1.0], [1.0],
  [#logr], [0.99994], [1.0],
  ..best([#slr], [0.99952], [1.0]),
  ..bad([#gbt], [0.9995], [1.0]),
  ..bad([#svc], [#na], [1.0]),
)

#include "mi-tabla.typ"
