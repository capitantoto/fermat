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

#let tabla_csv(path, caption: none, short-caption: none) = {
  let data = csv(path)
  let scope = (fkdc: fkdc, kn: kn, fkn: fkn, kdc: kdc, lr: logr, svc: svc, gnb: gnb, gbt: gbt, slr: slr)
  let headers = data.at(0)
  let rows = data.slice(1)
  let cells = (
    table.hline(stroke: 1pt),
    ..headers.map(h => table.cell(align: center)[*#eval(h, mode: "markup", scope: scope)*]),
    table.hline(stroke: 0.5pt),
    ..rows.flatten().map(eval.with(mode: "markup", scope: scope)),
    table.hline(stroke: 1pt),
  )
  let t = table(
    columns: headers.len(),
    stroke: none,
    ..cells,
  )
  if caption != none {
    figure(t, caption: flex-caption(caption, if short-caption != none { short-caption } else { caption }))
  } else {
    t
  }
}


// ### TOC y listados
#outline(depth: 2)

#pagebreak()

= Vocabulario y Notación

A lo largo de esta monografía tomaremos como referencia enciclopédica el excelente _Elements of Statistical Learning_ @hastieElementsStatisticalLearning2009. En la medida de lo posible, basaremos nuestra notación en la suya.

Típicamente, denotaremos a las variables independientes #footnote[También conocidas como predictoras, o #emph[inputs]] con $X$. Si $X$ es un vector, accederemos a sus componentes con subíndices, $X_j$. En el contexto del problema de clasificación, la variable _cualitativa_ dependiente #footnote[También conocida como variable respuesta u #emph[output]] será $G$ (de "G"rupo). Usaremos letras mayúsculas como $X, G$ para referirnos a los aspectos genéricos de una variable. Los valores _observados_ se escribirán en minúscula, de manera que el i-ésimo valor observado de $X$ será $x_i$ (de nuevo, $x_i$ puede ser un escalar o un vector).

Representaremos a las matrices con letras mayúsculas en negrita, #XX; e.g.: el conjunto de $N$ vectores $p$-dimensionales ${x_i, i in {1, dots, N}}$ será representado por la matriz #XX de dimensión $N times p$.

En general, los vectores _no_ estarán en negrita, excepto cuando tengan $N$ componentes; esta convención distingue el $p-$vector de #emph[inputs] para la i-ésima observación,  $x_i$, del $N-$vector $bu(x)_j$ con todas las observaciones de la variable $X_j$. Como todos los vectores se asumen vectores columna, la i-ésima fila de #XX es $x_i^T$, la traspuesta de la i-ésima observación $x_i$. El elemento de la $i$-ésima fila y $j$-ésima columna de la matrix #XX se notará $XX_(i,j)$.

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
  image("img/dos-circulos-jointplot.svg", width: 75%),
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
  image("img/curse-dim.svg"),
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
La frontera de $B_epsilon (p)$ es una "subvariedad" de #MM ortogonal a las geodésicas que irradian desde $p$. Una concepción intuitiva de qué es una bola normal es "un entorno de $p$ en el que las geodésicas que pasan por $p$ son minimizadoras de distancias". El siguiente concepto es útil para entender "cuán lejos vale" la aproximación local a un espacio euclídeo en la variedad.

#defn(
  [radio de inyectividad #footnote[Basado en @munozEstimacionNoParametrica2011[Def. 3.3.16] Una definición a mi entender más esclarecedora se encuentra en @docarmoRiemannianGeometry1992[§13.2, _The cut locus_], que introducimos aquí informalmente. El _cut locus_ o _ligne de partage_ $C_m (p)$ - algo así como la línea de corte - de un punto $p$ es la unión de todos los puntos de corte: los puntos a lo largo de las geodésicas que irradian de $p$ donde éstas dejan de ser minimizadoras de distancia. El ínfimo de la distancia entre $p$ y su línea de corte es el radio de inyectividad de #MM en $p$, de modo podemos escribir $ "iny" MM = inf_(p in MM) d(p, C_m (p)) $
      donde la distancia de un punto a una variedad es el ínfimo de la distancia a todos los puntos de la variedad.]],
)[
  Sea $(MM, g)$ una $d-$variedad Riemanniana. Llamamos "radio de inyectividad en $p$" a
  $
    "iny"_p MM = sup{s in RR > 0 : B_s (p) " es una bola normal"}
  $
  El ínfimo de los radios de inyectividad "puntuales" es el radio de inyectividad de la variedad #MM.
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

La belleza de esta regla es que combina "sin costuras" el peso de los _priors_ $hat(pi)_i$ - a través de los elementos no nulos de la suma cuando $ind(G_i = k) = 1$) - con el peso de la "evidencia" - vía su cercanía "suavizada" al punto de interés $K_h (p, X_i)$.

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

La pieza clave del algoritmo es la estimación de la distancia geodésica en #MM a través de la distancia en el grafo de vecinos más cercanos. Si la muestra disponible es "suficientemente grande", es razonable esperar que en el entorno de una observación $x_0$ las distancias euclídeas aproximen bien las distancias geodésicas, y por ende un "paseo" por el grafo $bu(N N)$ debería describir una curva prácticamente contenida en #MM. Isomap resultó ser un algoritmo sumamente efectivo que avivó el interés por el aprendizaje de distancias, pero todavía cuenta con un talón de Aquiles: la elección del parámetro de cercanía, $epsilon$ ó $k$:
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
  gamma(t) & : [0, 1] -> RR^D, quad
             gamma(t) & = a + (b - a) t, quad
                        gamma'(t) & = b - a
$
$
  d_bu(N)(a, b) & = D_(r_P) = inf_gamma J_(r_P) (gamma) = J_(r_P) (overline(a b)) \= integral_0^1 r_{a, b} (gamma(t)) times norm(gamma'(t)) dif t \
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

Una utilidad de este resultado es que permite calcular con precisión qué valores de $k$ estimar $d_bu(N)$ sobre el grafo pesado por aristas cuadradas $bu(N N)_k (XX)$  es un "suficientemente buen reemplazo" del cálculo equivalente --- pero mucho más costoso --- sobre $bu(C)(XX)$. En @chuExactComputationManifold2019[Theorema 1.3], observan que con tomar $k = O(2^d ln n)$ basta.

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
] <convergencia-sfd>

#obs[
  El factor de escala $beta = (alpha-1)/d$ depende de la dimensión intrínseca $d$ de la variedad, y no de la dimensión $D$ del espacio ambiente.
]

La distancia muestral de Fermat $D_(Q, alpha)$ se puede aproximar a partir de una muestra "lo suficientemente grande _sin conocer ni la variedad #MM ni su dimensión intrínseca_. Además, tiene garantías de convergencia a una distancia basada en densidad (DBD) --- la distancia de Fermat "macroscópica" $cal(D)_(f, beta)$ --- para todo $beta,$. ¡Hemos encontrado la pieza faltante para nuestro clasificador en variedades _desconocidas_! Estamos finalmente en condiciones de proponer un algoritmo de clasificación que reúna todos los cabos del tejido teórico hasta aquí desplegado.

Los trabajos de @littleBalancingGeometryDensity2021 @mckenziePowerWeightedShortest2019 --- contemporáneos a Groisman et al --- consideran lo que ellos llaman "distancias de caminos mínimos pesadas por potencias" #footnote["power-weighted shortest-path distances" o PWSPDs por sus siglas en inglés], aplicándoles no a problemas de clasificación, sino de _clustering_ #footnote[i.e., de identificación de grupos en datos no etiquetados]. Las definiciones de ambos grupos son muy similares en espíritu, con una diferencia menor: la distancia microscópica que plantean Little et al no es la suma de las aristas pesadas por $q=alpha$ como en Bijral et al y Groisman et al, sino la raíz $alpha$-ésima de tal suma, en una especia de reversión de la distancia de Minkowski. Siendo la sustancia de ambos trabajos en esencia la misma, pasaremos directamente a la próxima sección --- nuestra propuesta original.

= Propuesta Original

Al comienzo de este sendero teórico nos preguntamos: ¿es posible mejorar un algoritmo de clasificación reemplazando la distancia euclídea por una aprendida de los datos? Habiendo explorado el área en profundidad, entendemos que sí pareciera ser posible, y en particular la distancia muestral de Fermat #sfd es un buen candidato de reemplazo. Deseamos también comprender si el efecto de la #sfd aprendida es independiente del algoritmo de clasificación que la incorpora. Para saldar ambas cuestiones, nos propusimos:

1. Implementar un clasificador basado en estimación de densidad por núcleos según @kde-variedad @loubesKernelbasedClassifierRiemannian2008, al que llamaremos "KDC" #footnote[_Kernel Density Classifier_, por sus siglas en inglés].
2. Implementar un estimador de densidad por núcleos basado en la distancia de Ferma, "F-KDC", a fines de comparar la _performance_ de KDC con distancia euclídea y con distancia de Fermat.
3. Implementar un clasificador de $k$ vecinos más cercanos según @kn-clf, pero con distancia muestral de Fermat en lugar de euclídea.
4. Comparar sistemáticamente la capacidad de clasificación de cada algoritmo propuesto --- y algunos más de referencia --- en datasets de diversas características.
5. Analizar los resultados e identificar en qué condiciones es que la distancia de Fermat aporta mejoras significativas sobre la tradicional distancia euclídea.

El método de aprendizaje de la distancia muestral de Fermat y los tres algoritmos novedosos #footnote[En los tres ser requirieron desarrollos nuevos al menos parcialmente. KDC en variedades según @clf-kde-variedad está definido en @loubesKernelbasedClassifierRiemannian2008 pero no implementado; la estimación de densidad por núcleos  multivariada  de @kde-mv cuenta con múltiples implementaciones en código pero no conocemos algorimos de clasificación "llave en mano" que se basen en ella; $k-$NN como en @kn-clf es un algoritmo de clasificación harto común que soporta distancias no-euclídeas, pero no la distancia de Fermat específicamente.] componen un repositorio de código abierto que acompaña esta tesis y está a disposición de cualquier investigador que desee corroborar los resultados en Github #footnote[https://github.com/capitantoto/fermat].

A continuación, mencionamos algunos aspectos salientes sobre los desarrollos de código necesarios así como la metodología de evaluación diseñada, antes de pasar a los resultados.

== Estimación de distancia de Fermat _out-of-sample_

Un proyecto de código pre-existente a esta monografía ya implementa el cálculo de la distancia de Fermat microscópica o muestral para un conjunto de observaciones dado: #link("https://pypi.org/project/fermat/")[fermat], de Facundo Sapienza. Este paquete fue desarrollado para soportar los experimentos de @sapienzaWeightedGeodesicDistance2018 que exploran los efectos de esta noción de distancia en tareas de _clustering_. Al ser una tarea no-supervisada #footnote[Una tarea supervisada de aprendizaje es aquella en que se entrena el algoritmo con un conjunto de observaciones para el que _ya se sabe_ el valor correcto de resuesta. Una tarea "no supervisada" no cuenta con una "respuesta correcta" de antemano. _Clustering_ --- identificar grupos en la muestra --- es una tarea no supervisada; _clasificación_ --- asignar elementos a clases conocidas de antemano --- es una tarea supervisada.], se utilizan todas las observaciones disponibles y solo se requiere calcular la distancia entre dos elementos cualesquiera de la muestra #XX, pero nunca contra otros $p : p in MM, p in.not XX$.

Entrenar un algoritmo _supervisado_ de clasificación requiere apartar una fracción de las observaciones disponibles #footnote[De no hacerlo y evaluar al clasificador sobre los mismos datos de entrenamiento, se corre el riesgo de sobreajustar el clasificador a los datos. De entrenar $k-$NN con tal criterio $k = 1$ acertará la clase correcta _siempre_, ya que cada observación es su propia vecina con distancia cero.] para evaluar la pérdida objetivo $L$. ¿Cómo calculamos entonces la distancia _muestral_ de una _nueva_ observación $x_0$ a los elementos de cada grupo $GG_i, i in [K]$?

Para cada una de las $GG_i in GG$ clases, definimos el conjunto $ Q_i= {x_0} union {x_j : x_j in XX, GG_j = GG_i} $
y calculamos $D_(Q_i, alpha)$. No es difícil en principio este cómputo, pero resultaría absurdamente costoso computacionalmente recomputar $D_(Q_i, alpha)$ para cada una de las $K$ clases por cada nueva observación. En su lugar, implementamos un sencillo algoritmo "incremental", que permite recomputar únicamente las geodésicas que cambian al agregar la nueva observación $x_0$ al grafo completo de la clase en cuestión.

== Elección del ancho de banda para clasificación

Al estimar densidades con distancia de Fermat en una variedad, la elección del ancho de banda se simplifica considerablemente: en lugar de una matriz completa $HH$ como en el KDE multivariado euclídeo, basta con dos escalares --- $h$ y $alpha$. Idealmente, convendría elegir un par $(h_i^*, alpha_i^*)$ óptimo para cada clase $GG_i$, ya que las densidades individuales pueden diferir sustancialmente.

Sin embargo, @hallBandwidthChoiceNonparametric2005 muestran que el $h$ óptimo para la estimación de densidad no es necesariamente el óptimo para la _clasificación_: la tarea de clasificar no requiere estimar bien la densidad en todo el soporte, sino distinguir bien _en las fronteras_ entre clases. Teniendo esto en cuenta y para simplificar la configuración, parametrizamos #fkdc con un único $h$ y $alpha$ globales #footnote[y #kdc con un único $h$, elegido sobre una grilla mucho más fina que la de #fkdc]. La búsqueda de simplicidad en la configuración no es un deseo, es una necesidad: la elección de hiperparámetros óptimos por validación cruzada en una grilla requiere entrenar el mismo clasificador en una cantidad de configuraciones que crece exponencialmente con la cantidad de hiperparámetros, lo cual prohíbe configuraciones mucho más complejas que unos pocos parámetros.

== Metodología

La unidad de evaluación de los algoritmos a considerar es una `Tarea` #footnote[cf. el archivo `fkdc/tarea.py` en el repositorio adjunto para más detalles.], que se compone de:
- un _dataset_ con el conjunto de $N$ observaciones en $D$ dimensiones repartidas en $K$ clases, $(XX, bu(g))$,
- un _split de evaluación_ $r in (0, 1)$, que determina las proporción de los datos a incluir en la muestra de entrenamiento $XX_"train"$ ($1 - r$) y la de evaluación $XX_"test"$ ($r$),
- una _semilla_ $s in [2^32]$ que alimenta el generador de números aleatorios y define determinísticamente cómo realizar la división antedicha y
- una _métrica de evaluación_ #footnote[en muchos casos ésta coincidirá con la función de pérdida $L$ a minimizar durante el entrenamiento, pero no necesariamente] que resume la "bondad" de las predicciones sobre $XX_"test"$ del clasficador entrenado en $XX_"train"$.

=== Métricas de evaluación

En tareas de clasificación, la métrica más habitual es la _exactitud_ #footnote([Más conocida por su nombre en inglés, _accuracy_.])

#defn(
  "exactitud",
)[Sean $(XX, bu(g)) in RR^(N times p) times RR^N$ una matriz de $N$ observaciones de $p$ atributos y sus clases asociadas. Sea además $hat(bu(g)) = hat(G)(XX)$ las predicciones de clase resultado de una regla de clasificación $hat(G)$. La _exactitud_ ($"exac"$) de $hat(G)$ en #XX se define como la proporción de coincidencias con las clases verdaderas $bu(g)$:
  $ op("exac")(hat(G) | XX) = n^(-1) sum_(i=1)^n ind(hat(g)_i = g_i) $
] <exactitud>

La exactitud está bien definida para cualquier clasificador que provea una regla _dura_ de clasificación. Ahora bien, cuando un clasificador provee una regla suave, la exactitud como métrica pierde información: dos clasificadores binarios que asignen respectivamente 0.51 y 1.0 de probabilidad de pertenecer a la clase correcta a todas las observaciones tendrán la misma exactitud, $100%$, aunque el segundo es a las claras mejor. A la inversa, cuando un clasificador erra al asignar la clase: ¿lo hace con absoluta confianza, asignando una alta probabilidad a la clase equivocada, o con cierta incertidumbre, repartiendo la masa de probabilidad entre varias clases que considera factibles? Una métrica natural para evaluar una regla de clasificación suave es la _verosimilitud_ de las predicciones.

#defn(
  "verosimilitud",
)[Sean $XX, bu(g)$ como en @exactitud . Sea además $hat(bu(Y)) = clf(XX) in RR^(n times k)$ la matriz de probabilidades de clase resultado de una regla suave de clasificación #clf. La _verosimilitud_ ($"vero"$) de #clf en #bu("X") se define como la probabilidad conjunta que asigna #clf a las clases verdaderas #bu("y"):
  $
    op("vero")( clf | XX ) & = Pr(hat(bu(y)) = bu(y)) = product_(i=1)^n Pr(hat(y)_i =y_i) \
    op("vero")( clf | XX ) & = product_(i=1)^n hat(bu(Y))_(i, y_i)
  $

  Por conveniencia, se suele considerar la _log-verosimilitud promedio_,
  $ op(cal(l))(clf) = n^(-1) log(op("L")(clf)) = n^(-1)sum_(i=1)^n log(hat(bu(Y))_((i, y_i))) $
] <vero>

La verosimilitud de una muestra varía en el rango $[0, 1]$ y su log-verosimilitud, en $(-oo, 0]$. Como métrica, esta se vuelve comprensible al expresarla _relativa a otros clasificadores_, por ejemplo, como propone @mcfaddenConditionalLogitAnalysis1974.

#defn(
  [$R^2$ de McFadden],
)[Sea $clf_0$ el clasificador "nulo", que asigna a cada observación y posible clase, la frecuencia empírica de clase encontrada en la muestra de entrenamiento $XX_("train")$. Para todo clasificador suave $clf$, definimos el $R^2$ de McFadden como
  $ op(R^2)(clf | XX) = 1 - (op(cal(l))(clf)) / (op(cal(l))(clf_0)) $
] <R2-mcf>

#obs[ $op(R^2)(clf_0) = 0$. Un clasificador perfecto --- un "oráculo" --- $clf^star$ que otorgue toda la masa de probabilidad a la clase correcta, tendrá $op(L)(clf^star) = 1$ y log-verosimilitud igual a 0, de manera que $op(R^2)(clf^star) = 1 - 0 = 1$. Un clasificador _peor_ que $clf_0$ en tanto asigne bajas probabilidades a las clases correctas, puede tener un $R^2$ infinitamente negativo.
]

Tanto #kdc, #fkdc y #fkn son clasificadores suaves, por lo que los evaluaremos principalmente según el $R^2$ de @R2-mcf. Sin embargo, mantendremos un ojo en la exactitud de @exactitud, para asegurarnos de que su _performance_ en esta métrica tan tradicional no sea significativamente peor que la de los algoritmos de referencia.

=== Algoritmos de referencia

Pírrica victoria sería mejorar con la distancia de Fermat la _performance_ de #kdc o #kn para encontrar que aún así, el algoritmo no es competitivo contra el estado del arte en la misma tarea. A modo de referencia consideramos también los siguientes algoritmos:
- Naive Bayes Gaussiano (#gnb),
- Regresión Logistica (#logr),
- _Gradient Boosting Trees_ (#gbt) #footnote[Una traducción literal sería "árboles (de decisión) por potenciación del gradiente", pero este término casi nunca se traduce en la práctica.] y
- Clasificador de Soporte Vectorial (#svc)

#v(1em)

Esta elección no pretende ser exhaustiva, sino que responde a un "capricho informado" del investigador. Naive Bayes (@naive-bayes) una elección natural, ya que es la simplificación que surge de asumir independencia en las dimensiones de $X$ para KDE multivariado (@kde-mv), y se puede computar para grandes conjuntos de datos en muy poco tiempo.

La regresión logística es "el" método para clasificación binaria, y su extensión a múltiples clases no es particularmente compleja. Para resultar mínimamente valioso, un nuevo algoritmo necesita ser al menos tan bueno como #logr y sus ya más de 80 años en el campo #footnote[la referencia más temprana a la regresión logística data de @berksonApplicationLogisticFunction1944; la referencia clásica al marco formal moderno está en @coxRegressionAnalysisBinary1958. Un trabajo aún anterior sobre estimación de probabilidades pero usando la función _probit_ --- la distribución acumulada de la normal estándar --- en lugar de la función _logit_ o sigmoidea, es @blissCALCULATIONDOSAGEMORTALITYCURVE1935].

Por último, fue nuestro deseo incorporar algunos métodos más cercanos al estado del arte. A tal fin incorporamos un método de _boosting_ #footnote[ El _gradient boosting_ fue introducido por @friedmanGreedyFunctionApproximation2001, y desde entonces ha dado lugar a implementaciones altamente eficientes como XGBoost @chenXGBoostScalableTree2016 y LightGBM @keLightGBMHighlyEfficient2017.] y el antedicho clasificador de soporte vectorial. El clasificador de soporte vectorial @cortesSupportvectorNetworks1995, #svc, se evaluó en dos variantes: con núcleos (_kernels_) lineales y RBF #footnote[del inglés _radial basis functions_, "funciones de base radial"].


Por conocerlo en profundidad y en virtud de su sencillez de uso, la implementación se realizó utilizando `scikit-learn` @JMLR:v12:pedregosa11a, un poderoso y extensible paquete para tareas de aprendizaje automático en Python.


=== Pre-tratamiento de los datos

Mantuvimos al mínimo el pre-tratamiento de los datos de entrada. Esta decisión fue deliberada: nos interesaba evaluar si la distancia de Fermat era capaz de capturar la estructura de la variedad subyacente sin asistencia adicional en la preparación de los datos, si bien reconocemos que este supuesto no es del todo razonable en aplicaciones del mundo real, donde el pre-procesamiento suele ser una etapa fundamental.

La excepción fue la regresión logística, de la cual es bien sabido que su _performance_ se degrada considerablemente cuando las variables predictoras se encuentran en escalas muy distintas. Por este motivo incluimos tanto #logr --- regresión logística sobre los datos originales --- como #slr, una variante en la que los datos fueron previamente estandarizados.

=== Entrenamiento de los algoritmos
La especificación completa de un clasificador requiere no solo elegir un algoritmo sino también especificar de sus _hiperparámetros_, a fin de optimizar su rendimiento bajo ciertas condiciones de evaluación. Para ello, se definió de antemano para cada clasificador una _grilla_ de hiperparámetros: durante el proceso de entrenamiento, la elección de los "mejores" hiperparámetros se efectuó maximizando la log-verosimilitud @vero para los clasificadores suaves, y la exactitud @exactitud para los duros #footnote[Entre los mencionados, el único clasificador duro es #svc. Técnicamente es posible entrenar un clasificador suave a partir de uno duro con un _segundo_ estimador que toma como _input_ el resultado "crudo" del clasificador duro y da como _output_ una probabilidad calibrada (cf. #link("https://scikit-learn.org/stable/modules/calibration.html")[Calibración] en la documentación de `scikit-learn`  @buitinckAPIDesignMachine2013), pero es un proceso computacionalmente costoso.] con una búsqueda exhaustiva por validación cruzada de 5 pliegos #footnote[Conocida en inglés como #emph[Grid Search 5-fold Cross-Validation]] sobre la grilla entera.

=== Estimación de la variabilidad en la _performance_ reportada
En última instancia, cualquier métrica evaluada no es otra cosa que un _estadístico_ que representa la "calidad" del clasificador en la tarea a mano. A fines de conocer no solo su estimación puntual sino también darnos una idea de la variabilidad de su _performance_, para cada dataset y colección de algoritmos, se entrenaron y evaluaron #reps tareas idénticas salvo por la semilla $s$, que luego se usaron para estimar estadísticos de locación (media, mediana, rango intercuartil) y dispersión (varianza y desvío estándar) en la exactitud (@exactitud) y el pseudo-$R^2$ (@R2-mcf).

Cuando el conjunto de datos proviene del mundo real y por lo tanto _preexiste a nuestro trabajo_, las #reps semillas $s_1, dots, s_#reps$ fueron utilizadas para definir diferentes partticiones de entrenamiento/evaluación. Por el contrario, cuando el conjunto de datos fue generado sintéticamente, las semillas se utilizaron para generar #reps versiones distintas pero perfectamente replicables del dataset, y en todas se utilizó una misma semilla maestra $s^star$ para definir el _split_ de evaluación.


=== Regla de Parsimonia

La estrategia de validación cruzada intenta evitar que los algoritmos sobreajusten durante el entrenamiento, evaluando su comportamiento en $XX_"test"$, disjunto de $XX_"train"$.
No todas las parametrizaciones son equivalentes: en general, para cada hiperparámetro se puede establecer una dirección en la que el modelo se complejiza, en tanto se ajusta más y más a los datos de entrenamiento #footnote[Por ejemplo, #kn se complejiza a medida que  _disminuye_ $k$, la cantidad de vecinos: las predicciones de $1-$NN sobre la variedad cambian más seguido que las de $100$-NN]. Esto nos recuerda un principio filosófico clásico:

#obs(link("https://es.wikipedia.org/wiki/Navaja_de_Ockham")[Navaja de Occam])[
  Atribuida a William de Ockham (c. 1287--1347), también se conoce como "Principio de Parsimonia", y se suele citar --- en palabras que su autor nunca pronunció exactamente --- como _Entia non sunt multiplicanda praerter necessitatem_, "No se deben multiplicar las entidades sin necesidad". Popularmente, se suele parafrasear como "de entre dos teorías en disputa, es preferible la explicación más simple de un fenómeno".
]
Reformulando, diremos que sujeto a la implementación de _cierto_ algoritmo, cuando dos hiperparametrizaciones $nu, mu$ tienen _casi_ las mismas consecuencias --- alcanzan pérdidas tales que $abs(L(nu) - L(mu)) < c$ con $c$ "suficientemente pequeño" --- preferiremos la más sencilla: la de menor _complejidad_ $C$, para cierta función $C$ a definir.

La validación cruzada de $k$ pliegos nos provee naturalmente de $k$ realizaciones de la métrica a optimizar para cada hiperparametrización, que podemos utilizar para estimar el desvío estándar de la misa. Sobre esta base, implementamos la siguiente regla:
#defn([regla de un desvío estándar o "R1SD"])[
  Sea $mu^star$ la hiperparametrización que minimiza la pérdida de entrenamiento y $hat(s)(L(mu^star))$ el desvío estimado de dicha pérdida. De entre todas las hiperparametrizaciones casi tan satisfactorias como $mu^star$, elíjase _la más sencilla_:
  $         & mu^(1 sigma) = arg min_(mu in Mu) C(mu) \
  "donde" & Mu = {mu : L(mu) <= L(mu^star) + hat(s)(L(mu^star))) } $.
] <r1sd>

Para definir $C$ en modelos con $dim(h) > 1$, definimos el orden de complejidad creciente _para cada clasificador_ jerárquicamente como una lista de pares ordenados de hiperparámetros y la dirección de complejidad creciente. Para #fkdc, por ejemplo,
$ C_#fkdc (mu) = [(alpha, "ascendente"), (h, "descendente")]. $
La decisión de ordenar así los parámetros, con $alpha$ primero y $C$ ascendente en $alpha$, hace que la evaluación "prefiera" naturalmente a #kdc por sobre #fkdc#footnote[$#kdc = op(#fkdc)(alpha = 1)$], ya que el mínimo $alpha = 1$ estudiado resulta siempre preferido. En consecuencia, solo se elegirá un $alpha^star > 1$ cuando la _performance_ de #fkdc sea significativamente mejor que la de KDC --- con $alpha equiv 1$.

#obs([complejidad en $h$])[
  La complejidad es _descendente_ en el tamaño de la ventana $h$: a mayor $h$, tanto más grande se vuelve el vecindario donde $K_h (d(x, x_i)) >> 0$ y por ende pesa en la asignación. Análogamente, $k-"NN"$ y su primo $epsilon- "NN"$ tienen complejidad _descendente_ en $k, epsilon$.
]

=== Medidas de locación y dispersión no-paramétricas
Nos dedicaremos a la  estimación de densidad basada en distancia de Fermat en una variedad de Riemann desconocida. Resulta imposible conocer _a priori_ la teoría de la distribución para estos estimadores, por lo que nos resulta razonable comparar la _performance_ con medidas de locación robustas. Por ello compararemos la _performance_ mediana (y no media) entre las #reps repeticiones con distintas semillas de cada clasificador, y las visualizaremos con un _boxplot_ en lugar de un intervalo de confianza.

= Resultados <resultados>

== In Totis
// TODO: Repasar esta sección al terminar la edición de "= Resultados"
En total, ejecutamos unas 4,500 tareas, producto de #reps repeticiones por dataset y clasificador, sobre un total de 20 datasets y 9 clasificadores diferentes. Recordemos que todos los estimadores se entrenaron con _score_ `neg_log_loss` (para optimizar por $R^2$), salvo #svc, que al ser un clasificador duro se entrenó con `accuracy`. Así, entre los clasificadores blandos la distancia de Fermat rindió frutos, con el máximo $R^2$ mediano en 10 de los 20 experimentos: 7 preseas fueron para #fkdc y 3 para #fkn.

#gbt "ganó" en 5 datasets, entre ellos en varios con mucho ruido (`_hi` y `_12`). #kdc resultó óptimo en 2 datasets, consolidando la técnica del @kde-variedad como competitiva de por sí. Por último, tanto #kn como #logr (en su versión escalada, #slr) resultaron medianamente mejores que todos los demás en ciertos datasets, y solo #gnb no consiguió ningún podio - aunque resultó competitivo en casi todo el tablero.
La amplia distribución de algoritmos óptimos según las condiciones del dataset pone de relieve la existencia de ventajas relativas en todos ellos.

#let data = csv("data/mejor-clf-por-dataset-segun-r2-mediano.csv")
#let headers = data.at(0)
#let rows = data.slice(1, count: data.len() - 1)
#figure(
  table(columns: headers.len(), table.header(..headers), ..rows.flatten()),
  caption: flex-caption([TODO: copete largo tabla mejor clf por R²], [TODO: copete corto tabla mejor clf por R²]),
)

El mismo análisis con métrica de exactitud es, desde luego, menos favorable a nuestros métodos entrenados para otra cosa. #svc, entrenado a tono, resulta un algoritmo casi imbatible, con sólidos números en todo tipo de datasets y máximos en 6 datasets. #gbt vuelve a brillar en datasets con mucho ruido y siguen figurando como competitivos un amplio abanico de estimadores: hasta #fkdc retiene su título en 1 dataset, `espirales_lo`.

#let data = csv("data/mejor-clf-por-dataset-segun-accuracy-mediano.csv")
#let headers = data.at(0)
#let rows = data.slice(1, count: data.len() - 1)
#figure(
  table(columns: headers.len(), table.header(..headers), ..rows.flatten()),
  caption: flex-caption(
    [TODO: copete largo tabla mejor clf por exactitud],
    [TODO: copete corto tabla mejor clf por exactitud],
  ),
)


Solo considerar la _performance_ de #fkdc y #fkn en los 20 datasets daría unas 40 unidades de análisis, y en el espíritu de indagación curiosa que guía esta tesis, existen aún más tendencias y patrones interesantes en los 4,500 experimentos realizados. No es nuestra intención matar de aburrimiento al lector, con lo cual a continuación haremos un paneo arbitrario por algunos de los resultados que (a) nos resultaron más llamativos o (b) se acercan lo suficiente a alguno de la literatura previa como para merecer un comentario aparte.
== Lunas, círculos y espirales ($D=2, d=1, k=2$)

Para comenzar, consideramos el caso no trivial más sencillo con $D>d$: $D=2, d=1, k=2$, y exploramos tres curvas sampleadas con un poco de "ruido blanco" añadido: dos "lunas" --- semicírculos no superpuestos con sus centros en un extremo del semicírculo opuesto ---, dos círculos concéntricos y dos espirales con el mismo origen y rotación en sentidos opuestos #footnote[No entraremos en demasiado detalle sobre cómo se generó o de dónde se tomó cada _dataset_ para mantener el foco en los resultados de la experimentación. En el paquete adjunto, las rutinas completas para generar cada conjunto de datos se puede leer en `fkdc/datasets.py`].

#v(-1em)

#defn(
  "ruido blanco",
)[Sea $W = (W_1, dots, W_d) in RR^d$ una variable aleatoria tal que $"E"(W_i)=0, "Var"(W_i)=SS thick forall i in [d]$. Llamaremos "ruido blanco con escala $SS$" a $N$ realizaciones #iid de $W, thin bu(W) in RR^(N times d)$.] <ruido-blanco>

#obs[Dado que la dimensión de la variedad subyacente ($d=1$) es menor que la del espacio ambiente ($D=2$), sin ruido las observaciones caerían exactamente sobre la curva y la tarea de clasificación resultaría casi trivialmente sencilla. Para acercarnos a un escenario más realista que simule la incertidumbre inherente en cualquier toma de muestras, las observaciones se generan dentro de un _tubo_ de radio $tau$ alrededor de #MM, es decir, en el conjunto $B(MM, tau) = {x in RR^D : min_(y in MM) norm(x - y)_2 <= tau}$, tal como @mckenziePowerWeightedShortest2019 mencionan como posible extensión a su trabajo.]

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




En una primera variación con "bajo ruido" (y sufijada "`_lo`") #footnote[en inglés, _low_ y _high_ - baja y alta - son casi homófonos de _lo_ y _hi_], a las observaciones #XX sobre la variedad #MM se les añadió ruido blanco con un parámetro de escala $sigma$ según la distribución normal bivariada, $epsilon ~ cal(N)_2(0, sigma^2 bu(I))$. $sigma$ se ajustó a cada dataset para resultar "poco" relativo a la escala de los datos #footnote[La distribución normal multivariada no determina un radio finito para el tubo $B(MM, tau)$. En la práctica, con muestras relativamente pequeñas como las nuestras --- 400 observaciones por clase --- el tubo de diámetro $tau approx 6 sigma$ _no_ captura a todas la observaciones con probabilidad menor a uno en un millón.].
$ sigma_"lunas" = 0.25 quad sigma_"circulos" = 0.08 quad sigma_"espirales" = 0.1 $.

En los tres datasets, el resultado es muy similar: #fkdc es el estimador que mejor $R^2$ reporta, y en todos tiene una exactitud comparable a la del mejor para el dataset. En ninguno de los tres datasets #fkdc tiene una exactitud muy distinta a la de #kdc, pero saca ventaja en $R^2$ para `lunas_lo` y `espirales_lo`.

Entre el resto de los algoritmos, los no paramétricos son competitivos: #kn, #fkn y #gbt, mientras que #gnb, #slr, #logr rinden mal pues las _fronteras de decisión_ que pueden representar no cortan bien a los datos.


// Mapeo de nombres CSV a macros de clasificadores
#let clf_macros = (
  "fkdc": fkdc,
  "kdc": kdc,
  "fkn": fkn,
  "kn": kn,
  "gnb": gnb,
  "lr": logr,
  "slr": slr,
  "svc": svc,
  "gbt": gbt,
)

#let highlights_table(highlights) = {
  let csv_string = highlights.at("summary")
  let best_clf = highlights.at("best", default: none)
  let bad_clfs = highlights.at("bad", default: ())
  let lines = csv_string.split("\n").filter(l => l.len() > 0)
  let headers = lines.at(0).split(",")
  let rows = lines.slice(1)

  let best_fill = rgb("#7cff9dc9")
  let bad_alpha = 70%

  let cells = ()
  for row_str in rows {
    let fields = row_str.split(",")
    let clf_key = fields.at(0)
    let clf_label = clf_macros.at(clf_key, default: raw(clf_key))
    let is_best = clf_key == best_clf
    let is_bad = clf_key in bad_clfs

    for (i, field) in fields.enumerate() {
      let content = if i == 0 { clf_label } else if field == "" { align(center)[--] } else { field }
      if is_best {
        cells.push(table.cell(fill: best_fill)[#content])
      } else if is_bad {
        cells.push(table.cell()[#text(fill: black.transparentize(bad_alpha))[#content]])
      } else {
        cells.push([#content])
      }
    }
  }

  table(
    columns: headers.len(),
    stroke: none,
    align: (x, y) => if y == 0 { center } else if x == 0 { right } else { left },
    table.header(..headers.map(h => {
      let label = if h == "clf" { [clf] } else if h == "r2" { [$R^2$] } else if h == "accuracy" { [exac] } else { [#h] }
      text(weight: "bold", label)
    })),
    table.hline(stroke: 1pt),
    table.vline(x: 1, start: 1, stroke: .5pt),
    ..cells,
  )
}

#let highlights_figure(dataset, height: 8em, width: 140%) = {
  let highlights = json("data/" + dataset + "-r2-highlights.json")
  let tabla_resumen = highlights_table(highlights)

  figure(
    table(
      columns: 2,
      rows: 2,
      stroke: 0pt,
      image("img/" + dataset + "-scatter.svg"), text(size: 8pt)[#tabla_resumen],
      image("img/" + dataset + "-r2-boxplot.svg"), image("img/" + dataset + "-accuracy-boxplot.svg"),
    ),
    caption: flex-caption[_Scatterplot_, tabla resumen y _boxplots_ de $R^2$ y _accuracy_ en el _dataset_ #raw(dataset)][Resumen de resultados para #raw(dataset)],
  )
}

#let wide-figure(width: 140%, body, ..args) = figure(
  box(width: width, body),
  ..args,
)


#let euc = $norm(thin dot thin)_2$
#let sfd = $D_(Q, alpha)$

#obs("riesgos computacionales")[
  Una dificultad de entrenar un clasificador _original_ es que hay que definir las rutinas numéricas "a mano" #footnote[Usando librerías estándares como `numpy` y `scipy` para las operaciones elementales, pero nada más.]. Además, _debugear_ errores en rutinas numéricas es particularmente difícil, puesto que las operaciones no producen errores obvios, sino que casi siempre retornan, salvo que retornan valores irrisorios #footnote[Hubo montones de estos, cuya resolución progresiva dio lugar al módulo `fkdc/fermat.py` y las clases `SampleFermatDistance, FermatKNeighborsClassifier, FermatKDE` y `KDClassifier`--- que acepta tanto la métrica euclídea como de Fermat --- en la pequeña librería que acompaña esta tesis. Creemos que no los hay, pero todo error de cálculo que pueda persistir en el producto final depende exclusivamente de mí.].

  A ello se le suma que el cómputo de la distancia muestral de Fermat #sfd es realmente caro. Aun siguiendo "buenas prácticas computacionales" #footnote[Como sumar logaritmos en lugar de multiplicar valores "crudos" siempre que sea posible], implementaciones ingenuas pueden resultar impracticables hasta en datasets de baja cardinalidad y pocas dimensiones.

  Por otra parte, el teorema de convergencia @convergencia-sfd nos garantiza que cuando $n->oo, quad sfd -> cal(D)_(f, beta)$, pero esa es una afirmación asintótica y aquí estamos tomando $k=5$ pliegos de entre $n = 800$ observaciones, con $n_"train" = n_"eval" = n slash 2$ observaciones para un tamaño muestral efectivo de $(k-1)/k n/2 = 320$. ¿Es 320 un tamaño muestral "lo suficientemente grande" para que sea válida?

  Por todo ello, que la bondad de los clasificadores _no empeore_ con el uso de #sfd en lugar de #euc es de por sí un hito importante.
]

=== `lunas_lo`

#highlights_figure("lunas_lo")

#logr tiene un rendimiento decente en `lunas_lo`:

#figure(
  image("img/lunas_lo-lr-decision_boundary.svg", height: 20em),
  caption: flex-caption(
    [Frontera de decisión para #slr en `lunas_lo`, $s = #plotting_seed$],
    [Frontera de #slr en `lunas_lo`],
  ),
)
Nótese que la frontera _lineal_ entre clases (al centro de la banda gris) aprendida por #logr separa _bastante_ bien la muestra: pasa por el punto del segmento que une el "centro" de cada luna, y de todas las direcciones con tal origen, elige la que mejor separa las clases. _Grosso modo_, en el tercio de la muestra más cercano a la frontera, alcanza una exactitud de $~50%$, pero en los tercios al interior de cada región está virtualmente en 100%, que da un promedio global de $1/3 50% + 2/3 100% = 86.7%$, casi exactamente la exactitud observada.

=== `circulos_lo` y `espirales_lo`

#highlights_figure("circulos_lo")


#highlights_figure("espirales_lo")

Una inspección ocular a las fronteras de decisión revela las limitaciones de distintos algoritmos, siendo el caso de las espirales el más vistoso y pedagógico. #logr y #slr solo pueden dibujar fronteras "lineales", y como ninguna frontera lineal que corte la muestra logra dividirla en dos regiones con densidades de clase realmente diferentes, el algoritmo falla. #gnb falla de manera análoga, aunque su problema es otro - no lidia bien con distribuciones con densidades marginales muy similares.

#let clfs = ("kdc", "fkdc", "svc", "kn", "fkn", "gbt", "slr", "lr", "gnb")
#wide-figure(width: 160%,
  grid(columns: 3, gutter: 4pt, ..clfs.map(clf => image(
    "img/espirales_lo-" + clf + "-decision_boundary.svg",
  ))),
  caption: flex-caption(
    [Fronteras de decisión de los nueve algoritmos evaluados sobre `espirales_lo` con semilla $s=#plotting_seed$. Nótese la incapacidad de #logr, #slr y #gnb para separar las clases, la aproximación rectangular de #gbt, y la nitidez de las fronteras de #fkdc y #svc.],
    [Fronteras de decisión en `espirales_lo`],
  ),
) <fig-fronteras-espirales>

Entre #kn y #fkn casi no observamos diferencias, asunto en el que ahondaremos más adelante. Por lo pronto, sí se nota que se adaptan bastante bien a los datos, con algunas regiones "claras" de incertidumbre que resultan onerosas en términos de $R^2$: a primera vista los mapas de decisión recién expuestos se ven muy similares, pero las pequeñas diferencias de probabilidades resultaron en una diferencia de $0.19$ en $R^2$ _en contra_ del modelo más complejo para esta semilla #footnote[La diferencia en la _mediana_ de $R^2$ para ambos es mucho menor, $approx 0.03$, lo cual resalta la sensibilidad de los resultados a la semilla aleatorizante y la importancia de realizar muchas repeticiones de cada experimento para evitar resultados espurios]. También resulta llamativa la "creatividad" de #gbt para aproximar las verdaderas fronteras --- espirales curvas --- con una serie de preguntas binarias, que le permiten dibujar una especie "espirales rectangulares".

#kdc ofrece una frontera aún más regular que #kn, sin perder en $R^2$ y hasta mejorando la exactitud. Y por encima de esta ya destacable _performance_, el uso de la distancia de Fermat _incrementa_ la confianza en estas regiones --- nótese cómo se afinan las áreas grises y aumenta la superficie de rojo/azul sólido, mejorando otro poco el $R^2$.

#figure(
  columns(2)[
    #image("img/espirales_lo-fkdc-decision_boundary.svg")
    #colbreak()
    #image("img/espirales_lo-svc-decision_boundary.svg")
  ],
  caption: flex-caption(
    [Fronteras de decisión de #fkdc (izq.) y #svc (der.) en `espirales_lo`, $s = #plotting_seed$],
    [Fronteras de #fkdc y #svc en `espirales_lo`],
  ),
)

Por último, observamos las fronteras de #svc, que no tienen gradiente de color sino solo una frontera lineal #footnote[Como aprendimos: la frontera de una variedad riemanniana de dimensión intrínseca $d$ es una variedad sin frontera de dimensión intrínseca $d-1$; la frontera de estas regiones en es una curva parametrizable en $RR^1$ embebida en $RR^2$] puesto que al ser un clasificador duro determina una frontera abrupta donde cambia la clase predicha. Es sorprendente la flexibilidad del algoritmo, que consigue dibujar una única frontera sumamente no-lineal que separa los datos con altísima exactitud. La ventaja que #fkdc pareciera tener sobre #svc es que la frontera que dibuja pasa "más lejos" de las observaciones de clase, mientras que la #svc parece estar muy pegada a los brazos de la espiral, particularmente en el giro más interno.

=== Estudio de ablación: $R^2$ para #kdc/ #kn con y sin distancia de Fermat.

Según la #link("https://dle.rae.es/ablaci%C3%B3n")[RAE], "Del lat. tardío ablatio, -ōnis 'acción de quitar'." --- ¿qué se pierde en términos de $R^2$ al _no_ usar #sfd en estos algoritmos?. Sirvan para concentrar la atención en esta diferencia, los gráficos de dispersión del $R^2$ alcanzado en $XX_"test"$ para #kn y #kdc con y sin distancia de Fermat, en las #reps repeticiones de cada Tarea.

#let curvas = ("lunas", "circulos", "espirales")
#wide-figure(
  grid(
    columns: (auto, 1fr, 1fr),
    gutter: 4pt,
    align: horizon,
    // column headers
    [], align(center)[*#kdc vs. #fkdc*], align(center)[*#kn vs. #fkn*],
    // rows: one per curve
    ..curvas.map(c => (
      rotate(-90deg)[#raw(c + "_lo")],
      image("img/" + c + "_lo-kdc-fkdc-r2-scatter.svg"),
      image("img/" + c + "_lo-kn-fkn-r2-scatter.svg"),
    )).sum(),
  ),
  caption: flex-caption(
    [Gráficos de dispersión de $R^2$ para #kdc (izq.) y #kn (der.) con (eje $y$) y sin (eje $x$) distancia de Fermat.],
    [$R^2$ con y sin distancia de Fermat para #kdc y #kn],
  ),
) <fig-17>

Para #kn y #fkn, los resultados son casi exactamente iguales para todas las semillas en `lunas_lo` y `circulos_lo`; con ciertas semillas #fkn saca ventaja en `espirales_lo`, pero también tiene dos muy malos resultados con $R^2 approx 0$ que #kn evita.

Para #fkdc, pareciera evidenciarse alguna ventaja para varias semillas en `lunas_lo` y `espirales_lo`, menos así para `circulos_lo`.

Veamos primero qué sucede durante el entrenamiento para `circulos_lo`: ¿es que no hay ninguna ventaja en usar #sfd? Consideremos la _superficie de pérdida_ que resulta de graficar en 2D la pérdida $L$ usada _durante el entrenamiento_ para cada hiperparametrización considerada:

#obs(
  "unidades de la pérdida",
)[Si bien consideramos como _score_ (a más, mejor) $R^2$, el entrenamiento se realizó con `neg_log_loss` #footnote[
    N. del E.: A posteriori de la experimentación escubrimos que entre las numerosas funciones de puntaje --- _score_ --- que tolera `scikit-learn`, se incluye #link("https://scikit-learn.org/stable/modules/model_evaluation.html#d2-score-classification")[`d2_log_loss_score`], que es esencialmente el $R^2$ de McFadden que proponemos como métrica de evaluación. Sería ideal recomputar los experimentos entrenándolos con dicha función objetivo, pero no haby razones de peso apra suponer que el resultado sería demasiado distinto: al fin y al cabo, tanto la log-verosimilitud como el $R^2$ se maximizan en el mismo punto que la verosimilitud.], que aunque tiene la misma monotonicidad que $R^2$, está en otras unidades: entre $(-oo, 0]$.]

#figure(
  image("img/circulos_lo-8527-fkdc-bandwidth-alpha-loss_contour.svg"),
  caption: flex-caption(
    [Superficie de _score_: para cada valor de $alpha$ considerado, una cruz roja marca el valor de $h$ que maximizó el _score_.],
    [Superficie de _score_ en `circulos_lo`],
  ),
)
Nótese que la región amarilla, que representa los máximos puntajes durante el entrenamiento, se extiende diagonalmente a través de todos los valores de $alpha$. Es decir, no hay _un_ par de hiperparámetros óptimos $(alpha^star, h^star)$, sino que fijando $alpha$, siempre pareciera existir un $tilde(h)(alpha)$ que alcanza (o aproxima) la máxima exactitud _posible_ con el método en el dataset. En este ejemplo en particular, hasta pareciera ser que una relación log-lineal captura bastante bien el fenómeno, $tilde(h) prop log(alpha)$. En particular, entonces, $"exac"(tilde(h)(alpha), alpha) approx "exac"(h^star), alpha^star) thin forall alpha$, y se entiende que el algoritmo #fkdc no mejore significativamente la exactitud por sobre #kdc. Este resultado es consistente con el ya mencionado comentario de @bijralSemisupervisedLearningDensity2012[§5.1], que encuentran que fijar $p=2$ para la norma "de base" y $q=alpha=8$ "representa una elección razonable para la mayoría de los datasets".


Ahora bien, esto es solo en _un_ dataset, con _una_ semilla específica. ¿Se replicará el fenómeno en los otros datasets?

#let semillas = (7354, 8527, 1188)

#wide-figure(width: 150%,
  grid(
    columns: (auto, 1fr, 1fr, 1fr),
    gutter: 4pt,
    align: horizon,
    // column headers (seeds)
    [], ..semillas.map(s => align(center)[*s=#s*]),
    // rows: one per curve
    ..curvas.map(c => (
      rotate(-90deg)[#raw(c + "_lo")],
      ..semillas.map(s =>
        image("img/" + c + "_lo-" + str(s) + "-fkdc-bandwidth-alpha-loss_contour.svg")
      ),
    )).sum(),
  ),
  caption: flex-caption(
    [Superficies de pérdida para tres semillas $s in #semillas$ y cada uno de los tres datasets. El patrón log-lineal previamente observado se replica casi perfectamente en todos los casos.],
    [Superficies de pérdida para `[lunas|circulos|espirales]_lo`],
  ),
) <fig-19>

¡Pues sí replica! Podemos observar también en datasets como `circulos_lo`, $s =7354$, cómo actúa la regla de parsimonia. Dentro de la "meseta color lima" que ocupa toda el área por encima de la diagonal principal del gráfico,  todas las hiperparametrizaciones alcanzan resultados similares. Sin embargo, la validación cruzada elige consistentemente para cada $h$ el menor $alpha$ posible que no "cae" hacia la región azul de menores _scores_.

Estamos ahora frente a una contradicción: en la @fig-17 vimos que por ejemplo, para `lunas_lo`, #fkdc alcanzaba un $R^2$ consistentemente mejor que #kdc; mientras que de los paneles superiores de la @fig-19 observamos que los score que se alcanzan limitándonos a $alpha = 1$ son tan altos como los de $alpha > 1$. Es cierto que los resultados de @fig-17 son a través de _todas_ las semillas, y en el conjunto de evaluación, mientras que en la @fig-19 observamos _algunas_ semillas y sobre los datos de entrenamiento, pero la pregunta es válida: ¿de dónde proviene la ventaja de #fkdc en estos datasets?

==== Hiperparámetros óptimos en `lunas_lo` para #kdc, #fkdc

Hacemos entonces una comprobación fundamental: ¿qué parametrizaciones están siendo elegidas en el esquema de validación cruzada con regla de parsimonia? Hete aquí el detalle para las #reps repeticiones de `lunas_lo`:

#tabla_csv(
  "data/lunas_lo-best_params.csv",
  caption: [Hiperparámetros seleccionados por CV con regla de parsimonia para #kdc y #fkdc en `lunas_lo`, por semilla.],
  short-caption: [Hiperparámetros seleccionados por R1SD de #kdc y #fkdc en `lunas_lo`],
)

Durante el entrenamiento, a veces el mejor se obtiene con _otros_ valores de $alpha$, pero la mejora no es lo suficientemente grande para descartar alguna hiperparametrización con $alpha = 1$ bajo la regla de $1 sigma$  descrita en @r1sd.

// TODO: Recortar a 4 decimales, simplificar a solo $alpha$
#tabla_csv(
  "data/lunas_lo-best_test_params.csv",
  caption: [Hiperparámetros minimizadores de pérdida en enrenamiento para #kdc y #fkdc en `lunas_lo`, por semilla.],
  short-caption: [Hiperparámetros minimizadores de pérdida de #kdc y #fkdc en `lunas_lo`],
)
Resulta ser que
- al entrenar #fkdc se está eligiendo $alpha=1$ para _todas_ las semillas, y
- el ancho de banda seleccionado es ligera pero consistentemente _menor_ que el que toma #kdc.

Veamos cómo se comparan los valores de $R^2$ que alcanza cada algoritmo en cada semilla:
#figure(
  columns(2)[
    #image("img/lunas_lo-[f]kdc-score-vs-bandwidth.svg")
    #colbreak()
    #image("img/lunas_lo-[f]kdc-delta_r2-vs-delta_h.svg")],
  caption: flex-caption(
    [(izq.) Dispersión de $R^2$ en función de $h$ por clasificador y semilla en `lunas_lo`, para #fkdc, #kdc;
      (der.) dispersión de $Delta_(R^2) = R^2_#kdc - R^2_#fkdc$ en función de $Delta_h = h^star_#fkdc - h^star_#kdc$ para cada semilla.],
    [$R^2$ vs. $h$ y $Delta_(R^2)$ vs. $Delta_h$ en `lunas_lo`],
  ),
)
En el panel izquierdo se observa una clara tendencia a mejorar ligeramente el $R^2$ a medida que disminuye el ancho de la ventana $h$ (en el rango en cuestión). En el panel derecho, para confirmar que la tendencia sucede _en cada repetición del experimento_, comparamos no los valores absolutos sino las diferencias relativas en $R^2, h$ entre #fkdc y #kdc apareando los resultados _para cada semilla_, y vemos que a mayor diferencia en el $h$ de #kdc por sobre #fkdc, peor es la caída en $R^2$.

Cabe aquí una crítica al diseño experimental: si #fkdc está tomando siempre $alpha =1$, por qué #kdc no puede elegir el mismo $h$ que #fkdc y así equiparar su rendimiento? ¿Se exploró una grilla de hiperparámetros a propósito desfavorable para #kdc? Pues no, todo lo contrario #footnote[La definición exacta está en `fkdc/config.py`, y es `np.logspace(-5, 6, 45)` para #fkdc y `np.logspace(-5, 6, 136)` para #kdc]: las grillas de $h$ para #kdc y #fkdc
- cubren de manera "logarítmicamente equidistante" el mismo rango de $h: [10^(-5), 10^6]$ y
- la grilla de #kdc cuenta con $approx$ el triple de puntos de #fkdc ($136 "vs." 45$).

Como en el entrenamiento de #fkdc se gastaron 13 veces más recursos evaluando 13 valores distintos de $alpha in {1 + 0.25 i, thick i in [13]} subset [1, 4]$, consideramos oportuno permitirle a #kdc explorar más valores de $h$, y la cantidad se eligió para que la grilla de #kdc coincida en lo posible con la de #fkdc, y tenga además otros dos valores "entre medio" de dos valores cualesquiera de la grilla de #fkdc #footnote[
  N. del E.: Para hace esto correctamente, deberíamos haber tomado $(45 - 1) times (2 + 1) + 1= 133$ elementos en la segunda grilla, pero olvidamos restar 1 a 45 --- hay 45 puntos pero 44 "espacios" entre puntos de la grilla --- y por eso obtuvimos 136 puntos, con lo cual las grillas difieren ligeramente y una no es un subconjunto de la otra. De todas maneras, la grilla de #kdc contiene el $0.173$, mucho más cercano al $0.178$ óptimo de #fkdc, con lo cual no se termina de explicar que la elección "modal" de #kdc haya sido $0.251$
].
En efecto, en el rango de interés, las grillas contaban con los valores:
$
  #fkdc: & [0.1, 0.178, 0.316, 0.562] \
   #kdc: & [0.119, 0.143, 0.173, 0.208, 0.251, 0.303, 0.366, 0.441, 0.532] \
$
con lo cual #kdc _podría_ haber encontrado el ligeramente más conveniente $h^star approx 0.17$, pero la convalidación cruzada se inclinó por valores concentrados en el rango $[0.25, 0.3]$. De repetir el experimento tomando una grilla más fina en este rango crucial, es posible que $Delta_h^star approx 0$ y por ende $Delta_(R^2)$ también, aunque por el mismo argumento de tomar una grilla más fina para $alpha approx 1$ terminaríamos tal vez encontrando un $alpha^star > 1$ para #fkdc #footnote[Hete aquí la dificultad de enunciar propiedades generales a partir de experimentos particulares: siempre hay _un experimento más_ para hacer, pero lamentablemente, en algún momento había que culminar la etapa experimental.]. En cualquier caso, hemos de aceptar que la ventaja de #fkdc en `lunas_lo` y `espirales_lo` sobre #kdc _no_ se debe a la inclusión del hiperparámetro $alpha$, sino a una validación cruzada aleatoriamente favorable.

=== Efectos de aumentar el ruido

Consideremos ahora los mismos datasets que hasta ahora, pero sampleando las observaciones sobre la variedad con "más ruido"; i.e., aumentando el valor de $sigma$ en el ruido blanco (@ruido-blanco) que le agregamos a los $X in MM$ según

$ sigma_"lunas" = 0.5 quad sigma_"circulos" = 0.2 quad sigma_"espirales" = 0.2 quad. $

#wide-figure(
  grid(columns: 3, gutter: 4pt,
    image("img/lunas_hi-scatter.svg"),
    image("img/circulos_hi-scatter.svg"),
    image("img/espirales_hi-scatter.svg"),
  ),
  caption: flex-caption["Lunas", "Círculos" y "Espirales" con "alto ruido"][ "Lunas", "Círculos" y "Espirales", alto ruido ],
) <fig-22>

En general, #fkdc y #fkn siguen siendo competitivos, pero el "terreno de juego" se ha nivelado considerablemente, y las ventajas antes vistas disminuyen.

- En `lunas_hi` observamos que #gbt alcanza un $R^2$ marginalmente mejor que el #fkdc pero no por mucho, y todos los métodos basados en densidad por núcleos (#fkdc, #kdc, #fkn, #kn) alcanzan una exactitud ligeramente mejor que la de #gbt.
- En `circulos_hi` #gnb  supera ligeramente pero significativamente en $R^2$ y exactitud, aunque aún su propia performance no es muy  alentadora con $R^2_#gbt approx 0.09$.

- En `espirales_hi` todos los métodos basados en densidad por núcleos (#fkdc, #kdc, #fkn, #kn) alcanzan un $R^2$ muy similar, #gbt queda largamente atrás y #gnb, #logr y #slr no no se distinguen del $0$. #svc obtiene la mejor exactitud, pero no supera a #fkdc por mucho. Las ventajas de #fkdc por sobre #kdc son casi nulas en este contexto.

==== `lunas_hi`

#highlights_figure("lunas_hi")

==== `circulos_hi`
#highlights_figure("circulos_hi")

==== `espirales_hi`
#highlights_figure("espirales_hi")



El aumento en la cantidad de ruido hace la tarea más difícil para _todos_ los estimadores, pero los métodos basados en densidad por núcleos parecen sufrirlo particularmente, aunque solo sea porque "caen desde más alto", a un nivel de _performance_ similar al de otros métodos.

#wide-figure(
  grid(
    columns: 3, gutter: 4pt,
    image("img/lunas-caida_r2.svg"),
    image("img/circulos-caida_r2.svg"),
    image("img/espirales-caida_r2.svg"),
  ),
  caption: flex-caption(
    [$R^2$ mediano por clasificador y dataset, comparado entre la variante con bajo (`_lo`) y alto (`_hi`) ruido en el sampleo. Se excluyen clasificadores con $R^2 approx 0$ en ambas variantes.],
    [Caída de $R^2$ mediano al aumentar el ruido],
  ),
)


Por último, veamos las fronteras de decisión de  #fkdc y los más competitivos en términos de $R^2$ (#gbt) y exactitud (#svc):
#{
  let hi_clfs = (("fkdc", fkdc), ("gbt", gbt), ("svc", svc))
  let hi_datasets = ("lunas_hi", "circulos_hi", "espirales_hi")
  wide-figure(width: 160%,
    grid(
      columns: (auto, 1fr, 1fr, 1fr),
      gutter: 4pt,
      align: horizon,
      [], ..hi_datasets.map(d => align(center)[*#raw(d)*]),
      ..hi_clfs.map(((key, label)) => (
        rotate(-90deg)[#label],
        ..hi_datasets.map(d =>
          image("img/" + d + "-" + key + "-decision_boundary.svg")
        ),
      )).sum(),
    ),
    caption: flex-caption(
      [Fronteras de decisión para #fkdc, #gbt, #svc en regímenes de alto ruido, $s = #plotting_seed$. El $R^2$ de algunos pares `clasificador, dataset` no se logra distinguir del cero.],
      [Fronteras de decisión en alto ruido],
    ),
  )
}

Al ojo humano, queda claro que las fronteras y regiones de confianza que "dibuja" #fkdc se alinean "en espíritu" con la forma de las variedades que buscamos descubrir: la "región de indiferencia" gris en `lunas_hi` es una especie de curva casi-cúbica que efectivamente separa las lunas, el "huevo frito" de `circulos_hi` efectivamente tiene máxima confianza a favor de la clase interna en el centro de ambos círculos (y se va deformando progresivamente a medida que nos alejamos de él), y en `espirales_hi` casi logra dibujar la espiral. Sin embargo, esta deseable propiedad no es fácilmente reducible a una métrica en $RR$, y se desdibuja en las comparaciones puramente numéricas.

== Pionono, Eslabones, Hélices y Hueveras ($D=3$)

Consideraremos a continuación datasets sintéticos embebidos en 3 dimensiones ($D = 3$), con variedades de dimensión intrínseca  $1$ (`eslabones, helices`) y $2$ (`pionono, hueveras`).

=== Eslabones
#figure(
  image("img/eslabones-scatter-3d.svg"),
  caption: flex-caption([TODO: copete largo eslabones scatter 3D], [TODO: copete corto eslabones scatter 3D]),
)

// TODO: poner scatter 3D en highlight por dataset para $D=3$
#highlights_figure("eslabones_0")

Toda la familia de estimadores de densidad por núcleos alcanza un $R^2 approx 1$, y aun Naive Bayes tiene una _performance_ aceptable: con este nivel de ruido blanco en el sampleo, el "margen de separación" entre ambos anillos es tan amplio que la tarea resulta trivial.

Un punto en contra de #fkdc aquí es que el _boxplot_ de $R^2$ - no así el de exactitud - revela un fuerte outlier de $R^2_#fkdc 0.65, thin R^2_#kdc = 0.9$ para la semilla $2411$.

#tabla_csv(
  "data/eslabones_0-params-2411.csv",
  caption: [Parametrización de #fkdc para `eslabones_0`, $s=2411$.],
  short-caption: [Parámetros de #fkdc en `eslabones_0`, $s=2411$],
)

La semilla resultó adversa para ambos #footnote[Aún $R^2_#kdc = 0.91$ es un mal score _relativo a_ el IQR de rendimiento de #kdc en el dataset], pero particularmente para #fkdc. #gbt queda técnicamente a más de $1 sigma$ del $R^2$ medio de #fkn, pero en la práctica, ofrece un $R^2$  excelente _sin ningún outlier_.

==== Hélices

Este dataset consiste en dos hélices del mismo diámetro y "enroscadas" en la misma dirección, una de ellas empezando a "media altura" entre dos brazos consecutivos de la otra. El dataset es particularmente desafiante para Naive Bayes y regresión logística, que no logran diferenciarse en nada de un clasificador trivial que prediga siempre la misma clase.
#highlights_figure("helices_0")
#obs[La performance de #logr es mala únicamente porque se aplicó ciegamente a los datos. La primer tarea cuando se busca inferir la geometría de unos datos es graficarlos, y al observar la hélice uno puede parametrizarla de manera natural como $f(x, y, z) = ("ángulo, velocidad radial, velocidad vertical") ,$ entrenar sobre esta _representación_ y obtener un $R^2 approx 1$.
  Al final, "todo algoritmo funciona cuando los datos son buenos" --- la ventaja de algunos es que no hace falta ponerle demasaida cabeza a "masajearlos" hasta que "son buenos". Que a #gnb le resulte complejo no es sorprendente, ya que las distribuciones marginales son prácticamente idénticas.
  #figure(
    image("img/helices-pairplot.svg", width: 16em),
    caption: flex-caption([TODO: copete largo hélices pairplot], [TODO: copete corto hélices pairplot]),
  )
]
La clasificación dura con estimación de densidad por núcleos --- con distancia de Fermat o sin ella --- resulta ser superior a todas las alternativas en términos de exactitud --- ligeramente --- y $R^2$ --- por mucho. Encima de ello, #fkdc es todavía significativamente mejor en $R^2$ que #kdc por casi 5 puntos porcentualessalvo y consistentemente entoda las semillas salvo una particularmente negativa:

#figure(
  image("img/helices_0-r2-fkdc-vs-kdc.svg", height: 16em),
  caption: flex-caption([TODO: copete largo hélices R² fkdc vs kdc], [TODO: copete corto hélices R² fkdc vs kdc]),
)

En prácticamente todas las semillas el $R^2$ de #fkdc es estrictamente mejor al "control" de #kdc. ¿Con qué parámetros sucede?

#tabla_csv(
  "data/helices_0-parametros_comparados-kdc.csv",
  caption: [Parámetros comparados de #fkdc vs. #kdc en `helices_0`, ordenados por $Delta_(R^2)$.],
  short-caption: [Parámetros de #fkdc vs. #kdc en `helices_0`],
)

Ordenados por $Delta_(R^2) = R^2_#fkdc - R^2_#kdc$, la semilla con mayor diferencia a favor del resultado con distancia de Fermat corresponde a $mu = (alpha = 1.25, h = 0.006)$ --- una hiperparametrización no-trivialmente reducible a #kdc --- que resulta en un $Delta_(R^2) = 0.237 (= 0.953 - 0.716)$ puntos _en términos absolutos_#footnote[I.e., "un montón".] por encima de #kdc con $h = 0.208$, usando una ventana unas 35 veces más ancha.
Salta a la vista también que tales parametrizaciones tienen muy variado rendimiento por fuera del conjunto de entrenamiento #footnote[también OOS --- _out-of-sample_--- por sus siglas en inglés.] , pues para $s = 8096$ se eligió _la misma_ $mu$, contra $h_#kdc = 0.143 approx 25 h_#fkdc$ y se dio la segunda diferencia más amplia _en contra_ de #fkdc ($Delta_(R^2) = -0.098$).

Se podría argumentar en contra de #fkdc que $alpha = 1.25 approx 1$, pero al revisar el comportamiento de la regla de parsimonia,  encontramos por ejemplo que para $s = 1188, thin Delta_(R^2) = 0.227$  la parametrización óptima fue con $nu = h = 10^(-3), alpha=3$ y todas las hiperparametrizaciones  a menos de $1 sigma$ de $nu$ tenían $alpha >= 2.5$, lejos de 1.

Más aún, en unos cuantos casos --- $s in {1188, 1182, 2411}$ --- en que $alpha_#fkdc = alpha_#kdc = 1$, #fkdc todavía rinde un poco mejor que #kdc al elegir anchos de banda mucho más pequeños. Ya hemos visto que aún ligeras diferencias en la ventana $h$ podían llevar a mejoras en $R^2$ a favor de #fkdc por el detalle fino de la búsqueda en grilla que se definió. Por ejemplo,

$ s = 1182, quad Delta_R^2=0.111, quad alpha_#fkdc = alpha_#kdc = 1, quad h_#fkdc / h_#kdc approx 14.3 $

que cuesta explicar como una ligera discrepancia en la grilla de $h$.


#figure(
  columns(2, gutter: .5em)[
    #image("img/helices_0-1188-fkdc-bandwidth-alpha-loss_contour.svg")
    #colbreak()
    #image("img/r1sd+alpha.svg")
  ],
  caption: flex-caption(
    [Superficie de pérdida de #fkdc en `helices_0`.
      (izq., $s=1188$) Nótese la mínima "isla" alrededor de $h approx 10^(-3), alpha = 3$.(der., $s=1182$). #kdc encuentra (1) al entrenar, #fkdc se sale de $alpha=1$ y encuentra (2). La regla de parsimonia encuentra (3), de vuelta con $alpha = 1$.],
    [Superficies de pérdida para #fkdc en `helices_0`],
  ),
) <alpha-ne-1>

Nuestra hipótesis es que el dominio ampliado de hiperparámetros de #fkdc junto con la regla de parsimonia trabajan en tándem:

Durante el entrenamiento, #kdc encuentra la solución $h_#kdc=0.143$ (cf. posición $(1)$ de @alpha-ne-1, der.) ceñido al perfil en que $alpha = 1$ --- el borde inferior de la superficie. Presumiblemente, la varianza de la performance en testeo para dicha solución fue tal que ningún punto en el entorno de $h_#fkdc=0.01$ (cf. pos. $(3)$) estaba a menos de $1 sigma$ del _score_ en $(1)$. Cuando entrenamos #fkdc y ampliamos el dominio de la parametrización a toda la superficie computada, el entrenamiento por CV alcanza un máximo en $alpha=3, h = 0,000562$ (cf. $(2)$). Esta nueva solución tiene más varianza en sus resultados a través de cada pliego de CV, por lo que $hat(s)(mu_#fkdc) > hat(s)(mu_#kdc)$ y, la cota inferior de la R1SD de @r1sd será más laxa. En ese rango ampliado de hiperparametrizaciones "suficientemente buenas" se encuentra $alpha=1, h=0.01$ --- la solución de $(3)$ que en entrenamiento #kdc vio pero ignoró.

=== Efecto de #sfd en las vecindades óptimas de #kn

En el estimador de densidad en variedades de  @loubesKernelbasedClassifierRiemannian2008, al núcleo $K$ se lo evalúa sobre
$frac(d(x_0, X_i), h, style: "horizontal")$, y nuestra implementación de #fkdc estima $hat(d) = D_(Q_i, alpha)(XX)$. Si resultase que $D_(Q_i, alpha) prop ||dot||$ --- la distancia de Fermat es proporcional a la euclídea) --- podemos escribir

$
  (op(D_(Q_i, alpha))(x_0, X_i))/ h approx (c norm(x_0 - X_i))/ h = norm(x_0 - X_i) / h'
$
con $h' = h slash c$ y efectivamente los parámetros $(alpha, h)$ se solapan en sus funciones. Lamentablemente, sabemos que localmente esto _es_ cierto: en un vecindario de $x_0$, la densidad $f$ de $X$ es #math.approx constante, así que la distancia "macroscópica" $cal(D)_(f,beta)$, y el costo de integrarla al respecto de un sendero será proporcional a la longitud --- euclídea --- del mismo. Nuestra aproximación #sfd de $cal(D)_(f,beta)$ heredará las mismas características.

// TODO: averiguar de dónde sale este hecho sobre k_n opt, optima en qué sentido
La serie $k_n$ que minimiza el error cuadrátrico medio cuando $n -> oo$ es $k prop n^(d/(d+4))$, que para nuestros tamaños muestrales de CV resulta en $320^(3/(3+4)) =320^(3/7) approx 12$. Es decir que tomando #math.approx decena de vecinos, alcanzaría para entrenar un clasificado #kn decente --- #fkn podrá ser mejor con $k_#fkn >> 12$, pero no mejor que #kn con $k_#kn approx 12$.  le basta con conseguir un buen resultados en el primer  Pues bien, cuando miramos el mejor rendimiento en test por `n_neighbors` para #kn y #fkn, vemos que elegir un $alpha$ variable le permite a #fkn mantener una óptima performance en términos de log-verosimilitud #footnote[y por ende $R^2$ también]. para _cualquier_ valor de $k$ #footnote[`n_neighbors` en la parametrización de `scikit-learn`.]

// TODO: esto es realmene el _mean_ test score? podemos hacer el mismo gráfico con una linea por semilla y clf, todas con menor alpha, para ver si la rutas de fkn son siempre mejores que las de kn?
#figure(
  image("img/helices_0-fkn_kn-mean_test_score.svg", height: 12em),
  caption: flex-caption(
    [TODO: Log-verosimilitud máxima durante el entrenamiento de lso clasificadores de veciones más cercanos sobre `helices_0`],
    [$cal(l)$ en entrenamiento para #fkn y #kn en `helices_0`],
  ),
)

Llegamos a la misma conclusión que antes por otra dirección: si el espacio están tan biem sampleado que el clasificador depende de vecindarios muy pequeños #footnote[vía $k$ en $k$-vecinos-más-cercanos, $h$ en KDE] para estimar densidades el efecto de reemplazar la distancia euclídea por la distancia de Fermat aprendida de los datos no será muy notorio.

=== Pionono

#figure(
  image("img/pionono-scatter-3d.svg"),
  caption: flex-caption([TODO: copete largo pionono scatter 3D], [TODO: copete corto pionono scatter 3D]),
)
#highlights_figure("pionono_0")

Este dataset "clásico" para testear algoritmos de _clustering_ no-lineales fue analizado ya con #sfd en @sapienzaWeightedGeodesicDistance2018, así que decidimos incluirlo en la serie experimental. El trabajo citado tiene otro objetivo ---  _clustering_ basado en el algoritmo $k-$medoides --- y provee un gráfico de exactitud que compara con la obtenida por Isomap. Los autores encuentran que
#quote[existe un amplio rango de $d$ #footnote[$alpha$ en nuestra notación] para los que la $d-$distancia se porta significativamente mejor que Isomap. [...] para la exactitud esta región está limitada a $1.7 <= d <= 2.2$
]

Con un objetivo distinto --- _clasificación_ con DBDs --- no encontramos diferencia significativa entre #kdc y #fkdc, que a su vez rinden tan bien como el estado-del-arte en exactitud (#svc) y $R^2$ (#gbt).

// TODO: solamente dan igual las performances de fkdc/kdc, o ademas fkdc tambien elige alpha = 1 casi siempre?

=== Hueveras ($D=3, d=2, k=2$)

Este dataset sintético consiste de dos clases con idénticas distribuciones pero signo opuesto en la dirección de la coordenada vertical $ z = plus.minus(sin(x) times sin(y)) $ y se puede imaginar aproximadamente bien como cartones de maple de huevos, uno invertido respecto al otro, intentando ocupar el mismo lugar en el espacio.

#highlights_figure("hueveras_0")

// TODO: Mencionar la familia cal(K) antes y acá simplemente decir $cal(K)$
La exactitud de la familia de estimadores basados en densidad por núcleos --- $cal(K)={#fkdc, #kdc, #fkn, #kn}$ --- es competitiva contra la de #svc, que parece ser ligera y significativamente mejor. En términos de $R^2$, la familia $cal(K)$ es la única en alcanzar valores no-nulos, y #sfd parece resultar en mejoras significativas al menos para #fkn.


En efecto, observando los parámetros comparados de #fkdc v. #kdc, se repite que la hiperparametrización $(alpha_"opt", h_"opt")$ que maximiza $R^2$ en entrenamientotiene tiene $alpha > 1$, pero existe otra  $(alpha_(1 sigma), h_(1_sigma))$ con $alpha_(1 sigma) =1$   y $h_(1_sigma)$ "significativamente distinto" a $h_"opt"$ que cumple la regla de parsimonia.  Las tres semillas en la que #fkdc saca más ventaja sobre #kdc tiene por óptimos
// TODO: reemplazar tabla completa por únicamente las tres primeras columnas de la tabla: en cada columna que se repita tres veces el mismo valo, mostrarlo una sola vez en la fila del medio: delta_r^2, alhpa_fkdc, h_fkdc, h_kdc. no mostrarla columna de score_alpha_test
#tabla_csv(
  "data/hueveras_0-parametros_comparados-kdc.csv",
  caption: [Parámetros comparados de #fkdc vs. #kdc en `hueveras_0`, ordenados por $Delta_(R^2)$.],
  short-caption: [Parámetros de #fkdc vs. #kdc en `hueveras_0`],
)

En #fkn, la distancia de Fermat parece ofrecer una diferencia significativa en $R^2$ sobre #kn, con varias repeticiones del experimento donde aún con regla de parsimonia, #fkn y #kn eligen _la misma cantidad_ de vecinos pero $alpha_#fkn > 1$:

// TODO: filtrar lista a únicamente los casos con k_fkn = k_kn, dejar delta_r2, k=k_fkn=k_kn, alpha_fkn.
#tabla_csv(
  "data/hueveras_0-parametros_comparados-kn.csv",
  caption: [Parámetros comparados de #fkn vs. #kn en `hueveras_0`, con idéntico $k = k_#fkn = k_#kn$. Nótese que $alpha_#fkn > 1$ y $Delta_(R^2) > 0$ en casi todas, indicando una ganancia neta de usar #sfd.],
  short-caption: [$Delta_R^2= R^2_#fkn - R^2_#kn$ en `hueveras_0` con $k_#fkn = k_#kn, alpha_#fkn > 1$],
)


== Otros datasets

Presentamos aquí los resultados para diez datasets adicionales, agrupados en tres categorías: versiones en 15 dimensiones de los datasets 3D (con 12 dimensiones de ruido gaussiano añadidas), datasets multiclase de uso común en la literatura, y datasets de alta dimensionalidad.

=== Datasets 3D con ruido añadido (15D)

Los cuatro datasets tridimensionales del cuerpo principal (pionono, eslabones, hélices, hueveras) se ampliaron con 12 dimensiones de ruido gaussiano para evaluar la robustez de los clasificadores ante dimensiones irrelevantes.

#figure(
  image("img/anexo-15d-delta-r2.svg"),
  caption: flex-caption(
    [Caída de $R^2$ (mediana) al agregar 12 dimensiones de ruido a los datasets 3D. Valores negativos indican peor $R^2$ en 15D.],
    [Caída de $R^2$: 3D vs. 15D],
  ),
)

=== `pionono_12`

#highlights_figure("pionono_12")

En la versión 15D del pionono, #gbt domina con claridad ($R^2 approx 0.79$), seguido de lejos por #gnb ($R^2 approx 0.54$). Los clasificadores de densidad (#fkdc, #kdc) y los de vecinos (#fkn, #kn) colapsan a valores de $R^2 approx 0.10$, prácticamente indistinguibles entre variantes Fermat y euclídea. Los métodos lineales (#slr, #logr) se ubican en un rango intermedio ($R^2 approx 0.43$). Las 12 dimensiones de ruido degradan severamente a los estimadores por densidad de núcleos, cuya localidad los hace particularmente vulnerables a la maldición de la dimensionalidad.

=== `eslabones_12`

#highlights_figure("eslabones_12")

Nuevamente #gbt es el mejor clasificador ($R^2 approx 0.92$), seguido por #gnb ($R^2 approx 0.75$). Los clasificadores de densidad y vecinos se agrupan alrededor de $R^2 approx 0.24$, sin diferencia significativa entre variantes Fermat y euclídeas. La estructura de los eslabones --- dos anillos entrelazados --- se vuelve difícil de capturar por métodos locales cuando se añade ruido en 12 dimensiones adicionales.

=== `helices_12`

#highlights_figure("helices_12")

=== `hueveras_12`

#highlights_figure("hueveras_12")

Estos dos son de los datasets más difíciles del conjunto: la exactitud máxima de _cualquier_ clasificador apenas supera el 50%, valores muy cercanos al azar. Los clasificadores de densidad (#fkdc, #kdc) obtienen $R^2$ negativo, lo que indica un desempeño peor que el de un clasificador trivial. Estas geometrías, ya de por sí adversas aun sin ruido añadido, se vuelven intratables con este nivel de ruido en alta dimensión.

El fenómeno de las dimensiones de ruido sin correlación es particularmente pernicioso para los algoritmos basados en densidad por núcleos, aun con distancias basadas en densidad. Como la distancia de Fermat está computada como una geodésica en un grafo completo, y los pesos de cada arista están basados en distancia euclídea, las dimensiones de ruido puro "alejan" puntos cercanos entre sí en las dimensiones que importan. La ventaja de #gbt en _algunos_ de estos datasets del régimen de alto ruido es que al proceder con preguntas binarias sobre _un predictor a la vez_, puede identificar más fácilmente que cualquier pregunta sobre las columnas de ruido puro nunca sirve para partir la muestra en dos grupos con densidades bien distintas, y por eso las ignora.

=== Datasets multiclase

Se evaluaron cuatro datasets de clasificación con tres o más clases, provenientes de repositorios clásicos de _machine learning_ o generados sintéticamente.

#figure(
  image("img/anexo-multik-fkdc-vs-kdc.svg"),
  caption: flex-caption(
    [$R^2$ por semilla de #fkdc vs. #kdc en los 4 datasets multiclase reales.],
    [$R^2$ de #fkdc vs. #kdc en datasets multiclase],
  ),
)

=== `iris` ($k = 3$, $D = 4$)

#highlights_figure("iris")

Este dataset es bien servido por métodos lineales: #logr ($R^2 approx 0.89$) tiene un desempeño tan bueno como el mejor de los métodos basados en densidad, #fkn. #fkdc tiene un $R^2$ ligeramente peor ($approx 0.86$) pero competitivo. Es un dataset donde los métodos simples bastan; las clases son casi linealmente separables en el espacio original y hasta #gnb alcanza una exactitud del $91%$ y $R^2 approx 0.81$.

=== `pinguinos` ($k = 3$, $D = 4$)

#highlights_figure("pinguinos")

El dataset de pingüinos de Palmer es casi linealmente separable: #slr ($R^2 approx 0.96$) y #logr ($R^2 approx 0.96$) dominan. Los clasificadores de densidad y vecinos se ubican en un rango intermedio, muy por debajo de los métodos lineales. La diferencia entre #fkdc y #kdc es despreciable ($approx 1$ centésimo de $R^2$).

Al investigar en mayor detalle qué es lo que sucede, la matriz de confusión muestra que #fkdc no predice la clase Chinstrap en absoluto: las 34 observaciones de Chinstrap en el conjunto de evaluación se clasifican como Adelie.

#figure(
  image("img/pinguinos-fkdc-confusion_matrix.svg", width: 60%),
  caption: flex-caption(
    [Matriz de confusión de #fkdc en `pinguinos`. La clase Chinstrap se confunde enteramente con Adelie.],
    [Matriz de confusión de #fkdc en `pinguinos`],
  ),
)

Si miramos el _pairplot_ con los gráficos de dispersión dimensión a dimensión, se observa que las distribuciones marginales de Adelie y Chinstrap se solapan considerablemente en las cuatro dimensiones del dataset.

#figure(
  image("img/pinguinos-pairplot.svg", width: 80%),
  caption: flex-caption(
    [_Pairplot_ del dataset `pinguinos`. Las distribuciones marginales de Adelie y Chinstrap se solapan en todas las dimensiones, dificultando la clasificación por métodos basados en densidad.],
    [_Pairplot_ de `pinguinos`],
  ),
)

Si entrenamos nuevamente #fkdc quitando una de las dimensiones altamente correlacionadas, la _performance_ mejora y si bien no supera a la de los mejores métodos, se vuelve comparativamente razonable. Es una victoria pírrica, que nos recuerda la importancia de realizar inspecciones visuales de los _datasets_ antes de largarse a entrenar algoritmos en tareas del mundo real.

=== `vino` ($k = 3$, $D = 11$)

#highlights_figure("vino")

El dataset de vinos ($k = 3$, $D = 11$) favorece a #gbt ($R^2 approx 0.90$) y #slr ($R^2 approx 0.90$), los métodos paramétrico y no-paramétrico por excelencia en este conjunto. Hay una brecha notable entre #logr ($R^2 approx 0.69$) y #slr ($R^2 approx 0.90$), lo que sugiere que el escalado de _features_ es importante en este espacio de 11 dimensiones. #fkdc ($R^2 approx 0.43$) aventaja modestamente a #kdc ($R^2 approx 0.40$) pero ambos quedan lejos de los líderes. Se intentó el mismo acercamiento que en `pinguinos`, quitando dimensiones altamente correlacionadas para ver si el rendimiento mejoraba, pero esta vez la misma técnica no ayudó. Este es probablemente el peor dataset para #fkdc y #fkn. Sería de sumo interés investigar por qué en detalle.

=== `anteojos` ($k = 3$, $D = 2$)

#highlights_figure("anteojos")

Dataset sintético bidimensional con tres clases en forma de anteojos ($k = 3$, $D = 2$), en el que todos los estimadores salvo #logr y #slr alcanzan casi la misma exactitud, 97% --- casi perfecta. #fkdc saca una ventaja mínima pero significativa en $R^2$. Al ser bidimensional, es el único dataset multiclase donde las fronteras de decisión se pueden visualizar directamente.

#{
  let clfs = ("kdc", "fkdc", "svc", "kn", "fkn", "gbt", "slr", "lr", "gnb")
  wide-figure(width: 160%,
    grid(columns: 3, gutter: 4pt, ..clfs.map(clf => image(
      "img/anteojos-" + clf + "-decision_boundary.svg",
    ))),
    caption: flex-caption(
      [Fronteras de decisión de los nueve algoritmos evaluados sobre `anteojos` con semilla $s=#plotting_seed$. Se observa que #logr y #slr no logran separar las tres clases, mientras que los demás algoritmos alcanzan fronteras muy similares entre sí.],
      [Fronteras de decisión en `anteojos`],
    ),
  )
} <fig-fronteras-anteojos>

=== Datasets de alta dimensionalidad

#figure(
  image("img/anexo-hd-fkdc-vs-kdc.svg"),
  caption: flex-caption(
    [$R^2$ por semilla de #fkdc vs. #kdc en `digitos` ($D = 64$) y `mnist` ($D = 784$).],
    [$R^2$ de #fkdc vs. #kdc en alta dimensión],
  ),
)

=== `digitos` ($k=10, D=64$)

#highlights_figure("digitos")
El dataset de dígitos de scikit-learn ($N = 1797$, $k = 10$, $D = 64$, imágenes de $8 times 8$ píxeles, $approx 180$ observaciones por clase) es el caso más favorable a #kdc en todo el experimento: #fkdc es el mejor clasificador global ($R^2 approx 0.98$), superando ligeramente a #kdc ($R^2 approx 0.97$), y estos dos superan más claramente a todos los demás. En este espacio de alta dimensión con estructura manifiesta de variedades --- un dígito manuscrito es una curva parametrizada con $d=1$ en el espacio ambiente $D=2$ de la hoja ---, hubiéramos esperado alguna ventaja más notable de #fkdc sobre #kdc, o #fkn sobre #kn, que no se comprobó.

=== `mnist` ($k=10, D=96$)
#highlights_figure("mnist")

La versión grande del problema de `digitos`, al dataset de `mnist` ($N = 800$, $approx 80$ observaciones por clase) se lo redujo de $D = 768$ a $D = 96$ #footnote[número que se eligió para conservar al menos el 90% de la variación en los datos originales] por PCA para volverlo más manejable. Presenta un resultado interesante: #kdc ($R^2 approx 0.77$) supera a #fkdc ($R^2 approx 0.74$), con bastante menor dispersión también según se observa en el _boxplot_. Nos llena de orgullo que la implementación propia del clasificador de @loubesKernelbasedClassifierRiemannian2008 obtenga un $R^2$ que se separa por bastante de otros métodos más complejos como #gbt y casi tan bueno como el mejor en exactitud (#svc), aunque hemos de admitir que termina obteniendo la misma _performance_ que el más que clásico #logr en exactitud y $R^2$.

= Conclusiones

A priori, nuestras tres propuestas de estimación --- #kdc, #fkdc y #fkn --- obtuvieron resultados a la par de métodos de primera línea tanto paramétricos (#svc) como no paramétricos (#gbt). Al evaluarlos por "exactitud", a pesar de estar entrenados para maximizar la log-verosimilitud, los métodos resultaron competitivos aunque sin mejoras significativas. Al evaluarlos por $R^2$, sí se observaron excelentes rendimientos para toda la familia $cal(K)$: #fkdc obtuvo el máximo $R^2$ mediano en 7 de los 20 datasets evaluados, y #fkn en otros 3. #kdc resultó óptimo en 2 datasets adicionales, consolidando la técnica del clasificador de densidad por núcleos como competitiva por sí misma.

Ya existía una implementación previa de la Distancia de Fermat #link("https://www.aristas.com.ar/fermat/fermat.html")[como librería de Python], pero dado el extenso número de _datasets_ sobre los que evaluamos los clasificadores, nos inclinamos por una reimplementación propia, 
- con capacidad de estimación "out-of-sample",
- métodos mínimamente performantes basados en "primitivos" --- de `numpy` y `scipy` --- bien optimizados y
- compatible con el marco de métricas aceptado por los clasificadores de `scikit-learn`, para extenderlos con Distancia de Fermat pero mantener la capacidad de evaluación sistemática comparada.


Nuestros resultados son consistentes con la observación de @bijralSemisupervisedLearningDensity2012 de que basta con aprender un único parámetro $alpha$ ---que subsume los efectos de la dimensión intrínseca $d$ y el exponente $beta$--- sin necesidad de estimar la dimensión intrínseca de la variedad para que la distancia de Fermat resulte efectiva.

En ninguno de los datasets estudiados con bajo $D in {2, 3}$ se vieron modos "catastróficos" donde la _performance_ de #fkdc/#fkn fuese muchísimo peor que la de sus pares euclídeos. Sí hubo dos datasets "reales" --- `vino` y `pinguinos`, $k=3$ --- en los que #fkdc confunde dos de las clases por estar muy "ensimismadas" en algunas direcciones, y pierde mucho en exactitud. En los datasets en que se comprueba una ventaja sistemática de #fkdc (resp. #fkn) sobre #kdc (resp. #kn), se puede explicar por dos efectos:
1. En todos los casos examinados, una parte importante de la ventaja se da por una "simbiosis" positiva entre el mecanismo de selección de modelos de @r1sd, y el espacio de parámetros ampliado por la dimensión de $alpha$. Esta resulta en parametrizaciones de #fkdc (resp. #fkn) con $alpha=1$ y ligeramente mejor $R^2$ que las que #kdc (resp. #kn) selecciona.
2. En ciertos casos (como #fkn en `hueveras_0` o #fkdc en `helices_0`), al menos parte de la mejora se debe a la elección de parametrizaciones de #fkn (resp. #fkdc) que coinciden en el $k$ (resp. $h$) elegido con #kn (resp. #kdc), pero además registran un $alpha > 1$ - i.e., *una mejora netamente gracias al uso de la distancia de Fermat muestral*.

En el "caso general" en el que #fkdc anda aproximadamente como #kdc, observamos una relación log-lineal, $ log(h) prop alpha $ que se discierne en la _superficie de pérdida_ de entrenamiento como un "risco" de parametrizaciones equivalentes en bondad. Entendemos que esto sucede porque
- los datasets están "bien sampleados" y
- para todo $p in MM$ una variedad de Riemann, siempre existe un vecindario dentro del radio de inyectividad de $"iny"_p MM$ en el que $cal(D)_(f,beta) prop norm(dot)$
En estas circunstancias existe un $h <= "iny"_p MM$ tal que el efecto de $alpha$ "(des)inflando" la distancia euclídea puede ser sustituido completamente por una parametrización con distinto $h$, y no hay ventaja alguna que obtener usando distancia de Fermat (#fkn o #fkdc) en lugar de euclídea.

Los métodos de estimación por densidad de núcleos son "altamente locales", y por ende solo vemos mejoras no-triviales de $R^2$ en circunstancias extraordinarias, como en los datasets de `espirales`, `helices` o `hueveras` en que aun los vecindarios locales son altamente no-euclídeos. Observamos que a medida que nos alejamos de estos vecindarios, la performance de #fkdc y #fkn se mantiene muy competitiva, y la de #kdc, #kn degrada velozmente. Es un descubrimiento alentador, pero inútil en el contexto de tareas de clasificación como estas. Sí se observó, no obstante, que las fronteras de decisión que traza #fkdc se alinean cualitativamente con la forma de las variedades subyacentes ---una propiedad deseable que no resulta fácilmente reducible a una métrica escalar, y que se desdibuja en las comparaciones puramente numéricas.

Al aumentar el nivel de ruido en los datasets bidimensionales, el terreno de juego se nivela considerablemente: las ventajas de la distancia de Fermat disminuyen, y los métodos basados en densidad por núcleos sufren una degradación particularmente pronunciada, si bien caen desde rendimientos inicialmente superiores a un nivel de _performance_ similar al de otros métodos.

Con respecto a los tiempos de cómputo, no se hizo un análisis exhaustivo esencialmente porque no hizo falta: corrimos 25 repeticiones de 20 datasets para 9 clasificadores en unas 12 horas en mi computadora personal #footnote[Macbook Air M1 2020, 8GB RAM, 256GB SSD] media docena de veces hasta tenerlo todo a punto, y en general ni siquiera fue #sfd el algoritmo más problemático. Es cierto que el cómputo de #sfd --- que implica calcular geodésicas en grafos completos --- puede requerir varios órdenes de magnitud mayores recursos que el de la distancia euclídea, pero para datasets "moderados" (en el desarrollo se consideraron $n_k <= 1000, p <= 90$) el tiempo de cómputo de base es pequeñísimo, y aunque crezca por órdenes de magnitud no afecta significativamente la experiencia del investigador.

Un aspecto a tener en cuenta es la mayor varianza de #fkdc y #fkn respecto de sus pares euclídeos. Si bien la distancia de Fermat a veces mejora el $R^2$, también puede perjudicarlo: observamos _outliers_ negativos en datasets como `eslabones_0` ---donde una semilla adversa produjo $R^2_#fkdc approx 0.65$ contra $R^2_#kdc approx 0.9$--- y semillas en `helices_0` donde la misma parametrización que en otras repeticiones aventajaba a #kdc resultó significativamente peor. Esta mayor dispersión es un costo inherente al espacio ampliado de hiperparámetros.

En los casos de peor rendimiento ---particularmente los datasets de 15 dimensiones con ruido añadido, así como `helices_12` y `hueveras_12`--- la degradación se explica por la combinación de ruido excesivo para el tamaño muestral disponible, o por no haber realizado una reducción _lineal_ de dimensionalidad antes de aplicar la distancia de Fermat. Las 12 dimensiones de ruido añadidas a los datasets tridimensionales degradaron severamente a todos los estimadores por densidad de núcleos, cuya localidad los hace particularmente vulnerables a la maldición de la dimensionalidad, y el agregado del hiperparámetro $alpha$ no logra subsanar.

Cabe notar que _todo_ algoritmo evaluado --- incluso #logr sin escalar--- obtuvo el mejor rendimiento, o uno muy cercano, en al menos un dataset. No existe un único algoritmo universalmente óptimo; la elección sigue quedando a criterio del investigador según las características de cada problema. La distancia de Fermat y nuestras implementaciones de #fkdc y #fkn constituyen una herramienta más en la paleta del investigador.

== Trabajo Futuro
En el presente trabajo hemos desarrollado una librería y un marco teórico sumamente riguroso para intentar identificar condiciones en las cuales estimadores de densidad entrenados con distancia de Fermat muestral son estrictamente mejores que sus versiones euclídeas.

Es _infinita_ la cantidad de circunstancias en las que podemos poner a prueba una técnica de clasificación, y en los experimentos ejecutados y presentados no hemos hecho más que rascar la superficie. Así y todo, pareciera ser que en espacios ralamente sampleados o altamente curvos, donde "no quede otra" que tomar una ventana $h > "iny" MM$ para tener una densidad "viable", el uso de la distancia de Fermat mejora, si no la exactitud de los algoritmos, sí su $R^2$ y por ende la capacidad de discernimiento "relativo" de estos estimadores.

Sería interesante entonces investigar si existen condiciones reales en las que sepamos "a priori" que las variedades intrínsecas son altamente no-euclídeas, y en ese contexto probar si en ciertos tamaños muestrales $n$ (y por cada clase, $n_1, dots, n_k$) pequeños relativos a la dimensión ambiente es particularmente conveniente el uso de la distancia de Fermat.


Asimismo, la grilla de $alpha in [1, 4]$ utilizada en nuestros experimentos podría ampliarse ---@bijralSemisupervisedLearningDensity2012 reportan buenos resultados con $alpha = 8$--- para explorar si valores más extremos del parámetro ofrecen ventajas adicionales en datasets con geometrías particularmente complejas.


= Listados

#outline(target: figure.where(kind: image), title: "Listado de Figuras")
#outline(target: figure.where(kind: table), title: "Listado de Tablas")
#bibliography("references.bib", style: "harvard-cite-them-right")
