// Basado en How do I get the "LaTeX look?",
// de https://typst.app/docs/guides/guide-for-latex-users/
#set page(margin: 1.75in, numbering: "1 de 1")
#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)

#set heading(numbering: "1.1")
#import "@preview/ctheorems:1.1.3": *
#show: thmrules


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

#outline(depth: 2)

#outline(target: figure.where(kind: image), title: "Listado de Figuras")
#outline(target: figure.where(kind: table), title: "Listado de Tablas")
#outline(target: figure.where(kind: raw), title: "Listado de código")
= Tioremas
#let defn = thmbox("definition", "Definición", inset: (x: 1.2em, top: 1em))
#let obs = thmplain("observation", "Observación").with(numbering: none)

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

Como se explica en @la-mar-en-coche, es muy arriesgado cruzar el océano en un auto. O sea, no.
= Notación

Ahora se ve lo que escribo?


- $RR$: los números reales
- $d_x$
- $RR^(d_x)$
- $[k]$, el conjunto de los k números enteros, ${1, dots, k}$
- $cal(M)$
- $bold(upright(H))$
- $norm(dot)$
- ${bold(upright(X))}$
- $X_(i, j)$
// - $bu(H)$
#let bu(x) = $bold(upright(#x))$
$bu(X)$
#bu("H")

$abs(x + 2/(1/(1 + theta)))$

#bu("1")


// #let ops = array(rg, cos, cosh, cot, coth, csc, csch, ctg, deg, det, dim, exp, gcd, hom, id, im, inf, ker, lg, lim, liminf, limsup, ln, log, max, min, mod, Pr, sec, sech, sin, sinc, sinh, sup, tan, tanh, tg, tr)

$Pr(x in A)$
$bb(P)$

#let ind = $ op(bb(1)) $

$ind(x/y)$
= Preliminares

== El problema de clasificación


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

== La maldición de la (alta) dimensionalidad

=== NB: El clasificador "ingenuo" de Bayes
#defn("Naïve Bayes")[] <gnb>

==== Una muestra adversa

=== KDC Multivariado
#defn("KDE Multivariado")[] <kde-mv>
#defn("KDC")[] <kdc>

==== El caso 2-dimensional

==== Relación entre H y la distancia de Mahalanobis

== La hipótesis de la variedad

#figure(
  caption: flex-caption[La variedad $cal(U)$ con $dim(cal(U)) = 1$ embebida en $RR^2$. Nótese que en el espacio ambiente, el punto rojo está más cerca del verde, mientras que a través de $cal(U)$, el punto amarillo está más próximo que el rojo][Variedad $cal(U)$],
)[#image("img/variedad-u.svg", width: 70%)]
=== KDE en variedades de Riemann

=== Variedades desconocidas

== Aprendizaje de distancias

=== Isomap

=== Distancias basadas en densidad

== Distancia de Fermat
- Groisman & Jonckheere @groismanNonhomogeneousEuclideanFirstpassage2019
- Little & Mackenzie @littleBalancingGeometryDensity2021
- Bijral @bijralSemisupervisedLearningDensity2012
- Vincent & Bengio @vincentDensitySensitiveMetrics2003
#defn("Distancia Muestral de Fermat")[]<sample-fermat-distance>
= Propuesta Original

Habiendo andado este sendero teórico, la pregunta natural que asoma es: ¿es posible mejorar un algoritmo de clasificación reemplazando la distancia euclídea por una aprendida de los datos, como la de Fermat? Para investigar la cuestión, nos propusimos:
1. Implementar un clasificador basado en estimación de densidad por núcleos (@kde) según @loubesKernelbasedClassifierRiemannian2008, que llamaremos "KDC". Además,
2. Implementar un estimador de densidad por núcleos basado en la distancia de Fermat, a fines de poder comparar la _performance_ de KDC con distancia euclídea y de Fermat.

Nótese que el clasificador enunciado al inicio (k-NN, @knn), tiene un pariente cercano, $epsilon-upright("NN")$

#defn($epsilon-"NN"$)[] <eps-NN>

@eps-NN es esencialmente equivalente a KDC con un núcleo "rectangular", $k(t) =  ind(d(x, t) < epsilon) / epsilon$, pero su definición es considerablemente más sencilla. Luego, propondremos también
3. Implementar un clasificador cual @knn, pero con distancia muestral de Fermat en lugar de euclídea.

== KDC con Distancia de Fermat Muestral

== f-KNN

= Evaluación

Nos interesa conocer en qué circunstancias, si es que hay alguna, la distancia muestral de Fermat provee ventajas a la hora de clasificar por sobre la distancia euclídea. Además, en caso de existir, quisiéramos en la medida de lo posible comprender por qué (o por qué no) es que tal ventaja existe.
A nuestro entender resulta imposible hacer declaraciones demasiado generales al respecto de la capacidad del clasificador: la cantidad de _datasets_ posibles, junto con sus _configuraciones de evaluación_ es tan densamente infinita como lo permita la imaginación del evaluador. Con un ánimo exploratorio, nos proponemos explorar la _performance_ de nuestros clasificadores basados en distancia muestral de Fermat en algunas _tareas_ puntuales.

== Métricas de _performance_

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

== Algoritmos de referencia

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

== Metodología
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

== Resultados

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

#let params = json.decode(json("data/best_params-2d-lo.json"))
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

== Datasets sintéticos, Baja dimensión

== Datasets orgánicos, Mediana dimensión

== Alta dimensión: Dígitos

== Efecto de dimensiones guillemot ruidosasguillemot

== fKDC: Interrelación entre $h,alpha$

== fKNN: Comportamiento local-global

= Comentarios finales

== Conclusiones

== Posibles líneas de desarrollo

== Relación con el estado del arte

= Referencias

= Código Fuente

== sklearn

== fkdc

== Datasets
=== Datasets 2d

Esto digo yo

@rosenblattRemarksNonparametricEstimates1956
@carpioFingerprintsCancerPersistent2019
@chaconDatadrivenDensityDerivative2013

#bibliography("../bib/references.bib", style: "harvard-cite-them-right")
