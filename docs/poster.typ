// Tempalte generosamente tomado de 
// https://github.com/pncnmnp/typst-poster/blob/master/poster.typ
// Veremos si funca

#import "typst-poster.typ": *

#set text(lang: "es")

#show: poster.with(
  size: "36x54",
  title: [Clasificación por KDE con Distancia de Fermat en variedades desconocidas: una aproximación empírica.  #text(0.8em)[]],
  authors: "Lic. Gonzalo Barrera Borla (IC), Dr. Pablo Groisman (DM)",
  departments: "Facultad de Ciencias Exactas y Naturales, UBA",
  univ_logo: "logofac.jpg",
  footer_text: "II Jornada de Temas de Investigación en Estadística Matemática, 2024",
  footer_url: "https://github.com/capitantoto/fermat",
  footer_email_ids: "gonzalobb@gmail.com",
  footer_color: "ebcfb2",
  
  // Use the following to override the default settings
  // for the poster header.
  num_columns: "2",
  univ_logo_scale: "75",
  font_size: "28",
  title_font_size: "60",
  authors_font_size: "40",
  univ_logo_column_size: "4",
  title_column_size: "28",
  footer_url_font_size: "24",
  footer_text_font_size: "38",
)
#show heading: text.with(size: 2em, weight: "bold")

#show "Mu": $cal(M)$

= Síntesis

Siguiendo a Loubes & Pelletier @loubesKernelbasedClassifierRiemannian2008, programamos y evaluamos un algoritmo de clasificación basado en _Kernel Density Estimation_ ("KDC") para v.a. soportadas en variedades de Riemann, "KDC". Luego, reemplazamos la distancia euclídea por  la _Distancia (muestral) de Fermat_ investigada por Groisman et al. @groismanNonhomogeneousEuclideanFirstpassage2019, e implementamos el clasificador resultante, Fermat KDC (FKDC")". Finalmente, evaluamos la exactitud ("accuracy") de ambos clasificadores contra otros algoritmos estándares: SVC, regresión logística, kNN y Naive Bayes. Resultados preliminares muestran que tanto KDC como FKDC performan consistentemente como los mejores algoritmos en cada tarea, pero la performance de FKDC nunca supera la de su par euclídeo, técnicamente es un caso particular de FKDC. Concluimos con algunas hipótesis sobre el comportamiento observado.

= Contexto
== KDE en variedades de Riemann
Sea $(cal(M), g)$ resp. una variedad Riemanniana Mu y su métrica $g$, compacta y sin frontera de dimensión $d$, y denotemos $d_g$ la distancia cpte. Sea $X$ un elemento aleatorio (e.a.) con soporte en Mu y función de densidad $f$, y ${X_1, dots, X_N}$ una muestra de ee. aa. i.i.d. a $X$. Sean, además, $K$ una "función núcleo" y $h > 0$ un "ancho de banda". Entonces, la estimación de $f$ por KDE es
$ hat(f)(x) = N^(-1)sum_(i=1)^N frac(1, h^d)frac(1, theta_(X_i)(x))K(frac(d_g (x, X_i), h)) $

donde $theta_(p)(q)$ es la _función de densidad volumétrica_ en Mu alrededor de $p$. Obsérvese que cuando $cal(M) = RR^d$ y $g$ es la métrica euclídea, $theta_(p)(q) = 1 forall (p, q)$, y $hat(f)$ se reduce a la más conocida
$ hat(f)(x) = N^(-1)sum_(i=1)^N frac(1, h^d)K(frac(||x - X_i||, h)) $

El "núcleo gaussiano" $K(x) = 1 / sqrt(2 pi) exp(-1/2 x^2)$ es casi universal; la elección de $h$ es crítica y está ampliamente tratada en la literatura, no así la elección de la distancia $d_g$.

== Clasificación por KDE
Sean ahora $k in NN$ "clases", y la muestra ${(X_1, Y_1), dots, (X_N, Y_N)}, Y_i in {1, dots, k}$ de N elementos separados en k submuestas de tamaño $N_1, dots, N_k$, cada una soportada en su propia variedad (no encesariamente la misma) con densidad $f_j, j in {1, dots, k}$. Sea $p_j$ la proporción poblacional de la clase $j$, aproximada por $hat(p)_j=n_j/N$, y $hat(f)_j$ el estimador por KDE de $f_j$ ya descrito. Loubes & Pelletier @loubesKernelbasedClassifierRiemannian2008, basándose en el criterio de Bayes, plantean como regla de clasificación para un nuevo $(x, y)$

$ hat(y) = arg max_(j in 1, dots, k) hat(f_j)(x) hat(p)_j = sum_(i=1)^N bb(1){Y_i=j}K_h (x, X_i) $ <clf>

donde $bb(1){dot}$ es la función indicadora, y $K_h (x, X_i) = frac(1, h^d)frac(1, theta_(X_i)(x))K(frac(d_g (x, X_i), h))$. Implementar la regla de @clf requiere conocer la geometría de la(s) variedad(es) involucradas, que rara vez es factible. Una alternativa es _aprender la distancia de los datos_.
== Aprendizaje de Distancias: Isomap, Distancia de Fermat
Si los elementos muestrales $X_i in cal(M)$, y la variedad es "suficientemente regular", el segmento $overline(X_i X_j)$ también pertenece a Mu. Isomap (Tenenbaum et al @tenenbaumGlobalGeometricFramework2000), pionero en esta tónica, plantea esencialmente aproximar la distancia en Mu por la geodésica en el grafo geométrico de $k$ o $epsilon$ vecinos más cercanos.
En una propuesta tal vez superadora, Groisman et al @groismanNonhomogeneousEuclideanFirstpassage2019 proponen la "Distancia de Fermat", una distancia propiamente dicha en Mu, y muestran cómo ésta se puede aproximar "microscópicamente". Sea $Q$ el grafo completo de la muestra, y $alpha >= 1$, luego

$ D_(Q, alpha)(x, y) = inf{sum_(i=1)^K ||q_(i-1) - q_i||^alpha : (q_0, dots, q_K) "es un camino de x a y"} $ <sample_fermat>

es la "distancia muestral de Fermat. Nótese que usar el grafo completo obvia la necesidad de elegir ($k \/ epsilon$), mientras que $alpha > 1$ "infla" el espacio y desalienta los "saltos largos" por "espacio vacío" fuera de Mu. Cuando $alpha = 1$, la distancia de Fermat se reduce a la distancia euclídea.

= Propuesta y Metodología
  title: [Clasificación por KDE con Distancia de Fermat en variedades desconocidas: una aproximación empírica.  #text(0.8em)[]],
En la tesis desarrollamos un clasificador compatible con el _framework_ de #link("https://arxiv.org/abs/1309.0238", `scikit-learn`)  según los lineamientos de @loubesKernelbasedClassifierRiemannian2008 que apodamos `KDC`. Luego, implementamos el estimador de @sample_fermat, y combinándolo con KDC, obtenemos la titular "Clasificación por KDE con Distancia de Fermat", `FKDC`. Evaluamos la _exactitud_ ("accuracy") de los clasificadores propuestos en diferentes _datasets_, relativa a técnicas bien establecidas:
#columns(2)[
- regresión logística (`LR`)
- clasificador de  soporte vectorial (`SVC`)
#colbreak()
- k-vecinos-más-cercanos (`KN`)
- Naive Bayes Gaussiano (`GNB`)]
El criterio de evaluación consiste en (1) partir la muestra en entrenamiento y testeo; (2) elegiremos _hiperparámetros óptimos_ por _validación cruzada en 5 pliegos_ entre los datos de entrenamiento, y
(3) medir la exactitud de cada algoritmo algoritmos en conjunto de testeo de (1).

Para tener una idea "sistémica" de la performance de los algoritmos, evaluaremos su performance con _datasets_ que varíen en el tamaño muestral $N$, la dimensión $p$ de las $X_i$ y el nro. de clases $k$.
= Análisiss



== Fantasías en $RR^2$ <fantasias>

#figure(image("img/poster-fig-1.png", width: 90%), caption: [Datasets sintéticos en $RR^2$])
#let best = it => table.cell(fill: luma(85%), it)
#set text(size: 24pt)
#set align(center)
#table(columns: 8, align: center+horizon,
table.header([*Ruido*],[*Dataset*],[*FKDC*],[*GNB*],[*KDC*],[*KN*],[*LR*],[*SVC*]),
table.cell(rowspan: 3, [Alto]),[Circulos], [67.2 (4.4)],[63.5 (7.0)],[67.0 (4.3)],[67.3 (4.5)],[44.8 (4.6)],best[71.3 (5.1)],
[Espirales],[76.2 (4.8)],[48.5 (6.2)],[76.6 (4.4)],[76.0 (5.2)],[48.6 (5.7)],best[78.7 (4.0)],
[Lunas],[79.7 (5.6)],[80.4 (3.9)],best[81.3 (4.8)],[80.9 (4.4)],[80.7 (3.9)],[81.2 (5.0)],
table.cell(rowspan: 3, [Bajo]),[Circulos],[78.4 (4.1)],[67.7 (11.3)],[78.5 (4.1)],[79.1 (4.2)],[45.0 (4.5)],best[81.2 (5.4)],
[Espirales],[90.0 (3.2)],[49.6 (6.2)],[90.4 (3.2)],[90.3 (2.9)],[49.5 (6.5)],best[92.9 (1.7)],
[Lunas],[88.0 (4.6)],[83.6 (4.3)],best[88.1 (4.6)],[87.8 (4.6)],[83.9 (4.0)],[88.0 (3.7)],
)
Exactitud (en %), con sus respectivos desviós estándares a lo largo de 16 repeticiones de cada experimento.
#set text(size: 28pt)
#set align(left)
Comenzamos por 3 datasets, `lunas, circulos, espirales`, con $k=2, p=2, n=400, n_1=n_2=200$, que presentan variedades de dimensión intrínsica $d=1$, a las cuales se les agrego "ruido" gaussiano con "bajo" y "alto" desvío estándar ($sigma_"alto" approx 1.5 sigma_"bajo"$). 
Las performances de SVC, KN, KDC y FKDC no son significativamente distintas, aunque SVC parece ligeramente superior. Es alentador ver que la performance de KDC es siempre competitiva, pero descorazonador ver que FKDC es sistemáticamente igual o ligeramente peor que KDC.


== `vino`, `pinguinos`, `iris` y `anteojos` ($ k = 3$)

#figure(image("img/poster-fig-2.png", width: 100%), caption: [Datasets  con $k=3$. Salvo por `anteojos`, todos los datasets son pequeños pero reales.])

#set text(size: 24pt)
#set align(center)
#table(columns: 7, align: center+horizon,
table.header([*Dataset*],[*FKDC*],[*GNB*],[*KDC*],[*KN*],[*LR*],[*SVC*]),
[Anteojos],[97.5 (1.4)],[97.0 (1.8)],[97.4 (1.4)],[97.7 (1.4)],[50.5 (5.0)],best[97.7 (1.8)],
[Iris],[94.4 (4.3)],[94.6 (5.0)],[94.0 (4.4)],[95.4 (4.0)],best[97.5 (2.3)],[94.2 (5.8)],
[Pinguinos],[84.0 (4.2)],[97.8 (1.6)],[84.1 (4.2)],[85.2 (3.8)],[66.6 (4.5)],best[98.2 (1.0)],
[Vino],[71.9 (7.1)],best[96.9 (2.2)],[73.8 (6.3)],[71.0 (6.5)],[66.0 (6.7)],[95.3 (2.6)],
)
#set text(size: 28pt)
#set align(left)
En los datasets de `anteojos` e `iris`, se observa el mismo fenómeno que en los datasets "2D": (F)KDC es competitivo con los mejores métodos (SVC y LR, resp.), pero no superador. En los datasets de `pinguinos` y `vino`, la _performance_ de los métodos propuestos es significativamente peor. En todos los casos, no conseguimos mejoras significativas sobre KDC con FKDC.


== digitos
#image("img/poster-fig-3.png", width: 100%)
Los ee.aa. son imágenes de 8x8 (_id est_, en $RR^64$) que representan dígitos manuscritos. Es de esperar que la variedad donde yacen los trazos sea de menor dimensión, para hacer buen uso de la "estimación de la variedad" que promete FKDC. Consideramos dos regímenes de evaluación: sobre el 80% de entrenamiento ("escaso") y sobre el 20%("denso"), para ver si FKDC destaca en alguno.
#set text(size: 24pt)
#set align(center)
#table(columns: 7, align: center+horizon,
table.header([Eval.],[FKDC],[GNB],[KDC],[KN],[LR],[SVC]),
[20%],[98.8 (0.7)],[92.1 (1.1)],[98.9 (0.6)],[98.9 (0.4)],[96.7 (0.6)],best[99.0 (0.6)],
[80%],[97.0 (0.4)],[90.2 (0.6)],[96.9 (0.5)],[96.6 (0.8)],[94.5 (0.7)],best[97.5 (0.4)],
)
#set text(size: 28pt)
#set align(left)
Una vez más, FKDC no se distingue de  KDC, y a su vez ambos andan tan bien pero no mejor que KN y SVC. La diferencia entre ambos regímenes de evaluación es leve: pareciera que con sólo el 20% de los datos de entrenamiento, la muestra ya es suficientemente "densa".

= FKDC versus KDC
#columns(2, gutter: 2pt)[
El hecho de que la performance de FKDC sea casi idéntica a la de su primo euclídeo, se encuentra parcialemente por el hecho de que en la mayoría de los casos $alpha_"opt" approx 1$ (y el $h_"opt"$ de ambos métodos es similar, lo que indica la coherencia interna de FKDC), como se observa en la tabla a derecha con hiperparámetros óptimos para una semilla al azar en c/ experimento. Lo que no queda claro aún, es por qué la performance de FKDC tampoco mejora cuando $alpha_"opt" > 1$.
#colbreak()
#align(center)[
#table(columns: 4, align: center+horizon,
table.header(table.cell(rowspan: 2, [Dataset]),[KDC],table.cell(colspan: 2,[FKDC]), $h$, $h$, $alpha$),
[Circulos (alto)],[0.13],[0.06],[2.12],
[Espirales (alto)],[0.03],[0.16],[2.12],
[Lunas (alto)],[0.33],[0.29],[1.0],
[Circulos (bajo)],[0.09],[0.03],[1.94],
[Espirales (bajo)],[0.01],[0.01],[1.38],
[Lunas (bajo)],[0.48],[0.4],[1.19],
[Anteojos],[0.01],[0.01],[1.0],
[Iris],[0.04],[0.03],[1.0],
[Pinguinos],[57.54],[73.56],[1.0],
[Vino],[33.11],[29.29],[1.0],
[Digitos],[10.96],[15.85],[1.19],
)]
]
Cuando observamos la "accuracy" com función de $h$ para distintos $alpha$ en la etapa de testeo, pareciera ser que hay un "techo" a la performance, y aún cuando para cierto $h$ exista un $alpha_"opt" > 1$, lo cierto es que para _cualquier_ $alpha$, existe un $h_"opt" = f(alpha | "dataset")$ de performance equivalente. 
#figure(columns(2, [#image("img/poster-iris-test-score.png") #colbreak() #image("img/poster-digitos-test-score.png")]), caption: [Exactitud en validación para `iris` (izq.) y `digitos` (der.)])

= Conclusiones y Trabajo a Futuro
La sensación es "agridulce": el algoritmo de clasificación por KDE resulta competitivo con métodos bien establecidos, pero no encontramos aún mejoras marginales por el uso de la distancia muestal de Fermat. ¿Por qué? Tal vez en los datasets considerados la variedad subyacente no difiera mucho del espacio euclídeo ambiente. Esta hipótesis es problemática en tanto las `lunas, circulos y espirales` son claramente unidimensionales en $RR^2$, y cuesta pensar en `digitos` como elementos de $RR^64$.

Otra alternativa - no la única - es que cuando Mu dufiere de su espacio ambiente, $theta_(p)(q)$ (la _función de densidad volumétrica_ en Mu alrededor de $p$) sea sumamente variable en el espacio, e ignorarla nos haga pesar incorrectamente las observaciones. Al autor del trabajo no le resulta familiar la geometría riemanniana, lo cual dificulta la corroboración de dicha hipótesis. 

#bibliography("../bib/references.bib")
