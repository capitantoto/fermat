// Tempalte generosamente tomado de 
// https://github.com/pncnmnp/typst-poster/blob/master/poster.typ
// Veremos si funca

#import "typst-poster.typ": *

#set text(lang: "es")

#show: poster.with(
  size: "36x48",
  title: [Clasificación por KDE con Distancia de Fermat en variedades desconocidas: una aproximación empírica.  #text(0.8em)[]],
  authors: "Lic. Gonzalo Barrera Borla (IC), Dr. Pablo Groisman (DM)\n",
  departments: "Facultad de Ciencias Exactas y Naturales, UBA",
  univ_logo: "logofac.jpg",
  footer_text: "II Jornada de Temas de Investigación en Estadística Matemática, 2024",
  footer_url: "https://github.com/capitantoto/fermat",
  footer_email_ids: "gonzalobb@gmail.com",
  footer_color: "ebcfb2",
  
  // Use the following to override the default settings
  // for the poster header.
  num_columns: "2",
  univ_logo_scale: "100",
  font_size: "32",
  title_font_size: "68",
  authors_font_size: "50",
  univ_logo_column_size: "4",
  title_column_size: "24",
  footer_url_font_size: "24",
  footer_text_font_size: "38",
)
#show heading: text.with(size: 2em, weight: "bold")

#show "Mu": $cal(M)$

= Síntesis

En el presente trabajo, analizamos empíricamente el efecto de la elección de distancia en la performance de un clasificador por _Kernel Density Estimation_ (KDC) en variedades desconocidas. En particular, reemplazamos la distancia euclídea con una distancia "aprendida de los datos" propuesta por Groisman et al, la _Distancia de Fermat_, e intentamos medir la mejora marginal en la exactitud ("accuracy") de clasificación por sobre (a) KDC con distancia euclídea y (b) otros algoritmos estándares: SVC, regresión logística, kNN y Naive Bayes. Los resultados preliminares muestran que KDC es un método consistentemente performante, pero (a) nunca superior a SVC, y (b) que la performance de FKDC _no supera_ a la de su par euclídeo, que técnicamente es un caso particular de FKDC. Finalmente, consideramos por qué sería éste el caso, y listamos oportunidades de mejora del algoritmo.

= Contexto
En  aras de la brevedad, notamos aquí sólo conceptos fundamentales, siguiendo a Pelletier et al para la definición de KDC y el clasificador asociado y Groisman et al para la Distancia de Fermat.
== KDE en variedades de Riemann
Sea $(cal(M), g)$ resp. una variedad Riemanniana Mu y su métrica $g$, compacta y sin frontera de dimensión $d$, y denotemos $d_g$ la distancia cpte. Sea $X$ un elemento aleatorio (e.a.) con soporte en $cal(m)$ y función de densidad $f$, y ${X_1, dots, X_N}$ una muestra de ee. aa. i.i.d. a $X$. Sean, además, $K$ una "función núcleo" y $h > 0$ un "ancho de banda". Entonces, la estimación de $f$ por KDE es
 la estimación de densidad por KDE es

$ hat(f)(x) = N^(-1)sum_(i=1)^N frac(1, h^d)frac(1, theta_(X_i)(x))K(frac(d_g (x, X_i), h)) $

donde $theta_(p)(q)$ es la _función de densidad volumétrica_ en Mu alrededor de $p$. Obsérvese que cuando $cal(M) = RR^d$ y $g$ es la métrica euclídea, $theta_(p)(q) = 1 forall (p, q)$, y $hat(f)$ se reduce a la más conocida
$ hat(f)(x) = N^(-1)sum_(i=1)^N frac(1, h^d)K(frac(||x - X_i||, h)) $

Tomar por "núcleo gaussiano" $K(x) = 1 / sqrt(2 pi) exp(-1/2 x^2)$ es práctica casi universal, mientras que la elección de $h$ es crucial para encontrar un buen estimador y está ampliamente tratada en la literatura. Menos estudiada está la elección de la distancia $d_g$ en variedades desconocidas, y es aquí donde entra en juego la Distancia de Fermat.

== Clasificación por KDE
Sean ahora $k in NN$ "clases", y contemos con una muestra  ${(X_1, Y_1), dots, (X_N, Y_N)}, Y_i in {1, dots, k} forall i in {1, dots N}$, de manera que los N elementos se separen en k submuestas de tamaño $N_1, dots, N_k$, cada una soportada en su propia variedad (no encesariamente la misma), y función de densidad $f_j, j in 1, dots, k$. Sea $p_j$ la proporción poblacional de la clase $j$ en la muestra completa, que aproximamos como $hat(p)_j=n_j/N$, y $hat(f)_j$ el estimador por KDE de $f_j$ ya descrito. Loubes y Pelletier, basándose en el criterio de Bayes, plantean como regla de clasificación para un nuevo e.a. $(x, y)$

$ hat(y) = arg max_(j in 1, dots, k) hat(f_j)(x) hat(p)_j = sum_(i=1)^N bb(1){Y_i=j}K_h (x, X_i) $ <clf>

donde $bb(1){dot}$ es la función indicadora, y $K_h (x, X_i) = frac(1, h^d)frac(1, theta_(X_i)(x))K(frac(d_g (x, X_i), h))$.

Una implementación práctica de la regla en @clf requiere conocer la geometría de la(s) variedad(es) en la(s) que se soportan las muestras.  En la práctica, es harto común asumir la hipótesis de la variedad, pero desconocer su exacta geometría. En tal contexto, una alternativa es _aprender la distancia de los datos_.
#lorem(20)
== Aprendizaje de Distancias: Isomap, Distancia de Fermat
Un punto de partida natural, es asumir que los elementos muestrales $X_i in cal(M)$, y que si la variedad es suficientemente regular, el segmento $overline(X_i X_j)$ también pertenece a Mu. Con esta lógica, Tenenbaum et al @tenenbaumGlobalGeometricFramework2000 desarrollan Isomap, un algoritmo precursor en el aprendizaje de distancias:
+ Construya $G$, el grafo de $k$ o $epsilon$ vecinos más cercanos y pese cada arista por su métrica euclídea,
+ Tome por distancia aprendida $d$ entre dos nodos la longitud del camino mínimo entre ellos,
+ Compute una representación de menor dimensión en un espacio euclídeo.

Más allá de su efectividad, tal algoritmo no deja de ser una heurística inteligente, y depende crucialmente de una correcta elección del parámetro de cercanía ($k \/ epsilon$). En una propuesta similar pero superadora, Groisman et al @groismanNonhomogeneousEuclideanFirstpassage2019 proponen la "Distancia de Fermat", una distancia propiamente dicha en variedades, y muestran cómo ésta se puede aproximar "microscópicamente" a partir de una muestra. Sea $Q$ el grafo completo de la muestra, y $alpha >= 1$, luego

$ D_(Q, alpha)(x, y) = inf{sum_(i=1)^K ||q_(i-1) - q_i||^alpha : (q_0, dots, q_K) "es un camino de x a y"} $ <sample_fermat>

es la "distancia muestral" de Fermat. Nótese que usar el grafo completo obvia la necesidad de elegir ($k \/ epsilon$), mientras que $alpha > 1$ "infla" el espacio y desalienta los "saltos largos" por espacio vacío. Cuando $alpha = 1$, la distancia de Fermat se reduce a la distancia euclídea.

= Propuesta
En la tesis desarrollamos un clasificador compatible con el _framework_ de #link("https://arxiv.org/abs/1309.0238", `scikit-learn`)  según los lineamientos de @loubesKernelbasedClassifierRiemannian2008.  Asumiendo que Mu es $RR^d$, obtenemos un clasificador que apodamos `KDC`. Luego, implementamos el estimador de @sample_fermat, y reemplazamos la distancia euclídea de `KDC` por la distancia muestral de fermat, para compeltar nuestra propuesta, `FKDC`.

= Metodología
Deseamos evaluar la _exactitud_ ("accuracy") de los clasificadores propuestos en diferentes _datasets_, relativa a técnicas bien establecidas:
#columns(2)[
- regresión logística (`LR`)
- clasificador de  soporte vectorial (`SVC`)
#colbreak()
- k-vecinos-más-cercanos (`KN`)
- Naive Bayes Gaussiano (`GNB`)]

Para comparar equitativamente estos _algoritmos_ de clasificación,
- partiremos la muestra en entrenamiento y testeo,
- elegiremos los _hiperparámetros óptimos_ por _validación cruzada en 5 pliegos_ entre los datos de entrenamiento, y
- mediremos la exactitud de todos los algoritmos en el mismo conjunto de testeo.

= Análisiss
Para tener una idea "sistémica" de la performance de los algoritmos, evaluaremos su performance con diferentes _datasets_. Muchos factores en la definición de un dataset pueden afectar la exactitud de la clasificación; nos interesará explorar en particular 3 que a su vez figuran en el cálculo de la densidad en variedades:, el tamaño de la muestra $n$, la dimensión $p$ de las observaciones y la cantidad $k$ de categorías.


== Fantasías en $RR^2$ <fantasias>
#figure(image("img/fantasias-2d.png", width: 100%), caption: [Datasets sintéticos en $RR^2$])
#let best = it => table.cell(fill: green, it)

#table(columns: 8, align: center+horizon,
table.header([*Ruido*],[*Dataset*],[*FKDC*],[*GNB*],[*KDC*],[*KN*],[*LR*],[*SVC*]),
table.cell(rowspan: 3, [Alto]),[Circulos], [67.2 (4.4)],[63.5 (7.0)],[67.0 (4.3)],[67.3 (4.5)],[44.8 (4.6)],best[71.3 (5.1)],
[Espirales],[76.2 (4.8)],[48.5 (6.2)],[76.6 (4.4)],[76.0 (5.2)],[48.6 (5.7)],best[78.7 (4.0)],
[Lunas],[79.7 (5.6)],[80.4 (3.9)],best[81.3 (4.8)],[80.9 (4.4)],[80.7 (3.9)],[81.2 (5.0)],
table.cell(rowspan: 3, [Bajo]),[Circulos],[78.4 (4.1)],[67.7 (11.3)],[78.5 (4.1)],[79.1 (4.2)],[45.0 (4.5)],best[81.2 (5.4)],
[Espirales],[90.0 (3.2)],[49.6 (6.2)],[90.4 (3.2)],[90.3 (2.9)],[49.5 (6.5)],best[92.9 (1.7)],
[Lunas],[88.0 (4.6)],[83.6 (4.3)],best[88.1 (4.6)],[87.8 (4.6)],[83.9 (4.0)],[88.0 (3.7)],
)
Exactitud (espresada en porcentaje), con sus respectivos desviós estándares a lo largo de 16 repeticiones de cada experimento.

Los tres datasets, `lunas, circulos, espirales`, tienen $k=2, p=2, n=400, n_1=n_2=200$, y presentan variedades de dimensión intrínsica $d=1$, a las cuales se les agrego "ruido" gaussiano con "bajo" y "alto" desvío estándar ($sigma_"alto" approx 1.5 sigma_"bajo"$). 
En los tres datasets, la performance de SVC es consistentemente la mejor, aunque KN, KDC y FKDC no son significativamente distintos si consideramos un intervalo de confianza razonable.
Es alentador ver que la performance de KDC es siempre competitiva, pero descorazonador ver que FKDC es sistemáticamente igual o ligeramente peor que KDC.

== `vino`, `pinguinos`, `iris` y `anteojos`
El siguiente conjunto de datos contiene $k=3$ con diferente cantidad de predictores y $n$. Salvo por "anteojos", todos los datasets son pequeños pero reales.
#figure(image("img/fig2.png", width: 100%), caption: [Datasets  con $k=3$])


#table(columns: 7, align: center+horizon,
table.header([*Dataset*],[*FKDC*],[*GNB*],[*KDC*],[*KN*],[*LR*],[*SVC*]),
[Anteojos],[97.5 (1.4)],[97.0 (1.8)],[97.4 (1.4)],[97.7 (1.4)],[50.5 (5.0)],best[97.7 (1.8)],
[Iris],[94.4 (4.3)],[94.6 (5.0)],[94.0 (4.4)],[95.4 (4.0)],best[97.5 (2.3)],[94.2 (5.8)],
[Pinguinos],[84.0 (4.2)],[97.8 (1.6)],[84.1 (4.2)],[85.2 (3.8)],[66.6 (4.5)],best[98.2 (1.0)],
[Vino],[71.9 (7.1)],best[96.9 (2.2)],[73.8 (6.3)],[71.0 (6.5)],[66.0 (6.7)],[95.3 (2.6)],
)
En los datasets de `anteojos` e `iris`, se observa el mismo fenómeno que en los datasets "2D": (F)KDC es competitivo con los mejores métodos (SVC y LR, resp.), pero no superador. En los datasets de `pinguinos` y `vino`, la _performance_ de los métodos propuestos es significativamente peor. En todos los casos, no conseguimos mejoras significativas sobre KDC con FKDC.


== digitos
Los ee.aa. a estudiar son imaágenes de 8x8 (_id est_, en $RR^64$) que representan dígitos manuscritos. En este caso, aunque $p=64$, es de esperar que la variedad donde yacen los trazos sea de mucha menor dimensión, y mejores resultados esperaríamos de la "estimación de la variedad" que promete FKDC.

#figure(image("img/fig3.png", width: 80%), caption: [Dígitos manuscritos en B&N, 8x8 píxeles])

#table(columns: 7, align: center+horizon,
table.header([Eval.],[FKDC],[GNB],[KDC],[KN],[LR],[SVC]),
[20%],[98.8 (0.7)],[92.1 (1.1)],[98.9 (0.6)],[98.9 (0.4)],[96.7 (0.6)],best[99.0 (0.6)],
[80%],[97.0 (0.4)],[90.2 (0.6)],[96.9 (0.5)],[96.6 (0.8)],[94.5 (0.7)],best[97.5 (0.4)],
)



= Observaciones Generales

#bibliography("../bib/references.bib")
