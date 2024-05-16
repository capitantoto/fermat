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
== Fan
Para tener una idea "sistémica" de la performance de los algoritmos, evaluaremos su performance con diferentes _datasets_. Muchos factores en la definición de un dataset pueden afectar la exactitud de la clasificación; nos interesará explorar en particular 3 que a su vez figuran en el cálculo de la densidad en variedades:
- $n$, el tamaño de la muestra,
- $d$, la dimensión de las observaciones y
- $k$, la cantidad de categorías.


= Resultados y Análisis
== Fantasías en 2D
#figure(image("img/fantasias-2d.png", width: 100%), caption: [Datasets sintéticos en $RR^2$])
== Iris
== digits

= Observaciones Generales
= Preguntas abiertas


#lorem(30)

+ #lorem(10)
+ #lorem(10)
+ #lorem(10)

#lorem(10)

#set align(center)
#table(
  columns:(auto, auto, auto), 
  inset:(10pt),
 [#lorem(4)], [#lorem(2)], [#lorem(2)],
 [#lorem(3)], [#lorem(2)], [$alpha$],
 [#lorem(2)], [#lorem(1)], [$beta$],
 [#lorem(1)], [#lorem(1)], [$gamma$],
 [#lorem(2)], [#lorem(3)], [$theta$],
)

#set align(left)
$ mat(
  1, 2, ..., 8, 9, 10;
  2, 2, ..., 8, 9, 10;
  dots.v, dots.v, dots.down, dots.v, dots.v, dots.v;
  10, 10, ..., 10, 10, 10;
) $
== #lorem(5)

#lorem(65)
#figure(
  image("contour.png", 
        width: 60%),
  caption: [#lorem(8)]
)

= #lorem(3)

#block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  [
    - #lorem(10),
    - #lorem(10),
    - #lorem(10),
  ]
)
#bibliography("../bib/references.bib")
