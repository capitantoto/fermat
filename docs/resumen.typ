// Resúmenes en castellano e inglés de la tesis.
// Se compila como documento independiente: `typst compile docs/resumen.typ`.
// El contenido es una síntesis de la Introducción, Propuesta Original y
// Conclusiones de tesis.typ — para revisión del autor antes de la entrega.

#set page(paper: "a4", margin: 1.5in, numbering: none)
#set text(font: "New Computer Modern", lang: "es", size: 11pt)
#set par(leading: 0.55em, first-line-indent: 0pt, justify: true)
#show heading: set block(above: 1.4em, below: 1em)

// =====================
//  Resumen en español
// =====================

#align(center)[
  #text(size: 14pt, weight: "bold")[Resumen]
]

#v(0.5em)

En esta tesis se propone, implementa y evalúa una familia de clasificadores
supervisados no paramétricos que reemplazan la distancia euclídea por la
_distancia muestral de Fermat_ — una distancia basada en densidad que aproxima
la distancia geodésica sobre variedades de Riemann desconocidas.

Sobre la base del estimador de densidad por núcleos en variedades de Riemann de
Pelletier (2005) y su extensión a clasificación de Loubes y Pelletier (2008),
implementamos tres algoritmos: el clasificador de densidad por núcleos *KDC*;
su contraparte basada en distancia muestral de Fermat, *f-KDC*; y un análogo
de $k$-vecinos más cercanos también basado en distancia de Fermat, *f-KN*. Los
tres se distribuyen como biblioteca de código abierto, compatible con la
interfaz de `scikit-learn`.

La evaluación experimental sistemática abarcó 20 _datasets_ — sintéticos y
reales, de dimensiones intrínsecas y ambientes diversas — y nueve algoritmos.
Las propuestas resultaron competitivas con técnicas de referencia paramétricas
(regresión logística, SVM) y no paramétricas (_gradient boosting_): f-KDC
obtuvo el máximo $R^2$ mediano en 7 _datasets_, f-KN en 3 y KDC en 2 más. La
ventaja atribuible específicamente al uso de la distancia de Fermat se
concentra en espacios ralamente muestreados o altamente curvos, en los que el
ancho de banda requerido supera el radio de inyectividad de la variedad y los
supuestos de Pelletier dejan de cumplirse. En el régimen "bien muestreado",
los hiperparámetros de ancho de banda y exponente de Fermat se vuelven
intercambiables y no se observa ganancia neta. Sin embargo, las fronteras de
decisión que dibuja f-KDC se alinean cualitativamente con la geometría
subyacente — una propiedad deseable que escapa a las métricas escalares de
comparación.

#v(1em)

*Palabras clave:* clasificación supervisada, estimación de densidad por
núcleos, variedades de Riemann, distancia de Fermat, distancias basadas en
densidad, aprendizaje no paramétrico.

// =====================
//  Abstract in English
// =====================

#pagebreak()

#set text(lang: "en")

#align(center)[
  #text(size: 14pt, weight: "bold")[Abstract]
]

#v(0.5em)

This thesis proposes, implements, and evaluates a family of non-parametric
supervised classifiers that replace the Euclidean distance with the _sample
Fermat distance_ — a density-based distance that approximates the geodesic
distance on unknown Riemannian manifolds.

Building on Pelletier's (2005) kernel density estimator on Riemannian
manifolds and its classification extension by Loubes and Pelletier (2008), we
implement three algorithms: the kernel density classifier *KDC*; its
Fermat-distance counterpart *f-KDC*; and an analogous Fermat-distance-based
$k$-nearest-neighbours classifier *f-KN*. All three are distributed as an
open-source library compatible with the `scikit-learn` interface.

A systematic experimental evaluation was carried out on 20 datasets —
synthetic and real, with diverse intrinsic and ambient dimensions — using
nine algorithms in total. The proposed methods are competitive with both
parametric (logistic regression, SVM) and non-parametric (gradient boosting)
baselines: f-KDC achieves the highest median $R^2$ on 7 datasets, f-KN on 3
more, and KDC on a further 2. The advantage specifically attributable to the
Fermat distance concentrates in sparsely sampled or highly curved spaces,
where the required bandwidth exceeds the manifold's injectivity radius and
Pelletier's assumptions break down. In the "well-sampled" regime, the
bandwidth and Fermat exponent hyperparameters become interchangeable, and no
net gain is observed. Nonetheless, the decision boundaries drawn by f-KDC
align qualitatively with the underlying manifold geometry — a desirable
property that escapes scalar comparison metrics.

#v(1em)

*Keywords:* supervised classification, kernel density estimation, Riemannian
manifolds, Fermat distance, density-based distances, non-parametric learning.
