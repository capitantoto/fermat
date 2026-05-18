// Resúmenes en castellano e inglés de la tesis.
//
// El archivo expone dos versiones de cada resumen, controladas por la variable
// `version` (definida abajo): "larga" (≤400 palabras, default) o "corta"
// (≤240 palabras). La versión larga es la que arma `make TESIS_BARRERA_GONZALO.pdf`.
//
// Citas: las menciones inline a Pelletier, Loubes & Pelletier y Groisman et al.
// corresponden a las siguientes claves en `docs/references.bib`:
//   - Pelletier (2005)                  → @pelletierKernelDensityEstimation2005
//   - Loubes & Pelletier (2008)         → @loubesKernelbasedClassifierRiemannian2008
//   - Groisman, Jonckheere & Sapienza (2022) → @groismanNonhomogeneousEuclideanFirstpassage2022
//
// Compilación: `typst compile docs/resumen.typ`.

// =====================
//  Parámetro de versión
// =====================
#let version = "larga"  // "larga" (≤400 palabras) | "corta" (≤240 palabras)

// =====================
//  Estilo de página
// =====================
#set page(paper: "a4", margin: 1.5in, numbering: none)
#set text(font: "New Computer Modern", lang: "es", size: 11pt)
#set par(leading: 0.55em, first-line-indent: 0pt, justify: true)
#show heading: set block(above: 1.4em, below: 1em)

// =====================
//  Resumen en castellano
// =====================

#align(center)[
  #text(size: 14pt, weight: "bold")[Resumen]
]

#v(0.5em)

#if version == "larga" [
  La distancia entre observaciones es un ingrediente central en casi todo
  algoritmo de clasificación supervisada. La euclídea ---elección canónica---
  es trivial de computar, pero su poder discriminativo decae con la dimensión:
  la distancia en el espacio ambiente deja de ser informativa, y no es claro
  cuál es el dominio sobre el cual sí lo sería.

  La _hipótesis de la variedad_ propone que tal dominio existe: las
  observaciones de interés en alta dimensión yacen sobre una variedad
  ---típicamente embebida, compacta, sin frontera y de dimensión intrínseca
  mucho menor--- en la que cobra sentido una distancia geodésica informativa.
  Conjeturar la existencia de la variedad no alcanza, sin embargo, si no se
  la conoce. Pelletier (2005) y Loubes & Pelletier (2008) muestran que,
  conocida la variedad, el estimador de densidad por núcleos ---y por ende
  el clasificador basado en él--- admiten una adaptación natural a este
  contexto.

  En esta tesis recorremos la literatura fundacional ---el problema de
  clasificación, las variedades de Riemann, y la estimación de densidad desde
  Parzen hasta el contexto de variedades compactas--- y la cerramos con un
  desarrollo más reciente: el aprendizaje de distancias y, en particular, la
  _distancia muestral de Fermat_ (Groisman, Jonckheere & Sapienza, 2022), que
  aproxima la geodésica directamente desde los datos, sin necesidad de
  conocer la variedad.

  Nuestro aporte es empírico: examinar el efecto neto de reemplazar la
  distancia euclídea por la de Fermat en clasificadores basados en densidad
  ---el de densidad por núcleos (KDC) y el de $k$ vecinos más cercanos
  ($k$-NN). La implementación de la distancia y los estimadores conforma una
  biblioteca de código abierto compatible con la interfaz de `scikit-learn`,
  disponible en #link("https://github.com/capitantoto/fermat").

  Sobre 20 _datasets_ que varían en dimensión ambiente, dimensión intrínseca
  y cantidad de clases observamos que: (i) en _datasets_ de muy alta
  curvatura, la distancia de Fermat ofrece ventajas claras sobre la euclídea;
  (ii) en regímenes bien muestreados, su parámetro adicional $alpha$ ---que
  pondera las aristas del grafo completo--- se vuelve funcionalmente
  intercambiable con el ancho de banda, y las dos distancias arrojan
  resultados equivalentes. Las conclusiones discuten líneas de trabajo
  futuro.
] else [
  La distancia entre observaciones es central en casi todo algoritmo de
  clasificación supervisada. La euclídea ---elección canónica--- pierde
  poder discriminativo en alta dimensión, y el dominio sobre el cual lo
  conservaría es desconocido.

  La _hipótesis de la variedad_ propone que tal dominio existe: las
  observaciones yacen sobre una variedad compacta, sin frontera y de
  dimensión intrínseca menor, donde la distancia geodésica recobra
  significado. Pelletier (2005) y Loubes & Pelletier (2008) muestran que,
  conocida la variedad, la estimación de densidad por núcleos admite una
  adaptación natural a este contexto.

  En esta tesis recorremos la literatura fundacional ---clasificación,
  variedades de Riemann, KDE desde Parzen hasta variedades compactas--- y
  la cerramos con un desarrollo más reciente: el aprendizaje de distancias
  y la _distancia muestral de Fermat_ (Groisman, Jonckheere & Sapienza,
  2022), que aproxima la geodésica directamente desde los datos.

  Nuestro aporte es empírico: examinar el efecto neto de reemplazar la
  euclídea por la de Fermat en clasificadores basados en densidad ---KDC y
  $k$-NN. La implementación es una biblioteca de código abierto compatible
  con `scikit-learn`
  (#link("https://github.com/capitantoto/fermat")).

  Sobre 20 _datasets_ observamos que la distancia de Fermat aventaja a la
  euclídea en regímenes de alta curvatura; en regímenes bien muestreados,
  su parámetro $alpha$ se vuelve funcionalmente intercambiable con el ancho
  de banda y los resultados son equivalentes.
]

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

#if version == "larga" [
  The distance between observations is a central ingredient in nearly every
  supervised classification algorithm. The Euclidean distance ---the
  canonical choice--- is trivial to compute, but its discriminative power
  decays with dimension: distance in the ambient space ceases to be
  informative, and the domain on which it would remain so is unclear.

  The _manifold hypothesis_ posits that such a domain exists:
  high-dimensional observations of interest lie on a manifold ---typically
  embedded, compact, boundary-free, and of much lower intrinsic dimension---
  on which a meaningful geodesic distance exists. Conjecturing the existence
  of the manifold, however, is of no use unless the manifold itself is
  known. Pelletier (2005) and Loubes & Pelletier (2008) show that, given
  knowledge of the manifold, the kernel density estimator ---and therefore
  the classifier based on it--- admit a natural adaptation to this setting.

  This thesis traces the foundational literature ---the classification
  problem, Riemannian manifolds, and density estimation from Parzen windows
  through the compact-manifold setting--- and closes with a more recent
  development: distance learning and, in particular, the _sample Fermat
  distance_ (Groisman, Jonckheere & Sapienza, 2022), which approximates the
  geodesic directly from the data, without requiring knowledge of the
  manifold.

  Our contribution is empirical: we examine the net effect of replacing the
  Euclidean distance with the Fermat distance in density-based classifiers
  ---the kernel density classifier (KDC) and $k$-nearest neighbours ($k$-NN).
  The implementation of the distance and the estimators is an open-source
  library compatible with the `scikit-learn` interface, available at
  #link("https://github.com/capitantoto/fermat").

  Over 20 datasets varying in ambient dimension, intrinsic dimension, and
  number of classes we observe that: (i) on highly curved datasets, the
  Fermat distance offers clear advantages over the Euclidean; (ii) in
  well-sampled regimes its additional parameter $alpha$ ---which weights
  the edges of the complete graph--- becomes functionally interchangeable
  with the bandwidth, and the two distances yield equivalent results. The
  conclusions discuss avenues for future work.
] else [
  The distance between observations is central to nearly every supervised
  classification algorithm. The Euclidean distance ---the canonical
  choice--- loses discriminative power in high dimensions, and the domain
  on which it would remain informative is unknown.

  The _manifold hypothesis_ posits that such a domain exists: observations
  lie on a compact, boundary-free manifold of much lower intrinsic
  dimension, on which the geodesic distance recovers meaning. Pelletier
  (2005) and Loubes & Pelletier (2008) show that, given the manifold,
  kernel density estimation admits a natural adaptation to this setting.

  This thesis traces the foundational literature ---classification,
  Riemannian manifolds, KDE from Parzen through the compact-manifold
  setting--- and closes with a more recent development: distance learning
  and the _sample Fermat distance_ (Groisman, Jonckheere & Sapienza, 2022),
  which approximates the geodesic directly from the data.

  Our contribution is empirical: we examine the net effect of replacing the
  Euclidean distance with the Fermat distance in density-based classifiers
  ---KDC and $k$-NN. The implementation is an open-source library
  compatible with `scikit-learn`
  (#link("https://github.com/capitantoto/fermat")).

  Over 20 datasets we observe that the Fermat distance outperforms the
  Euclidean in high-curvature regimes; in well-sampled regimes its
  parameter $alpha$ becomes functionally interchangeable with the
  bandwidth, and the results are equivalent.
]

#v(1em)

*Keywords:* supervised classification, kernel density estimation, Riemannian
manifolds, Fermat distance, density-based distances, non-parametric
learning.
