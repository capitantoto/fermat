// Plan de tesis — traducido de plan.tex (LyX 2.3.6)
// Estilo visual "LaTeX look" siguiendo tesis.typ

// ##############
// ### estilo ###
// ##############

// Basado en "How do I get the LaTeX look?",
// de https://typst.app/docs/guides/guide-for-latex-users/
#set page(margin: 1.75in, numbering: "1 de 1")
#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern", lang: "es")
#set heading(numbering: "1.")
#set strong(delta: 100)
#set math.equation(numbering: "(1)")

#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)
#show link: it => underline(text(it, fill: blue))

// ################
// # definiciones #
// ################
#let R = $bb(R)$
#let dimx = $d_x$
#let Rd = $bb(R)^(d_x)$
#let M = $cal(M)$
#let dimm = $d_(cal(M))$

// ##############
// # contenido  #
// ##############

#align(center)[
  #underline[*Proyecto de tesis*] \
  #underline[para optar al título de Magister en Estadística Matemática]
]

#v(0.5em)

*Nombre del postulante:* Lic. Gonzalo Barrera Borla

*Directora:* Dr. Pablo Groisman

*Tema de trabajo:* Distancia de Fermat en Clasificadores de Densidad Nuclear

*Lugar de trabajo:* Departamento de Matemática

= Antecedentes existentes sobre el tema

El concepto de _distancia_ entre las observaciones disponibles
es central a casi cualquier tarea estadística, tanto en descripción
como inferencia. Consideremos, por caso, un ejercicio de clasificación.
Sea $bold(x) = (x_i)_(i=1)^N$ una muestra de $N$
observaciones de vv.aa. i.i.d. ($X_1, dots, X_N : X_i tilde cal(L)(X) space forall space i in [N]$),
con $x_i in bb(R)^(d_x) space forall space i in [N] equiv {1, dots, N}$,
donde cada observación pertenece a una de $M$ clases $C_1, dots, C_M$
mutuamente excluyentes y conjuntamente exhaustivas.

Dada una nueva observación $x$ cuya clase es desconocida, ¿a qué clase
deberíamos asignarla? Cualquier respuesta a esta pregunta implicará
combinar toda la información muestral disponible, ponderando las $N$
observaciones de manera relativa a su cercanía o similitud con $x$.
Cuando el dominio de las $x_i$ es un espacio euclídeo $#Rd$,
es costumbre tomar la _distancia euclídea_ para cuantificar la
cercanía entre elementos. Así, por ejemplo, $k$-vecinos más cercanos
($k$-NN) asignará la nueva observación $x$ a la clase modal entre
las $k$ observaciones de entrenamiento más cercanas (es decir,
que minimizan $||x - dot.c||$).

Una dificultad bien conocida con los métodos basados en distancias
es la _maldición de la dimensionalidad_: a medida que la dimensión
$#dimx$ del espacio euclídeo en consideración crece, el espacio se
vuelve tan grande que todos los elementos de la muestra están indistinguiblemente
lejos entre sí; o lo que es equivalente, a igual $N$, la densidad
de observaciones en el espacio decae exponencialmente con $#dimx$.

En estos casos, es de suponer que el dominio de las $X$ no cubre
_todo_ $#Rd$, sino que éstas se encuentran embebidas en una
variedad $#M subset bb(R)^(d_x)$ cuya dimensión intrínseca $dim(#M)$
es potencialmente mucho menor a $#dimx$, y por ende la distancia
_en la variedad_ es más informativa que la distancia (euclídea)
en el espacio ambiente $#Rd$. A este supuesto se lo suele llamar
"hipótesis de la variedad" (_manifold hypothesis_), y suele
ser particularmente acertado cuando las observaciones provienen "del
mundo real" (e.g., imágenes, sonido y texto). Según Bengio et al. (2013),
_aprender_ la estructura de $#M$ a partir de $bold(x)$
es una forma (entre muchas) de _aprendizaje de representaciones_
(_representation learning_), donde la representación de $x_i$ en
base a sus coordenadas en $#M$ (en $bb(R)^d$) es tanto o
más útil que la representación original en $#Rd$ para tareas de descripción
e inferencia.

La ganancia en reducción de dimensionalidad con la hipótesis de la
variedad debe ser contrastada con la dificultad extra de tener que
trabajar en una variedad arbitraria $#M$ en lugar de $#Rd$, a priori
desconocida y que debemos estimar. Pelletier (2005) describe
un estimador "nuclear" para la función de densidad de vv.aa. i.i.d.
en variedades Riemannianas compactas sin borde, junto con resultados
de consistencia y convergencia; Henry y Rodríguez (2009) los amplían para
probar la consistencia uniforme fuerte y la distribución asintótica
de estos estimadores.

Tanto Pelletier (2005) como Henry y Rodríguez (2009) asumen que la
distancia geodésica es conocida. Trabajos recientes (Sapienza et al., 2018; Groisman et al., 2022; McKenzie y Damelin, 2019; Little et al., 2022)
proponen aprender una distancia geodésica $cal(D)_f^p$ entre
los nodos del grafo (aleatorio) completo de la muestra $bb(X)_n$#footnote[O por simplicidad de cómputo, su aproximación por el grafo de $k$-vecinos
más cercanos.], con cada arista pesada por una potencia $p$ de la distancia euclídea
entre sus extremos. En Sapienza et al. (2018), el uso de esta
distancia --- que los autores llaman "de Fermat", por su analogía
con el fenómeno óptico --- parece rendir considerables mejoras de _performance_
empírica en tareas de clasificación. Cuando $p = 1$, el estimador
de la distancia geodésica $cal(D)_f^1$ resultante es idéntico
al que usa _Isomap_ (Tenenbaum et al., 2000) para construir los
_embeddings_ de dimensión reducida.

= Naturaleza del aporte original sobre el tema y objetivos

Uniendo los elementos enunciados anteriormente, nos proponemos estudiar
sistemáticamente qué valor aporta el uso de una distancia basada en
datos (la distancia de Fermat $cal(D)_f^p$) frente a la
elección canónica (la distancia euclídea $||x - dot.c||$),
en el aprendizaje de _estimadores de densidad nuclear_ (KDEs,
por sus siglas en inglés). Nos proponemos luego comparar sus bondades
relativas usándolos en tareas de _clasificación_ bajo una amplia
gama de condiciones:

- en datasets "reales" y "sintéticos",
- en relación a la dimensión $d_x$ del espacio ambiente, y
- en relación a las $k$ categorías posibles para $Y in {C_1, dots, C_k}$.

Aprender un clasificador a partir de KDEs con distancia euclídea (Hastie et al., 2009, cap. 6.6)
es un método bastante eficiente en términos de cómputo. En cambio,
un cálculo exacto del estimador muestral de $cal(D)_f^p$
requiere $n^3$ pasos. La pregunta al respecto de su eficacia, entonces,
debe considerar además comparativamente los costos computacionales
de ambas distancias, que en datasets "grandes" podrían ser demasiado
altos para obtener ganancias de _performance_ relativamente menores.
Para poner en contexto la capacidad predictiva de estos clasificadores
y su costo computacional, incluiremos como métodos de referencia:

- clasificadores de _Naive Bayes_ (Hastie et al., 2009, cap. 6.6.3),
  que usan $d$ KDEs unidimensionales en lugar de un KDE $d$-dimensional
  por clase,
- _gradient boosting trees_ (GBTs, Hastie et al., 2009, cap. 10),
  un método reconocido en la actualidad por su simplicidad de uso y
  escasez de requerimientos, y
- _random forests_ (Hastie et al., 2009, cap. 15), que capturan
  buena parte de las bondades de los GBTs con una estructura sencilla.

Además, nos proponemos dar argumentos teóricos que garanticen la consistencia
de la estimación de la densidad en el estilo de Pelletier, cuando
se reemplaza la distancia geodésica por una estimación empírica (_plug-in_).
