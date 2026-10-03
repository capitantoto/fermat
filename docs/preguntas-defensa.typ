#set document(title: "Preguntas posibles para la defensa")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt, lang: "es")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set block(above: 1.4em, below: 0.8em)

#let bu(x) = $bold(upright(#x))$

#let pregunta(p, r) = block(breakable: false, above: 1.1em)[
  #text(weight: "bold")[#p]
  #block(inset: (left: 1em), above: 0.5em)[#r]
]

#align(center)[
  #text(size: 16pt, weight: "bold")[Preguntas posibles para la defensa]

  #text(size: 10pt, style: "italic")[Borrador redactado con asistencia de IA (Claude) a pedido del autor, a partir de la tesis y de los resultados posteriores a la entrega (`sandbox/heroica`, `sandbox/fundacionales`). Las respuestas son puntos de partida: revisarlas y reescribirlas con voz propia antes de usarlas.]
]

= Sobre el método

#pregunta[¿Por qué la distancia de Fermat y no las geodésicas del grafo de vecinos, como en Isomap?][
  La distancia de Fermat incorpora la densidad: el funcional $integral_gamma f^(-beta)$ abarata los caminos por zonas densas, y su versión muestral converge a la macroscópica (Groisman, Jonckheere y Sapienza). Isomap aprende solo la geometría. Después de la entrega lo probamos: Isomap seguido de $k$ vecinos nunca superó a #raw("fkdc") en las variedades (en `espirales_lo`, $R^2$ mediano 0.66 contra 0.87).
]

#pregunta[¿El clasificador #raw("fkdc") es consistente?][
  No lo sabemos, y la tesis lo dice. Pelletier supone conocida la distancia de Riemann; Groisman et al. garantizan la convergencia de la distancia muestral. Falta un resultado que combine ambos para el núcleo evaluado sobre $D_(Q, alpha)$. Además, tomamos la densidad de volumen $theta equiv 1$ con un argumento cualitativo.
]

#pregunta[¿Por qué $alpha$ y $h$ se solapan?][
  Localmente, la distancia de Fermat muestral es proporcional a la euclídea, y el núcleo se evalúa sobre $d slash h$: lo que $alpha$ cambia en $d$ se puede compensar con $h$. En datasets bien muestreados aparece un «risco» $log h prop alpha$ de parametrizaciones equivalentes, y la regla de $1 sigma$ elige $alpha = 1$. Solo cuando la curvatura es comparable a la separación entre observaciones (`helices_0`, `hueveras_0`) aparece un $alpha > 1$ que ningún $h$ sustituye.
]

#pregunta[¿Cómo se evalúa un punto nuevo, si la distancia es muestral?][
  Para cada clase se agrega el punto a la muestra de esa clase, $Q_i = {x_0} union {x_j : g_j = g_i}$, y se calcula $D_(Q_i, alpha)(x_0, dot)$. El costo es el de caminos mínimos sobre el grafo completo; la tesis propone, como trabajo futuro, el grafo de $k$ vecinos con el $k$ de Chu et al.
]

#pregunta[¿Por qué el $R^2$ de McFadden y no la exactitud?][
  Los clasificadores de la tesis son suaves: el $R^2$ mide la calidad de las probabilidades, no solo la de la clase elegida. Por exactitud la ventaja de la familia de núcleos se diluye y `SVC` es casi imbatible; la tesis informa ambas.
]

#pregunta[¿La regla de $1 sigma$ no sesga la comparación entre #raw("fkdc") y #raw("kdc")?][
  Sí, y la tesis lo muestra: en `lunas_lo` y `espirales_lo` la ventaja de #raw("fkdc") viene de la interacción entre la regla de parsimonia y la grilla ampliada. En la réplica con 100 semillas guardamos la grilla completa y derivamos también una selección ciega a $alpha$: 113 de 132 afirmaciones replican con ella, contra 117 con la regla de la tesis.
]

= Sobre los resultados

#pregunta[¿La ventaja de #raw("fkdc") sobre #raw("kdc") en $R^2$ es real o es calibración?][
  En parte es calibración: #raw("kdc") es poco confiado, y el exponente $alpha$ agudiza las probabilidades. Con el $R^2$ templado (cada clasificador recalibrado por temperatura), la ventaja desaparece en `anteojos`, `digitos` y `mnist`, pero sobrevive en `lunas_lo` y `espirales_lo`. Comparamos siempre crudo con crudo y templado con templado.
]

#pregunta[¿Qué pasó con el error de `KernelDensity`?][
  Decirlo antes de que lo pregunten. El `KernelDensity` de `scikit-learn` calcula mal las log-densidades en las colas profundas, y la tesis lo usó para #raw("kdc"). Con un #raw("kdc") exacto, la ventaja de 5 pp de #raw("fkdc") en `helices_0` deja de ser significativa (43–32–25); #raw("fkdc") y #raw("fkn") nunca usaron ese código, así que solo cambian las comparaciones contra #raw("kdc"). Con 100 semillas y #raw("kdc") exacto, la ventaja de #raw("fkdc") en `lunas_lo` y `espirales_lo` se sostiene.
]

#pregunta[¿Las semillas son observaciones independientes?][
  En los datasets sintéticos y en `mnist`, sí: cada semilla es una muestra nueva. En `iris`, `vino`, `pinguinos`, `digitos` y `anteojos`, cada semilla es otra partición de los mismos datos, y los p-valores salen optimistas; esos datasets se leen de forma descriptiva.
]

#pregunta[¿Por qué no estandarizar siempre?][
  Porque depende de los datos: en `pinguinos` (gramos frente a milímetros) estandarizar llevó a #raw("fkdc") de 0.42 a 0.96 de $R^2$; en `digitos`, de escala homogénea, lo empeoró de 0.98 a 0.87. Después de la entrega probamos la generalización natural: aprender un ancho de banda por variable (la diagonal $bu(H) in cal(D)$ del trabajo futuro).
]

#pregunta[¿Qué se puede hacer con las dimensiones de ruido?][
  Con un ancho por variable elegido por verosimilitud dejando uno afuera (Hall, Racine y Li, 2004), las variables de ruido reciben anchos de 3.5 a 4.9 veces mayores que las de señal y quedan suavizadas: en `helices_12`, el $R^2$ pasa de 0 a 1.
]

#pregunta[¿Por qué 25 semillas?][
  Por costo. La réplica muestra que varias afirmaciones estaban al límite: para `espirales_lo` hacían falta unas 95 semillas para un 80 % de potencia. Por eso la réplica usó 100, con el prerregistro fijado antes de mirar resultados.
]

= Sobre el contexto

#pregunta[¿Por qué no compararon con métodos modernos, como TabPFN?][
  En la tesis no; después de la entrega, sí. TabPFN-3.5 queda en el grupo líder en 19 de 33 datasets; #raw("fkdc"), en 2 dentro de ese conjunto ampliado. TabPFN trae un prior aprendido de millones de tablas simuladas: aproxima la predictiva posterior bayesiana. Los métodos de la tesis aprenden de 400 puntos.
]

#pregunta[Si TabPFN gana, ¿para qué sirve la distancia de Fermat?][
  Para entender y para garantizar. #raw("fkdc") hace explícita la geometría que usa, no depende de un prior simulado ni de pesos preentrenados, y localiza por construcción. Nagler (2023) muestra que el sesgo de un PFN solo se anula si la predicción se localiza alrededor del punto de prueba, y que la arquitectura actual no lo garantiza. Además, métodos clásicos con anchos por variable o un superlearner superan a TabPFN en los datasets con ruido.
]

#pregunta[¿TabPFN es consistente?][
  No hay una garantía general: la red está fija y su sesgo no tiene por qué desaparecer con $n$. Su fortaleza es la optimalidad bayesiana promedio bajo su prior, con muestras chicas.
]

#pregunta[¿Cuándo recomendaría #raw("fkdc") o #raw("fkn")?][
  Con variedades muy curvas o ralamente muestreadas, atributos en una escala común y pocas dimensiones de ruido. #raw("fkn") tiene una ventaja práctica: es robusto a la elección de $k$, porque la distancia de Fermat permite agrandar el vecindario sin cruzar a otra variedad.
]
