#set document(title: "Respuesta al dictamen del jurado")
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt, lang: "es")
#set par(justify: true, leading: 0.62em, spacing: 0.9em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set block(above: 1.4em, below: 0.8em)

#let MM = $cal(M)$
#let XX = $bold(upright(X))$
#let fkdc = [$f$`-KDC`]
#let kdc = `KDC`
#let fkn = [$f$`-KN`]
#let kn = `KN`
#let svc = `SVC`
#let gbt = `GBT`
#let lr = `LR`
#let gnb = `GNB`

// TODO autor: este borrador lo redactó Claude (IA) a pedido del autor, a partir del dictamen, de la tesis y de los resultados posteriores a la entrega. Revisar y reescribir con voz propia antes de enviarlo. La sección final, «Nota para el autor», NO se envía: borrarla (o compilar solo hasta la firma).

#align(right)[Buenos Aires, 20 de noviembre de 2026]

#v(0.5em)
Dr. Matthieu Jonckheere \
Jurado de la tesis de Maestría en Estadística Matemática \
Facultad de Ciencias Exactas y Naturales, UBA

#v(0.5em)
*Ref.:* respuesta al dictamen sobre la tesis «Distancia de Fermat en clasificadores de densidad por núcleos».

#v(0.8em)
Estimado Matthieu:

Le agradezco la lectura atenta de la tesis y un dictamen tan generoso como preciso. Me alegra especialmente que haya valorado el §3.2.4, donde desarmo mi propio resultado favorable, porque sus tres observaciones van en la misma dirección: pedir que lo que la tesis afirma quede apoyado en un diseño que no deje lugar a dudas. Respondo a cada una por separado, indicando qué cambia en la versión final, qué hicimos después de la entrega y dónde, con respeto, discrepo. Las referencias a páginas y notas son las de la versión que usted leyó.

= Sobre el Resumen

En la valoración (§2) usted señala que el Resumen es algo más favorable que el cuerpo y remite a la «observación 1», pero la observación 1 del §3 trata del pseudocódigo; entiendo que la remisión quedó de una versión anterior del dictamen. Sobre el fondo tiene razón. El Resumen dice que «en datasets de alta curvatura y poca separación entre clases, clasificadores entrenados con distancia de Fermat aportan mejoras en términos de $R^2$ sobre sus pares con distancia euclídea, sin perder en exactitud», mientras que el Capítulo 4 acota esas mejoras a `helices_0` y `hueveras_0` y atribuye las de `lunas_lo` y `espirales_lo` a la regla de parsimonia. Con la réplica de 100 semillas que describo en la observación 3, además, la ventaja de #fkdc sobre #kdc en `helices_0` no se sostiene, y la mejora neta queda solo para #fkn. En la versión final, la oración dice:

#block(inset: (left: 1.5em, right: 1.5em))[_«En variedades muy curvas o ralamente muestreadas, el clasificador de $k$ vecinos con distancia de Fermat mejora sobre su par euclídeo, porque puede agrandar el vecindario sin cruzar a otra variedad; en el clasificador por núcleos, en cambio, las ventajas sobre la versión euclídea se explican por la interacción entre la regla de selección de hiperparámetros y el espacio de búsqueda ampliado.»_]

// TODO autor: la misma oración, traducida, ya está en el Abstract de docs/tesis.typ.

= Observación 1: pseudocódigo, costo y escalabilidad

Acepto la observación. La versión final incluye en el §2.1 un pseudocódigo (Algoritmo 1) que transcribe la implementación (`SampleFermatDistance` y `FermatKDE` en `fkdc/fermat.py`): en el entrenamiento, para cada clase $i$ se calcula una sola vez la matriz $A_i$ de caminos mínimos sobre el grafo completo de su muestra $XX_i$, con aristas $norm(q_a - q_b)^alpha$; para evaluar una observación nueva $x_0$ basta con un salto, $D_(Q_i, alpha)(x_0, q_b) = min_a (norm(x_0 - q_a)^alpha + A_i [a, b])$, con $Q_i = {x_0} union XX_i$. Como un camino mínimo no vuelve a pasar por $x_0$, el salto da exactamente la distancia en el grafo ampliado, sin aproximación; las geodésicas entre pares de $XX_i$ que $x_0$ pudiera acortar no se recalculan porque el clasificador no las usa. Aprovecho para corregir la redacción del §2.1, que habla de «recomputar únicamente las geodésicas que cambian»: lo que se calcula son las $N_i$ distancias desde $x_0$.

En cuanto al costo, con $N_i$ observaciones en la clase $i$ y $d$ atributos:
- *entrenamiento:* $O(N_i^2 d)$ para las aristas y $O(N_i^3)$ para los caminos mínimos entre todos los pares (Floyd--Warshall o Dijkstra desde cada nodo, ambos cúbicos en un grafo completo); memoria $O(N_i^2)$ por clase;
- *evaluación:* $O(N_i d + N_i^2)$ por clase y por observación nueva, y memoria auxiliar $O(N_i^2)$.

// TODO autor: verificar las cifras de tiempo medidas abajo antes de citarlas.
En la práctica, la mediana del tiempo por tarea completa (búsqueda en grilla con 5 pliegos, reajuste y evaluación) fue de 6,2 s para #fkdc y 8,5 s para #fkn, contra 1,4 s para #kdc; el dataset más caro fue `digitos`. Con $N <= 1797$ y clases de a lo sumo unos centenares de observaciones en entrenamiento, la memoria nunca fue una restricción. Tiene usted razón en que la tesis no discute la escalabilidad: la memoria cuadrática ($8 N_i^2$ bytes en doble precisión, unos 0,8 GB para $N_i = 10^4$ y 80 GB para $N_i = 10^5$) y el costo cúbico impiden usar el grafo completo más allá de unas decenas de miles de observaciones por clase. El camino para escalar ya figura en el trabajo futuro: reemplazar el grafo completo por el de $k$ vecinos con $k = O(2^(d_MM) ln N)$ (Chu et al.), o restringir los caminos a un subconjunto de puntos de referencia, como hace el paquete `fermat` de Sapienza. En la versión final agrego al §2.1 el párrafo de costo y remito desde allí a ese ítem del trabajo futuro.

= Observación 2: distancias entre clases de distinto tamaño

Es la observación más interesante del dictamen, y la tesis efectivamente no la discute. Separo tres preguntas: qué hace el clasificador, si el problema afectó los resultados informados y qué pasa cuando $alpha > 1$.

*Qué hace el clasificador.* La distancia de cada clase se calcula sobre el grafo de esa clase sola, $Q_i = {x_0} union XX_i$, y la regla compara $sum_(b) K_h (D_(Q_i, alpha)(x_0, q_b))$ entre clases: como se muestra en la deducción de la regla de Loubes y Pelletier (Ecuación 46 de la versión leída), el $N_i^(-1)$ de la densidad se cancela con el prior $hat(pi)_i = N_i slash N$. Esa cancelación se ocupa del _número_ de sumandos, pero no de su _escala_: por el Teorema de convergencia de $D_(Q, alpha)$, $D_(Q_i, alpha) approx N_i^(-beta) mu cal(D)_(f_i, beta)$, de modo que con un $h$ común la clase más numerosa ve sus observaciones más cerca y recibe, en efecto, una ventana más ancha. Su pregunta es pertinente.

*Los resultados informados no se ven afectados.* Con $alpha = 1$ la distancia de Fermat muestral sobre el grafo completo es exactamente la euclídea, y no depende de $N_i$. En `pinguinos` y `vino`, los dos datasets desbalanceados, la validación cruzada eligió $alpha = 1$ para #fkdc en las 25 semillas de `pinguinos`, `pinguinos_std` y `vino`, y en 24 de las 25 de `vino_std` (en la restante, $alpha = 3.5$). En la réplica con 100 semillas posterior a la entrega, $alpha = 1$ en 100 de 100 (`pinguinos`, `pinguinos_std`, `vino`) y 98 de 100 (`vino_std`). Las cifras de esos datasets en la tesis son, por lo tanto, las de un #kdc, y el efecto que usted señala no las alcanza.

*Qué pasa con $alpha > 1$.* Para contrastarlo como usted sugiere, fijamos $alpha in {2, 3, 4}$, elegimos $h$ por validación cruzada y comparamos, sobre las mismas 100 particiones de `pinguinos_std` (151 Adelie, 68 Chinstrap, 123 Gentoo), la regla de la tesis con dos variantes: la reescalada por $N_i^beta$ (con $d_MM$ desconocida usamos la dimensión ambiente, $beta = (alpha - 1) slash 4$) y una con el grafo de cada clase submuestreado al tamaño de la menor, manteniendo los priors. Con la regla de la tesis, la sensibilidad de la clase minoritaria cae de 0,934 ($alpha = 1$) a 0,915, 0,901 y 0,880; con el reescalado se mantiene en 0,945, 0,940 y 0,932, y mejora en 63, 75 y 82 de las 100 particiones (empeora en 1, 1 y 3). El submuestreo queda en un punto intermedio. La exactitud global apenas cambia (de 0,984 a 0,979 con $alpha = 3$). En `vino`, con un desbalance mucho menor (59, 71 y 48), no aparece ningún efecto sistemático hasta $alpha = 3$, y con $alpha = 4$ el reescalado empeora. En `pinguinos_std` el sesgo hacia la clase mayor existe, es pequeño y crece con $alpha$, y el reescalado que usted propone lo corrige.

// TODO autor: el control sintético que sigue está en la «Nota para el autor»; confirmar las cifras antes de enviarlo.
Como en esos datos no conocemos la regla de Bayes, lo contrastamos además en un caso donde sí: dos clases sobre la circunferencia unidad ($d_MM = 1$), con densidades de von Mises opuestas ($kappa = 1.5$) y 400 contra 100 observaciones de entrenamiento, 40 réplicas. Con $alpha = 1$ el exceso de riesgo sobre Bayes es 0,001 en todas las variantes. Con la regla de la tesis y $alpha = 2$ y $3$, el exceso sube a 0,041 y 0,075, y el clasificador asigna a la clase mayor 11 y 16 puntos porcentuales de la masa que la regla de Bayes asigna a la menor: la frontera se desplaza hacia la clase minoritaria, exactamente el sesgo que usted anticipa. Reescalando por $N_i^beta$ con la dimensión intrínseca correcta, el exceso baja a 0,003 y 0,017, con una leve corrección de más (2 y 5 puntos hacia la clase menor); submuestrear el grafo manteniendo los priors lo reduce solo a la mitad, y submuestrear también los priors sesga hacia la clase menor (9 a 10 puntos). Con clases balanceadas (400 y 400), las variantes prácticamente coinciden. Su observación, entonces, es correcta: con tamaños de muestra como los de la tesis y $alpha > 1$, #fkdc no es indiferente al desbalance. Que no haya afectado los resultados informados se debe a que la validación cruzada eligió $alpha = 1$ precisamente en los datasets desbalanceados; en la versión final lo agrego al §2 como limitación del método, con este contraste, y el reescalado por $N_i^beta$ (que exige estimar $d_MM$) pasa al trabajo futuro.

*Datasets con ruido.* Tiene razón en que varios datasets quedan fuera de las hipótesis. Pelletier supone una variedad compacta sin frontera, y ni las variantes con ruido ni varias sin ruido (las lunas o las hélices, que tienen extremos) lo son; con ruido de dimensión plena, además, la variedad que «ve» la distancia de Fermat a la escala del ruido es de dimensión $d$, no $d_MM$. La tesis ya reconoce que el estimador que de hecho entrenamos no está cubierto por ningún teorema; esos datasets funcionan como pruebas de estrés empíricas, no como ilustraciones del teorema, y conviene decirlo explícitamente al presentarlos.
// TODO autor: la versión final de docs/tesis.typ todavía no agrega esa aclaración; decidir dónde va (p. ej., al presentar los datasets con ruido en §3.2).

// TODO autor: confirmar contra Groisman, Jonckheere y Sapienza (2022) qué hipótesis exactas (soporte compacto, densidad acotada inferiormente) violan el ruido gaussiano y el tubo de radio $tau$.

= Observación 3: equidad del diseño de hiperparámetros

*Grillas anidadas.* Después de la entrega repetimos el experimento exactamente como usted propone, con 100 semillas en lugar de 25 y un #kdc que evalúa la densidad sin la aproximación de árbol de `scikit-learn` (que comete errores en las colas profundas). La grilla de #kdc pasó a ser $h = 10^(-5 + k slash 12)$, $k = 0, dots, 132$ (133 valores), y la de #fkdc, uno de cada tres de esos valores, que coincide bit a bit con los 45 de la tesis; así, la grilla de #kdc contiene a la de #fkdc con $alpha = 1$. Guardamos además todos los resultados de la validación cruzada para comparar dos reglas de selección: la de parsimonia de la tesis y una «ciega a $alpha$», que elige el mejor $h$ para cada $alpha$ con la misma regla y solo acepta un $alpha > 1$ si supera al mejor con $alpha = 1$ por más de un desvío estándar.

El resultado convierte el hallazgo ambiguo en uno limpio, en la dirección que el §3.2.4 ya sugería:
- con la regla de la tesis, #fkdc supera a #kdc en $R^2$ en `lunas_lo` en 73 de 100 semillas, sin ninguna derrota (27 empates; diferencia mediana $+0.017$), y en `espirales_lo` en 53, con 11 derrotas y 36 empates ($+0.028$), aunque eligió $alpha = 1$ en 100 y 94 semillas respectivamente;
- con la selección ciega a $alpha$, la ventaja desaparece: 100 empates exactos en `lunas_lo` y, en `espirales_lo`, 93 empates y 7 derrotas de #fkdc;
- en exactitud, #fkdc y #kdc son equivalentes en ambos datasets con cualquiera de las dos reglas.

La ventaja no se debía, entonces, al desalineamiento de las grillas ni a una validación cruzada «aleatoriamente favorable», como conjetura la nota 125, sino a un efecto sistemático: el umbral de la regla de parsimonia se calcula sobre toda la grilla, y las filas con $alpha > 1$ lo desplazan, de modo que #fkdc termina eligiendo en $alpha = 1$ un $h$ distinto (menor) del que elige #kdc. Es la «simbiosis» entre $h$ y $alpha$ del Capítulo 4, ahora aislada. En la versión final reemplazo la conjetura de la nota 125 por este resultado y reformulo la conclusión correspondiente.

*La versión final usa la réplica.* Todos los resultados de la versión final provienen de esta réplica, restringida a los ocho clasificadores y los 20 datasets de la tesis (más las cuatro variantes estandarizadas), con el #kdc exacto. La gran mayoría de las afirmaciones se sostiene; cambian tres, y prefiero decírselo. Primero, la ganancia de #fkdc sobre #kdc en `helices_0` que usted destaca (Tabla 5) no se sostiene: con la densidad evaluada en forma exacta, #kdc también elige ventanas muy chicas, y #fkdc gana en 43 semillas, pierde en 32 y empata en 25 (medianas de 0,968 y 0,964). La mejora neta atribuible a la distancia de Fermat queda para #fkn, que supera a #kn en 83 de las 100 semillas de `helices_0` (medianas de 0,851 y 0,688). Segundo, la afirmación de que la distancia de Fermat aumenta la varianza entre repeticiones tampoco se sostiene: en `helices_0`, #fkn es mucho más estable que #kn. Tercero, en `pionono_0` #fkdc sí supera a #kdc, pero eligiendo $alpha = 1$ en todas las semillas, es decir, otra vez por la regla de parsimonia.

*Presupuesto de búsqueda de los demás clasificadores.* No fue comparable, y la tesis debería decirlo. Las grillas de la corrida principal (`fkdc/config.py`) tienen:

#align(center, table(
  columns: 3,
  align: (left, left, right),
  stroke: none,
  table.hline(),
  [*Clasificador*], [*Hiperparámetros*], [*Configuraciones*],
  table.hline(stroke: 0.5pt),
  [#fkdc], [13 $alpha$ × 45 $h$], [585],
  [#fkn], [13 $alpha$ × 2 ponderaciones × 12--19 $k$], [312--494],
  [#kdc], [136 $h$ (133 en la versión final)], [136],
  [#svc], [61 $C$ × 2 reglas para $gamma$ (núcleo RBF)], [122],
  [#gnb], [40 suavizados de varianza], [40],
  [#lr], [36 $C$], [36],
  [#kn], [2 ponderaciones × 12--19 $k$], [24--38],
  [#gbt], [3 tasas de aprendizaje × 4 profundidades], [12],
  table.hline(),
))

(La cantidad de valores de $k$ depende del tamaño del pliego de entrenamiento de cada dataset.) Las grillas de referencia son entonces más chicas, en particular la de #gbt. Dos matices, sin embargo. Primero, la asimetría favorece a nuestros clasificadores, y así hay que leer la afirmación de que la familia de núcleos es competitiva; aun con esa ventaja, #svc lidera en exactitud en 7 datasets y #gbt domina los datasets con dimensiones de ruido. Segundo, el tamaño de la grilla no parece decisivo: después de la entrega comparamos también con TabPFN, sin ninguna búsqueda de hiperparámetros, y quedó en el grupo líder en 19 de 33 datasets. En la versión final agrego esta tabla al §2.4.4 («Entrenamiento de los algoritmos») y explicito la asimetría.

Al revisar las grillas encontré además una imprecisión en el §2.4.2, que dice que #svc «se evaluó en dos variantes: con núcleos lineales y RBF». Los experimentos informados usan solo el núcleo RBF; un núcleo lineal se probó en una ronda exploratoria y se descartó por rendir en general peor. La versión final lo dice así.

#v(1em)
Le agradezco nuevamente el tiempo y el cuidado. Las tres observaciones mejoran la tesis, y la segunda me deja una pregunta que pienso seguir.

#v(1em)
Atentamente,

#v(2em)
Lic. Gonzalo Barrera Borla

#pagebreak()

#heading(numbering: none)[Nota para el autor (no enviar)]

#text(size: 10pt)[
Redactada por Claude (IA). Resume lo verificado en código y datos, con fuente.

*Regla que implementa el código.* `KDClassifier.fit` (fkdc/fermat.py:226--244) separa la muestra por clase y ajusta un `FermatKDE` por clase; `SampleFermatDistance` construye el grafo de cada clase sola (fermat.py:55--57). `FermatKDE.score_samples` (fermat.py:149--161) devuelve $-log N_i - d log h - (d slash 2) log 2 pi + log sum_b exp(-1/2 (D_(Q_i, alpha)(x, q_b) slash h)^2)$, con $d$ la dimensión _ambiente_ (común a todas las clases, se cancela). `predict_proba` (fermat.py:246--254) suma $log(N_i slash N)$ y normaliza. Resultado: $hat(Pr)(i | x) prop sum_b exp(-1/2 (D_(Q_i, alpha)(x, q_b) slash h)^2)$. Tu intuición («el prior cancela el $N_i^(-1)$») es correcta, pero solo resuelve el conteo; no toca la escala $N_i^(-beta)$ de las distancias dentro del núcleo, que es lo que plantea el jurado. Con $alpha = 1$ desaparece ($beta = 0$ y la distancia es euclídea).

*Argumento heurístico, refutado en la práctica por el control de abajo (no es una prueba; está en `docs/sandbox/fermat-classifier-theory.md`, Prop. 1.3 y 3.4, también borrador IA).* Si $D_(Q_i, alpha)(x, y) approx mu N_i^(-beta) f_i (x)^(-beta) norm(x - y)$ en una ventana mesoscópica, la suma de núcleos de la clase $i$ es $approx C h^(d_MM) (N_i f_i (x))^alpha$ con $C$ común a todas las clases (si todas tienen la misma $d_MM$). Entonces el argmax coincide con el de $hat(pi)_i f_i$ (consistencia «accidental») y las probabilidades quedan templadas a la potencia $alpha$. El reescalado por $N_i^beta$ lleva a maximizar $hat(pi)_i f_i^alpha$, inconsistente, con sesgo hacia la clase _menor_; lo mismo el grafo submuestreado con priors. Supuestos frágiles: convergencia uniforme de Groisman et al. en ventanas que se achican (abierto), igual $d_MM$ por clase, $h_"eff"$ chico frente a la escala de variación de $f$.

*Lo que dicen los datos (scratchpad `obs2/`, 100 particiones, $h$ por CV de log-verosimilitud, sin regla de parsimonia, $alpha$ fijo).* `pinguinos_std`: sensibilidad de Chinstrap A (tesis) / B (reescalado, $d = 4$) / C (grafo balanceado): $alpha = 2$: 0,915 / 0,945 / 0,928; $alpha = 3$: 0,901 / 0,940 / 0,920; $alpha = 4$: 0,880 / 0,932 / 0,901. Cuota predicha de Chinstrap con $alpha = 3$: 0,182 / 0,194 / 0,189. `vino_std`: sin efecto sistemático hasta $alpha = 3$; con $alpha = 4$ B empeora (exactitud 0,858 → 0,833). Control: la variante A reproduce `KDClassifier(metric="fermat")` a $10^(-15)$.

*Control con Bayes conocido (`obs2/circulo.py`, `circulo.csv`).* Círculo unidad, von Mises $kappa = 1.5$ centradas en $0$ y $pi$, entrenamiento 400/100 (y 400/400 de control), evaluación 2000 de la mezcla, 40 semillas, $h$ por CV estratificada de 5 pliegos con log-verosimilitud. Variantes: A tesis; B reescalado con $d = 1$; C grafo balanceado con priors $N_i slash N$; D todo balanceado (priors uniformes). Exceso de riesgo $EE[abs(2 eta - 1) bb(1)("pred" != "Bayes")]$ y diferencia de cuota minoritaria respecto de Bayes: $alpha = 1$: 0,0013 / $-0.010$ en A y B. $alpha = 2$: A 0,041 / $-0.108$; B 0,003 / $+0.021$; C 0,023 / $-0.062$; D 0,033 / $+0.090$. $alpha = 3$: A 0,075 / $-0.165$; B 0,017 / $+0.051$; C 0,046 / $-0.100$; D 0,046 / $+0.104$. Balanceado 400/400: A, C y D coinciden (exceso 0,0005 a 0,0028); B igual salvo $alpha = 3$ (0,0052).

*Lectura honesta.* Tu intuición (la cancelación prior × $N_i^(-1)$ resuelve el problema) no se sostiene: resuelve el conteo, no la escala. El argumento asintótico de la nota del sandbox (argmax de $(hat(pi)_i f_i)^alpha$ = argmax de $hat(pi)_i f_i$, «consistencia accidental», C2 en `fermat-classifier-theory.md`) queda refutado en la práctica: con $N_i$ de 100 a 400 y $h$ elegido por CV, la regla de la tesis desplaza la frontera hacia la clase menor, y el reescalado del jurado casi lo arregla (con leve sobrecorrección, que sí es lo que el argumento predice para B). Mi lectura (no verificada): la CV elige un $h$ con el que la ventana efectiva de la clase mayor, $prop N_i^beta h$, sale de la ventana mesoscópica y sobresuaviza. Por eso saqué del borrador la conjetura asintótica: no la defiendas ante el jurado. La defensa de fondo es: (1) el jurado tiene razón; (2) no afectó ningún resultado informado porque la CV eligió $alpha = 1$ en `pinguinos` y `vino`; (3) en los sintéticos de dos clases (balanceados) las variantes coinciden; (4) la corrección exige $hat(d)_MM$. Pendiente: corregir C2 en la nota del sandbox; ver si la CV sobre $alpha$ elige $alpha = 1$ en el círculo desbalanceado (no lo medí).

*Grillas.* Campaña heroica (`sandbox/heroica/plan.md` ll. 27 y 112): $h_"kdc" = 10^(-5 + k slash 12)$, 133 valores; $h_"fkdc" = h_"kdc"[::3]$ bit a bit igual a `np.logspace(-5, 6, 45)`. Veredictos C03/C04 en `replica-heroica.csv` (regla de la tesis) y `replica-heroica-seleccion-v6.csv` (ciega a $alpha$). Con $R^2$ templado la ventaja con la regla de la tesis sobrevive pero es mínima (`lunas_lo` 53--20, mediana $+0.0006$).

*Costo.* Tiempos de `sandbox/heroica/plan.md` l. 78 (v5, M1). Ineficiencia de implementación, no de método: en la búsqueda en grilla, `score_samples` recalcula las distancias para cada $h$ (solo `fermat_muestral` está en caché), y el bucle por observación de `_distancia_muestral` es Python puro.
]
