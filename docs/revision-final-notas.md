# Revisión final de `docs/tesis.typ` — notas por bloque

Convenciones: **Aplicado** lista solo ediciones de fondo (las erratas no se listan); **Para decidir** son preguntas y sugerencias no aplicadas; **Estado** son convenciones que los bloques siguientes respetan.

## Bloque 1 — Carátula, Resumen, Abstract, Introducción

**Aplicado**
- Ninguna edición de fondo. Erratas: «igualmente sensible … que» → «tan sensible … como» (Resumen); «is just sensitive … as the euclidean» → «is just as sensitive … as the Euclidean one» (Abstract); «a priori» en cursiva en el Resumen, como en el cuerpo.

**Para decidir**
- Carátula: «Fecha de presentación del ejemplar: 19 de mayo de 2026». Si esta versión reemplaza a la entregada, verificá la fecha.
- Carátula: «Lugar de Trabajo» y «Fecha de Defensa» llevan mayúscula interna; la RAE pide «Lugar de trabajo», «Fecha de defensa». No lo toqué por si el formato lo fija la Facultad.
- Introducción: la cita del sitio de la materia conserva «sólo» con tilde y «bienvenides!» sin «¡» de apertura; es cita textual, la dejé intacta.

**Estado**
- Locuciones latinas no adaptadas («a priori», «in toto», «alla») en cursiva.
- Referencias a capítulos: `@preliminares`, `@propuesta-original`, `@resultados`, `@conclusiones` se leen «la Sección N»; sin cambios.

## Bloque 2 — Vocabulario y Notación

**Aplicado**
- Nota B: entradas reordenadas por primera aparición en el cuerpo. Eliminadas (no aparecen o son triviales): `∅`, medida de Lebesgue `λ(x)`, notación de flecha `a ↦ b`, proporcionalidad `∝`, «c.s.». Conservada `overline(S)` por su doble uso (clausura y segmento).
- Nota H: la sección usaba `p` como dimensión ambiente (siguiendo a Hastie) mientras el cuerpo usa `d`. Cambiado a `d` en las entradas (`RR^d`, `H ∈ RR^(d×d)`, `X ∈ RR^(N×d)`) y en la prosa: «el conjunto de $N$ vectores $p$-dimensionales … matriz #XX de dimensión $N times p$» → `d`; «el $p-$vector de inputs» → «el $d$-vector». Quedan dos `N times p` / `RR^p` en la sección de PCA y en la definición de Isomap (bloque 5), a resolver allí.
- Entrada `K`: agregado «también, el número de clases del problema de clasificación (el contexto desambigua)», porque el cuerpo usa `K` con ambos significados sin aviso.
- Erratas: notas al pie con mayúscula y punto (nota F); «Riemanniana» → «riemanniana» (adjetivo, minúscula); `$p-$vector`/`$N-$vector` (guion dentro del modo matemático se ve como signo menos) → `$d$-vector`, `$N$-vector`; «e.g.:» → «p. ej.,»; «norma euclídea del elemento $x$» (el símbolo era `‖·‖`) → «de un vector de $RR^d$»; `h ∈ RR` → `h ∈ RR_+`.

**Para decidir**
- Ninguno.

**Estado**
- Adjetivos derivados de nombre propio en minúscula: riemanniana, euclídea, gaussiana. El cuerpo tiene ~30 «Riemanniana/o» (bloque 4) y «Gaussianas» (bloque 5); se corrigen allí, sin tocar títulos en inglés.
- Dimensión ambiente `d` en todo el texto; `p` solo si reproduce la notación de un paper con aviso.

## Bloque 3 — Preliminares 2.1–2.3 (clasificación, KDE, KDE multivariado, maldición de la dimensionalidad, hipótesis de la variedad)

**Aplicado**
- Regla de Bayes, reescritura de $hat(G)$: las líneas 3 y 4 igualaban una probabilidad condicional a un máximo no normalizado. Antes: «$Pr(g|X=x) = max_(g in GG) Pr(X=x|g) times Pr(g)$» y «$Pr(GG_k|X=x) = max_(k in [K]) Pr(X=x|GG_k) times Pr(GG_k)$». Ahora el lado izquierdo es también «$Pr(X=x|g) times Pr(g)$» (resp. «$Pr(X=x|GG_k) times Pr(GG_k)$»).
- EPE: «Por la ley de la probabilidad total, podemos condicionar a X» → «Por la ley de la esperanza total»; y en la cadena $"EPE"(f) = …$ el argumento era $hat(G)(X)$ en lugar de $f(X)$ (tres líneas). También «la esperanza es contra la distribución conjunta» → «respecto de».
- Observación tras la def. de núcleo: «Todas las funciones de densidad simétricas centradas en 0 son núcleos» → «simétricas y unimodales centradas en 0» (la condición 4 exige máximo en 0).
- Elección de $HH$: «$mat(d; 2) = (d^2 - d) slash 2$ parámetros si $HH in cal(F)$» → «$binom(d + 1, 2) = (d^2 + d) slash 2$»: una matriz simétrica $d times d$ tiene $d(d+1)/2$ parámetros libres.
- Densidad normal multivariada: «$Phi(x) : RR^d -> RR = (2 pi)^(-d/2) exp(-(||x||^2)/2)$» → «$phi.alt : RR^d -> RR, quad phi.alt(x) = …$» ($Phi$ es la notación usual de la función de distribución; $phi.alt$ ya se usó para la densidad normal estándar unidimensional).
- Clasificador duro: «$arg max_(i in [K]) hat(G)_"Blando"(x_0)$» → «$arg max_(i in [K]) [hat(G)_"Blando"(x_0)]_i$»; en la observación siguiente, la suma del denominador reutilizaba el índice $i$ del numerador → índice $j$.
- Nota H: Naive Bayes usaba $p$ como dimensión ($X_1, dots, X_p$; $p$ densidades; $K times p$; $product_(k=1)^p$) → $d$; «$X_1, dots, X_k$» → «$X_1, dots, X_d$».
- Nota al pie sobre 8 bits: «$sop X = B^8000 = 2^64000$» confundía conjunto y cardinal → «$sop X = B^8000$ y $abs(B^8000) = 2^64000$».
- Nota G aplicada a las 20 ecuaciones destacadas del bloque. Nota F aplicada a 14 notas al pie.
- Erratas: coma sujeto–predicado (×3); «Regla de Bayes» → «regla de Bayes»; «plugin» → _plug-in_; «SVMs», «KDEs» → «SVM», «KDE» (las siglas no pluralizan); «data genómica» → «datos genómicos»; «$d-$variado» → «$d$-variado»; «no-lineal», «no-euclídeo», «no-supervisada» → sin guion; «$d u$» → «$dif u$»; «$GG_1, dots, GG_k$» → «$GG_K$»; desigualdad $<$ / $<=$ unificada en $U_h$; «interconexiones - es decir» → raya.

**Para decidir**
- §2.1 «Definición y vocabulario»: la primera línea de «$hat(G)(x) = arg min_f EE(L(G, f(X)))$» repite el mínimo global antes del punto a punto; podría leerse «$= arg min_(g) EE(L(G, g) | X = x)$». No lo cambié: el paso siguiente ya lo hace explícito.
- §2.3.5 (hipótesis de la variedad): la nota sobre Gallese y Bengio sigue siendo larga (ya señalado en `revision-editorial.md`); intacta.

**Estado**
- Puntuación de ecuaciones: signo dentro del `$ $`, sin `thin`; «donde/con» en minúscula tras coma.
- Siglas sin plural (KDE, SVM, PCA). Prefijo «no» sin guion (quedan 8 «no-…» en bloques posteriores).
- $phi.alt$ = densidad normal (estándar); $Phi$ no se usa.

## Bloque 4 — Preliminares 2.4 (variedades de Riemann, geodésicas, probabilidad y KDE en variedades, densidad de volumen)

**Aplicado**
- Def. de espacio topológico, axioma 2: el ejemplo reutilizaba $X$ (el conjunto total) como abierto genérico: «$X in T, Y in T => X inter Y in T$» → «$U in T, V in T => U inter V in T$». Def. de entorno: «$(X,Τ)$» usaba una tau griega mayúscula en lugar de $T$.
- Homeomorfismo: «es una función $phi$ entre dos espacios topológicos si es biyectiva y tanto ella como su inversa son continuas» (condición colgante) → «que es biyectiva y tal que tanto ella como su inversa son continuas».
- Observación $MM = RR^d$: «La base canónica de $T_p RR^d$ formada por las columnas de $bu(I)_d$ es una matriz positiva definida» (una base no es una matriz) → «La matriz identidad $bu(I)_d$, cuyas columnas forman la base canónica de $T_p RR^d = RR^d$, es definida positiva».
- Partición de la unidad: «Sea entonces: $[ecuación]$ es posible verificar…» (oración sin verbo principal) → «Definiendo entonces $[ecuación],$ es posible verificar…».
- Mapa exponencial: «$exp_p (v) : T_p MM -> MM = gamma_(p,v)(1)$» → «$exp_p : T_p MM -> MM, quad exp_p (v) = gamma_(p,v)(1)$» (misma corrección de forma que en la normal multivariada del bloque 3).
- Observación tras el núcleo isotrópico: «Todo núcleo válido en @kde-mv también es un núcleo isotrópico» es falso (un núcleo producto no es radial); invertido a «Todo núcleo isotrópico es también un núcleo válido según @kde-mv», que es lo que se usa.
- Nota H: en la observación sobre variedades con frontera, «$n$-variedad», «dimensión $n-1$» → $d$. En la def. de KDE en variedades, la nota que avisa que se mantiene la notación del original ahora también avisa que el teorema de Pelletier escribe $n$ por $N$.
- Nota al pie de Henry–Rodríguez: «la antípoda de $p, -p$ cae justo fuera de $"iny"_p S^d$» → «$S^2$» (la fórmula es para $S^2$).
- «$S^1 subset RR^2 = {(x, y) : x^2 + y^2 = 1}$» → «$S^1 = {(x, y) in RR^2 : x^2 + y^2 = 1}$».
- Título alemán de von Mises: «Über die 'ganzzahligkeit der' atomgewichte und verwandte fragen» → «Über die 'Ganzzahligkeit' der Atomgewichte und verwandte Fragen» (sustantivos con mayúscula; las comillas del original rodean solo «Ganzzahligkeit»).
- Notas G (18 ecuaciones) y F (16 notas al pie) aplicadas. «Riemanniana/o» → minúscula en todo el archivo salvo el título de la monografía de Muñoz. Patrón «$x-$palabra» (guion dentro del modo matemático, se ve como signo menos) → «$x$-palabra» en todo el archivo (28 casos: $d$-variado, $n$-esfera, $k$-NN, $p$-norma, etc.).
- Erratas: «sí y solo si» → «si y solo si»; «verifican» → «verifica» (sujeto «todo par»); «no es sujeto» → «no está sujeto»; «bilinear» → «bilineal»; «conceptos claves» → «conceptos clave»; «Nótese como» → «cómo»; «aún cuando» → «aun cuando»; «von Mises -- Fisher» → «von Mises--Fisher»; «Rodriguez» → «Rodríguez»; coma sujeto–predicado (×1); raya inconsistente (– y —) → «---»; «$s in RR > 0$» → «$s in RR_+$»; doble punto tras nota al pie en la restricción $h <= h_0$.

**Para decidir**
- Def. «variedad compacta»: «cerrada y acotada se denomina compacta» es Heine–Borel, válido en $RR^d$ pero no en un espacio métrico arbitrario (en variedades riemannianas completas lo da Hopf–Rinow, no citado). No lo toqué: corregirlo exige una referencia que no está en la bibliografía.
- Observación tras esa definición: el «cilindro infinito» ${(x, y, z) in RR^3 : x^2 + y^2 < 1}$ es el cilindro sólido abierto (3-variedad), no la superficie cilíndrica; lo dicho (ni acotado ni cerrado) sigue siendo cierto. Si querías la superficie, cambiar «$<$» por «$=$» y «cerrado» por «cerrada pero no acotada».
- Def. de KDE en variedades: la restricción se enuncia como $h <= h_0 <= "iny" MM$ y el teorema como $h_n < h_0 < "iny" MM$; verificar cuál escribe Pelletier.
- Núcleo isotrópico: en la tabla, «$Y ~ K$» usa `~` dentro del modo matemático; verificar en el PDF que se lea como «distribuido según».
- Encabezados con mayúsculas internas («Variedades Diferenciables», «Probabilidad en Variedades», «Propuesta Original», «Regla de Parsimonia», «Vocabulario y Notación») conviven con encabezados en minúscula («Algoritmos de referencia»). La RAE pide solo la inicial; es una decisión global, no la apliqué.

**Estado**
- $n$ se tolera solo en $S^n$ (esfera) y en enunciados que reproducen a Pelletier/Henry–Rodríguez con aviso.
- «no» + adjetivo sin guion, ya aplicado hasta la línea ~1020.

## Bloque 5 — Preliminares 2.5–2.6 (clasificación en variedades, aprendizaje de distancias, distancia de Fermat)

**Aplicado**
- Def. de consistencia (Devroye): «Sea $L_n = ind(hat(G)_n != G_n)$ la pérdida 0-1 para $hat(G)_n$» hacía de $L_n$ un indicador, con lo que «$lim L_n = L^*$ con probabilidad 1» no tenía sentido. Ahora: «$L_n = Pr(hat(G)_n (X) != G | XX, bu(g))$ la probabilidad de error de $hat(G)_n$ condicional a la muestra», que es la definición de Devroye §6.1 y hace consistentes las dos ecuaciones.
- Loubes et al.: en el denominador de $hat(Pr)(G=k|X)$ la suma reutilizaba el índice $k$ del numerador → índice $j$.
- Isomap, $k$-Isomap: la nota «o viceversa, pues en un grafo no dirigido la relación de vecinos más cercanos es mutua» era falsa (la relación no es simétrica; el grafo se simetriza). Ahora: «O viceversa: el grafo se toma no dirigido, así que basta con que uno de los dos sea vecino más cercano del otro».
- Curva rectificable: en $L(gamma) = sup sum |gamma(t_i) - gamma(t_(i-1))|$ la resta de puntos de #MM no está definida → $dg(gamma(t_i), gamma(t_(i-1)))$.
- Distancia de Chebyshev: «$norm(x)_(p->oo) = max |x_i - y_i|$» mezclaba norma de $x$ con diferencia $x - y$ → «$d_oo (x, y) = norm(x - y)_oo = max abs(x_i - y_i)$».
- Ejemplo edad/cabellos: «$sop(X) = RR^2$» → «$sop(X) subset RR^2$» (edad y cantidad de cabellos no cubren el plano); «$X(Omega) = (X_1, X_2)$» → «$X(omega) = (X_1(omega), X_2(omega))$».
- Nota sobre autocodificadores: el decodificador se llamaba $d(x)$ en la misma frase en que $d$ es la dimensión del código → $delta(x)$.
- Convergencia de $D_(Q, alpha)$ (Groisman et al., Teorema 2.7) estaba en un bloque `#defn` («Definición») → `#thm` («Teorema»). La referencia `@convergencia-sfd` en Resultados ahora dice «Teorema».
- Nota H, colisiones de símbolos:
  - $p$ era a la vez punto de #MM y exponente de la $p$-norma en la definición del costo $J_(g compose f)$ («entre dos puntos cualesquiera $p, q in MM$ … $norm(dot)_p$ es la $p$-norma») → los puntos pasan a $a, b$, coherente con la nota al pie que ya anunciaba ese cambio para Bijral et al.
  - $K$ era la longitud del camino en la def. de distancia muestral de Fermat, y $k$ la del paseo en Bijral et al. y en $d_bu(2)$ de Chu et al. → $m$ en los tres lugares.
  - PCA: «$XX in RR^(N times p)$ … $bu(U)_p$ … primeras $k <= p$ direcciones … $RR^(n times k)$» → dimensión $d$, componentes $m <= d$, $N$ filas. Isomap: «$x_i in RR^p$ … $p$-dimensionales» → $d$; la dimensión de la representación MDS («$d$-dimensional … $RR^d$ … valor óptimo de $d$», «con $d = 2$» en la figura) → $d_MM$.
  - Vincent & Bengio: «si la variedad es $d$-dimensional … $(d+1)$-ésima … las $d$ direcciones principales … $bu(V)_d bu(Lambda)_d bu(V)_d^T$ … dimensión intrínseca $d$ del paso (1)» → $d_MM$ (la subsección siguiente, Brand, ya usaba $d$ ambiente y $d_MM$ intrínseca).
  - Parametrización de Groisman et al.: «$r = beta = (alpha - 1) / d$» → «$d_MM$», como en la observación que sigue al teorema.
  - «$n = 3$ observaciones», «con $n -> oo$ converge», «$k = O(2^(d_MM) ln n)$», «$gamma: [a,b] -> RR^n$, $f: RR^n -> RR$» → $N$ / $RR^d$. «no depende para nada de la dimensión ambiente $D$» → sin el símbolo (Bijral et al. no lo usan así en el texto).
- Notas G (20 ecuaciones) y F (27 notas al pie) aplicadas. «et al» → «et al.» (×7). Erratas: «sí y solo si», «isómetrico», «$epsilon$ ó $k$», «de-por-sí», «cementa» → «cimenta», «aplicándoles» → «aplicándolas», «toma» → «toman» (Vincent et al.), «trabajando» → «trabajó» (oración sin verbo), «monótonicamente» → «monótona», «resulta muy difícil de obtener» → «resulta muy difícil obtener», «aún antes/aún en el habla» → «aun», «en límite» → «en el límite», «Hemos encontrado candidato» → «un candidato», «Data espacial» → «Datos espaciales», «DBDs/PWSPDs» → sin plural, «pseudo-métrica» → «pseudométrica», «$S^2 in RR^3$» → «$subset$», «$Phi$» → «$phi.alt$», «Gaussianas» → minúscula, «--,» → «---,» (×4), un «)» sobrante en «$norm(b - a)_p^q)$», un punto suelto al inicio de línea tras la ecuación de $hat(SS)_cal(K)$, «approx prop» → «prop».

**Para decidir**
- §2.6.3 (Vincent & Bengio): la nota al pie sobre el grupo de Bengio/Rifai es larga para lo que aporta; podría reducirse a la primera oración más las dos citas.
- §2.6.5, definición de «curva rectificable»: la frase «Las curvas rectificables son importantes porque permiten definir conceptos como la longitud de arco y la parametrización por longitud de arco, que son fundamentales en geometría diferencial y análisis» es relleno; se puede quitar sin pérdida.
- §2.6.5, Bijral et al.: la cadena «$approx … prop … = …$» termina en «$=$» con constante implícita; ya señalado en `revision-editorial.md`, sin cambios.
- §2.6.5, «Nótese que #sfd satisface la desigualdad triangular, define una métrica sobre $Q$ y una pseudométrica sobre $RR^d$»: para $x, y in.not Q$ con $alpha > 1$, $sfd(x, y)$ no es en general una pseudométrica en el sentido de la nota (puede fallar la desigualdad triangular fuera de $Q$ según cómo se defina «camino de $x$ a $y$»). Lo dejé; verificar contra la Observación 2.4 de Groisman et al.
- «dataset» aparece 111 veces en redonda y 6 en cursiva; unificar en redonda (lo hago en el bloque 11 salvo indicación contraria).

**Estado**
- $a, b$ para puntos genéricos de #MM desde el costo $J_(g compose f)$ en adelante; $m$ para longitudes de caminos/paseos; $D$ ambiente solo en la nota de autocodificadores, con aviso.
- «et al.» con punto.

## Bloque 6 — Propuesta Original y Metodología (3.1–3.5)

**Aplicado**
- Objetivo 2: «Implementar un estimador de densidad por núcleos basado en la distancia de Fermat, "$f$-KDC"» → «un clasificador de densidad por núcleos» ($f$-KDC es el clasificador, no el estimador).
- Distancia _out-of-sample_: «$Q_i = {x_0} union {x_j : x_j in XX, GG_j = GG_i}$» mezclaba la etiqueta de $x_j$ con el nombre de la clase → «$g_j = i$».
- Regla de parsimonia: «La estrategia de validación cruzada … evaluando su comportamiento en $XX_"test"$, disjunto de $XX_"train"$» confundía los pliegos de validación con el conjunto de evaluación → «evaluando su comportamiento, en cada pliego, sobre observaciones de $XX_"train"$ no usadas para ajustarlos».
- Def. R1SD: paréntesis sobrante en «$hat(s)_(L(mu^star)))$» y punto final movido dentro del `$ $`.
- Def. de verosimilitud: la log-verosimilitud usaba «$op("L")$» donde la verosimilitud se llamó «$op("vero")$» → unificado; «$RR^(n times k)$» → «$RR^(N times K)$»; «$hat(bu(Y))_(i, y_i)$» → «$_(i, g_i)$» (las clases se llaman $g_i$). Def. de exactitud: «$RR^(N times p)$ … $p$ atributos … $n^(-1) sum_(i=1)^n$» → $d$ y $N$ (nota H).
- «@trabajo-futuro["Trabajo Futuro", §]» (suplemento malformado) → «@trabajo-futuro».
- Notas G (5 ecuaciones) y F (11 notas al pie) aplicadas. Erratas: «a fines de» → «a fin de» (×2); «pre-existente», «pre-tratamiento», «pre-procesamiento» → sin guion (RAE); «Github» → «GitHub», con la URL como enlace; «aún así» → «aun así»; «Tanto #kdc, #fkdc y #fkn» → «Tanto #kdc como…»; «Sea además … las predicciones» → «Sean»; punto perdido antes de «Análogamente»; «$k-"NN"$, $epsilon- "NN"$» → «$k$-NN, $epsilon$-NN»; «_tradeoff_» → «_trade-off_»; «Naive Bayes Gaussiano» → «gaussiano»; «KDC» → macro `#kdc`; «$(h_i^*, alpha_i^*)$» → «$^star$»; «no-supervisada», «no-euclídeas», «no-paramétricas» → sin guion.

**Para decidir**
- §3.5.1, def. de verosimilitud: «$product_(i=1)^N Pr(hat(g)_i = g_i)$» es una notación informal (la probabilidad que el clasificador asigna a la clase verdadera); podría escribirse «$hat(Pr)(G = g_i | X = x_i)$». Sin cambios.
- §3.5.1: «Tanto #kdc como #fkdc y #fkn son clasificadores suaves» omite a #kn, que en `scikit-learn` también devuelve probabilidades (frecuencias de vecinos) y se evalúa por $R^2$ en todo el capítulo 4. Sugiero «#kdc, #fkdc, #kn y #fkn».
- §3.5.4, def. R1SD: «minimiza la pérdida de entrenamiento» convive con el $R^2$/log-verosimilitud que se *maximiza* (ya señalado en `revision-editorial.md`). Una línea «(o maximiza el _score_; la pérdida es su opuesto)» bastaría.
- §3.3 (omisión de $theta$): «retomamos como debilidad en @trabajo-futuro»: ahora sin el suplemento; verificar que la referencia se lea bien (debe decir «Sección 6.x»).

**Estado**
- Prefijos soldados: preexistente, pretratamiento, preprocesamiento. El anclaje `<pretratamiento>` no cambió.
