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

## Bloque 7 — Resultados 4.1–4.2 (In toto, curvas en el plano, anteojos)

**Aplicado**
- Nota A: «In Totis» → «In toto».
- Conteo de tareas: «unas 4900 tareas … más 900 sobre variantes estandarizadas» contaba las 100 tareas de s-LR, excluido del análisis. Con 8 clasificadores × 4 datasets × 25 semillas son 800 → «unas 4800 … más 800».
- `circulos_hi`: «#gnb es superior en $R^2$ y exactitud, aunque … $R^2_#gbt approx 0.09$» → «#gbt» (el JSON de la ficha da GBT 0.089 / 0.655; GNB 0.034).
- `espirales_lo`, fronteras de #svc: la oración «Las fronteras de #svc, que no tienen gradiente de color sino solo una frontera lineal [nota] puesto que…» no tenía verbo principal y la frontera no es lineal → «Las fronteras de #svc no tienen gradiente de color sino solo una línea [nota], puesto que…»; en la nota, «la frontera de estas regiones en es una curva» → «es una curva».
- Nota sobre el tubo $B(MM, tau)$: «el tubo de diámetro $tau approx 6 sigma$ _no_ captura a todas la observaciones con probabilidad menor a uno en un millón» era incorrecto con cualquiera de las dos lecturas: con radio $3 sigma$ (diámetro $6 sigma$), $Pr(norm(epsilon) > 3 sigma) = e^(-4.5) approx 0.011$ por observación y con 800 observaciones casi seguramente alguna queda afuera; con radio $6 sigma$, $800 dot e^(-18) approx 1.2 dot 10^(-5)$. Ahora: «el tubo de radio $tau = 6 sigma$ deja afuera alguna observación con probabilidad del orden de $10^(-5)$».
- Observación «riesgos computacionales»: $n$ por $N$ (nota H) en «$n -> oo$», «$n = 800$», «$n_"train" = n_"eval" = n slash 2$», «$(k-1)/k n/2 = 320$».
- Notas G (3 ecuaciones) y F (10 notas al pie) aplicadas. Erratas: «que de ser óptimos …, pasan» → «óptima … pasa» (sujeto «la familia»); «se puede leer» → «se pueden leer»; «Sirva … , los gráficos» → «Sirvan … los gráficos»; falta el «¿» de apertura en «por qué #kdc no puede elegir»; «de cada acierta» → «de cada una acierta»; «una especie "espiral rectangular"» → «una especie de»; «otorgga»; «casi-cúbica»; «Para hace esto» → «hacer»; «aún» → «aun» (×2); «i.e.» → «es decir»; «scatter plot» → «gráfico de dispersión»; «SVC» → macro `#svc`; «$~50%$» → «$approx 50%$» y «$1/3 50% + 2/3 100%$» con puntos de producto; «$quad.$» y punto fuera del `$ $` en las ecuaciones de $sigma$; punto final en el encabezado del estudio de ablación; «exactitud del $97%$» → «de aproximadamente $97%$» (GBT tiene 96,25 %).

**Para decidir**
- §4.1: la nota al pie que invita al _pull request_ sigue en el cuerpo (ya señalado en `revision-editorial.md`).
- §4.2, `lunas_lo`: el cálculo «$1/3 dot 50% + 2/3 dot 100% approx 86.7%$» sigue presentado como derivación; la sugerencia de «a ojo» de la revisión anterior no se aplicó.
- §4.2.4, observación «unidades de la pérdida» y nota de las grillas: las dos «N. del E.» son notas del autor, no de un editor; si el tribunal no comparte el chiste, «Nota:» basta.
- §4.2.5 («Efectos de aumentar el ruido»): «En `espirales_hi` … #svc obtiene la mejor exactitud apenas por encima de #fkdc» son 0,8575 contra 0,8325; «apenas» es generoso.
- «Hete aquí» aparece dos veces en tres páginas (ya señalado).

**Estado**
- Conteo oficial: 4000 tareas crudas + 800 estandarizadas = 4800, 8 clasificadores.

## Bloque 8 — Resultados 4.3 (datasets 3D y vecindarios de KN)

**Aplicado**
- `helices_0`: «en unos cuantos casos --- $s in {1188, 1182, 2411}$ --- en que $alpha_#fkdc = alpha_#kdc = 1$»: según `data/helices_0-parametros_comparados-kdc.csv`, la semilla 1188 eligió $alpha = 2.5$ (como el propio texto dice dos oraciones antes). Reemplazada por 6610 ($Delta_(R^2) = 0.098$, $alpha = 1$): ahora «$s in {1182, 6610, 2411}$».
- `helices_0`: «$h_#fkdc / h_#kdc approx 14.3$» estaba invertido ($h_#fkdc = 0.01$, $h_#kdc = 0.143$) → «$h_#kdc / h_#fkdc approx 14.3$»; «$Delta_R^2$» → «$Delta_(R^2)$»; «$h = 0,000562$» → «$0.000562$» (coma decimal aislada).
- «==== Hélices» estaba un nivel por debajo de «=== Eslabones», «=== Pionono» y «=== Hueveras» → «=== Hélices».
- Notas G (2 ecuaciones) y F (5 notas al pie). Erratas: «consiste de» → «consiste en»; «en la que … tiene» → «en las que … tienen»; «iguala o supera» → «igualan o superan»; «`helice`» → «`helices`»; «orácticamente», «alvanzado»; «todas independiente» → «independientes»; «aún con» → «aun con»; «$0.25-0.30$» → «$0.25$--$0.30$» (el guion en modo matemático es un menos); «no-reducible», «no-lineales», «no-nulos» → sin guion; «$||dot||$» → «$norm(dot)$»; espacio y punto en el pie de la figura de superficies (`$alpha = 3$.(der., …).`).

**Para decidir**
- `helices_0`, párrafo «Nuestra hipótesis…»: «el entrenamiento por CV maximiza el _score_ en $(alpha=3, h = 0.000562)$» para $s = 1182$; la columna `max_score_alpha_test` del CSV da 3.5 para esa semilla. Si esa columna es el maximizador que describe el texto, corresponde $alpha = 3.5$; si es otra cosa, nada que cambiar.
- `eslabones_0`: «La semilla resultó adversa para ambos» — la tabla previa solo muestra a #fkdc; no queda claro quién es el otro (¿#fkn?).
- «Todo algoritmo funciona OK» (observación sobre #logr en hélices): coloquial; «funciona bien» si se quiere formalizar.
- Pie de la tabla de #fkn vs. #kn en `hueveras_0`: «$Delta_(R^2) > 0$ en casi todos los casos» (hay un caso con $Delta = -0.092$; ya señalado).

**Estado**
- Sin novedades de notación.

## Bloque 9 — Resultados 4.4 (ruido 12D, datasets orgánicos, alta dimensión)

**Aplicado**
- `mnist`: «$d = 768$» (×2: en la introducción de «Alta dimensión» y en «A `mnist` ($N = 60000$, $d = 768$) se lo redujo de $d = 784$…») → «$d = 784$» ($28 times 28 = 784$, como dice el propio texto a continuación).
- Nota al pie sobre el estandarizador: el nombre de la clase enlazada estaba mal escrito, «`sklearn.prepocessing.StandarScaler`» → «`sklearn.preprocessing.StandardScaler`».
- Verificados contra los JSON/CSV de `docs/data`: podios de `pinguinos`, `iris`, `vino`, `digitos`, `mnist` y sus variantes `_std`; «a cinco milésimas de #logr» (0,9639 − 0,9586); «13 de las 25» y «18 de las 25» ($alpha = 1$ bajo parsimonia para #fkn); anchos de banda 316–562; tabla de escala de distancias (1300 / 2500 / 169; «entre diez y treinta veces», «menos de tres», «menos de dos»); caídas de $R^2$ en `pionono_12` ($approx 0.1$) y `eslabones_12` ($approx 0.25$). Todo coincide.
- Nota F (6 notas al pie). Erratas: «clses»; «exactitud y el $R^2$ … mejora» → «la exactitud y el $R^2$ … mejoran»; «se ajustan … bien la hipótesis» → «bien a la hipótesis»; «capturados» → «capturado» (el dígito); «desvió» → «desvío»; «(col .2)»; «pre-procesamiento» → «preprocesamiento»; «pero sin embargo … no sólo» → «y sin embargo … no solo»; «90%» → «90 %».

**Para decidir**
- §4.4.1: «sus 34 observaciones del conjunto de evaluación se clasifican como Adelie» y los rangos «entre 13 y 230» / «2700 a 6300» no se verificaron contra el dataset (ya señalado en `revision-editorial.md`); los rangos son consistentes con Palmer penguins de memoria.
- §4.4.3, `mnist`: «El clasificador de densidad degenera así en una especie de $1$-NN blando, y sin embargo #kdc supera por un buen margen a #kn»: la explicación de por qué un «1-NN blando» le gana a #kn no se da; una frase («porque el peso decae con la distancia en lugar de ser uniforme en los $k$ vecinos») cerraría el argumento, pero es contenido nuevo; no lo agregué.
- «_pooleadas_» (§4.3.5): anglicismo adaptado a la criolla; «agrupadas» o «concatenadas» si se quiere neutralizar.

**Estado**
- $d = 784$ para `mnist` original, $96$ tras PCA.

## Bloque 10 — Conclusiones y Trabajo futuro

**Aplicado**
- Ninguna edición de fondo. Conteos de las Conclusiones (7 / 5 / 2 por $R^2$; 20 datasets; cuatro referencias; 4000 tareas) coinciden con 4.1.
- Nota H: «$n$ decreciente», «$ln n$» → $N$. Nota F (2 notas al pie). Erratas: «empatado» → «empatados»; «logística. en `digitos`» → «logística; en `digitos`»; «puediera» → «pudiera»; «pre-procesar», «pre-tratamiento» → soldados; «Macbook … 8GB RAM» → «MacBook … 8 GB de RAM»; punto antes de la nota al pie del último ítem, que ya cerraba con «;».

**Para decidir**
- Conclusiones: «#svc resulta casi imbatible» y «Ningún algoritmo evaluado fue universalmente óptimo» siguen como estaban (ya señalados en `revision-editorial.md`).
- Trabajo futuro, segunda línea: «cuando es pequeño, hay que tomar $h > "iny" MM$» va dentro de la oración que empieza «Conjeturo que…», así que ya está marcada como conjetura; si se quiere reforzar, «habría que tomar».
- Trabajo futuro, tercera línea: «estimar $theta$ en lugar de ignorarla»: $theta$ es «la función de densidad de volumen» (femenino) pero en 3.3 se dice «el factor omitido»; el género flota entre «la» y «el» según se piense en la función o en el factor. Sin cambios.

**Estado**
- Sin novedades.

## Bloque 11 — Anexo A, nota sobre IA, listados y consistencia global

**Aplicado**
- Nota D: los encabezados de las fichas pasan a `outlined: true` y el anexo abre con `#outline(title: none, target: selector(heading.where(level: 4)).after(<anexo-fichas>))`. Compila y lista las 24 fichas en una columna limpia; el índice general (`depth: 2`) no cambia.
- «= Listados» era el único encabezado numerado después del cuerpo (el anexo y la nota sobre IA no lo están) → `#heading(numbering: none)[Listados]`.
- Nota sobre IA: «los resultados y su interpretación es mía» → «son míos»; «miusmo» → «mismo»; «LLMs» → «LLM»; espacios dobles.
- Nota E (bibliografía, sin editar): 54 entradas en `references.bib`; tres de tipo `@misc`, las tres preprints de arXiv: `bengioConsciousnessPrior2019`, `buitinckAPIDesignMachine2013`, `mckenziePowerWeightedShortest2019`. Ninguna entrada de Wikipedia (los enlaces a Wikipedia del texto son `#link`, no citas). Todas las claves citadas existen (el documento compila sin avisos); 24 entradas del `.bib` no se citan, lo que es normal en una exportación de biblioteca completa de Zotero: Typst solo imprime las citadas.
- Nota H, pasada global por `grep`: no quedan «Riemanniana/o» (salvo el título de la monografía de Muñoz), «sólo» (salvo la cita textual del sitio de la materia), «i.e.», «et al» sin punto, «no-» con guion, «$x-$palabra», ni «dataset» en cursiva (unificado en redonda: 6 casos). $p$ como dimensión ya no aparece; $D$ ambiente solo en la nota de autocodificadores con aviso; $n$ solo en los enunciados que reproducen a Pelletier, Devroye y Groisman et al., con aviso, y como índice de partición en la curva rectificable.
- Notas F y G: el chequeo automático sobre todo el archivo no encuentra notas al pie que empiecen en minúscula o terminen sin punto, ni ecuaciones destacadas sin puntuación (salvo las dos que terminan en «$square$» y la lista de valores de grilla, que llevan coma).

**Para decidir**
- Nota sobre IA: dice «_Claude Opus_ versiones 4.6 a 5.1»; el asistente de la revisión de septiembre de 2026 fue Claude Fable 5.1 (y en esta pasada final, también). Corregir el nombre o dejar «Claude (Anthropic), versiones de 2026».
- `CLAUDE.md` del repositorio sigue diciendo que la IA no formula hipótesis ni redacta pasajes; la nota sobre IA dice, a propósito, lo contrario. Conviene alinear el archivo con la nota antes de la entrega, ya que la nota remite al historial del repositorio.
- Encabezados: «Vocabulario y Notación», «Variedades Diferenciables», «Probabilidad en Variedades», «Propuesta Original», «Regla de Parsimonia», «Pionono, Eslabones, Hélices y Hueveras» llevan mayúsculas internas; el resto («Algoritmos de referencia», «Trabajo futuro») no. La RAE pide solo la inicial. Es un cambio global de una línea por encabezado; no lo apliqué por ser una decisión de estilo.

---

# Para decidir — lista consolidada, por sección

**Carátula e Introducción**
1. Fecha «19 de mayo de 2026» en la carátula: verificar si esta versión la reemplaza.
2. «Lugar de Trabajo», «Fecha de Defensa»: mayúsculas internas (RAE: solo la inicial), salvo que el formato lo fije la Facultad.
3. La cita del sitio de la materia conserva «sólo» y «bienvenides!» sin «¡»: es textual, se dejó.

**2. Preliminares**
4. §2.1: la primera línea de «$hat(G)(x) = arg min_f EE(L(G, f(X)))$» podría ser la versión condicional; el paso siguiente ya lo hace explícito.
5. §2.3.5: nota al pie sobre Gallese y Bengio, larga (ya en `revision-editorial.md`).
6. §2.4, def. «variedad compacta»: «cerrada y acotada = compacta» es Heine–Borel; en variedades vale por Hopf–Rinow (no citado). Se dejó.
7. §2.4, mismo párrafo: el «cilindro infinito» con «$<1$» es el sólido, no la superficie; lo afirmado sigue siendo cierto.
8. §2.4, KDE en variedades: restricción «$h <= h_0 <= "iny" MM$» vs. teorema «$h_n < h_0 < "iny" MM$»; verificar cuál escribe Pelletier.
9. §2.4, núcleo isotrópico: «$Y ~ K$» usa `~` en modo matemático; verificar en el PDF.
10. §2.6.3: nota al pie sobre el grupo de Bengio/Rifai, larga.
11. §2.6.5: la oración «Las curvas rectificables son importantes porque permiten definir…» es relleno.
12. §2.6.5, Bijral et al.: cadena «$approx … prop … = …$» con constante implícita (ya señalado).
13. §2.6.5, def. de $D_(Q, alpha)$: «define … una pseudométrica sobre $RR^d$»; verificar contra la Observación 2.4 de Groisman et al.

**3. Propuesta y metodología**
14. §3.5.1, verosimilitud: «$product Pr(hat(g)_i = g_i)$» es informal; alternativa «$hat(Pr)(G = g_i | X = x_i)$».
15. §3.5.1: «Tanto #kdc como #fkdc y #fkn son clasificadores suaves» omite a #kn, que se evalúa por $R^2$ en todo el capítulo 4.
16. §3.5.4, R1SD: «minimiza la pérdida» vs. el _score_ que se maximiza (ya señalado).

**4. Resultados**
17. §4.1: nota al pie que invita al _pull request_ (ya señalado).
18. §4.2, `lunas_lo`: «$1/3 dot 50% + 2/3 dot 100% approx 86.7%$» presentado como derivación; «a ojo» (ya señalado).
19. §4.2.4: las dos «N. del E.» son notas del autor; «Nota:» si el chiste no gusta.
20. §4.2.5, `espirales_hi`: «#svc … apenas por encima de #fkdc» son 0,8575 vs. 0,8325.
21. «Hete aquí» ×2 en tres páginas.
22. §4.3, `helices_0`: «maximiza el _score_ en $(alpha=3, h = 0.000562)$» para $s = 1182$; el CSV da `max_score_alpha_test = 3.5` para esa semilla. Verificar qué columna describe el texto.
23. §4.3, `eslabones_0`: «La semilla resultó adversa para ambos»: ¿quién es el otro?
24. §4.3, hélices: «Todo algoritmo funciona OK» (coloquial).
25. §4.3, `hueveras_0`, pie de tabla: «$Delta_(R^2) > 0$ en casi todos los casos» (hay uno con $-0.092$).
26. §4.4.1: «34 observaciones» y rangos 13–230 / 2700–6300 no verificados contra el dataset.
27. §4.4.3, `mnist`: no se explica por qué el «$1$-NN blando» le gana a #kn; una frase sobre el peso decreciente con la distancia cerraría el argumento (contenido nuevo, no agregado).
28. §4.3.5: «_pooleadas_».

**5. Conclusiones y Trabajo futuro**
29. «#svc resulta casi imbatible»; «Ningún algoritmo evaluado fue universalmente óptimo» (ya señalados).
30. «hay que tomar $h > "iny" MM$» está dentro de «Conjeturo que…»; «habría que» si se quiere reforzar.
31. Género de $theta$: «estimar $theta$ en lugar de ignorarla» (la función) vs. «el factor omitido».

**Anexo, nota sobre IA, repositorio**
32. Nota sobre IA: «_Claude Opus_ versiones 4.6 a 5.1» → el modelo de septiembre de 2026 fue Claude Fable 5.1.
33. `CLAUDE.md` contradice la nota sobre IA; alinear antes de la entrega.
34. Mayúsculas internas en encabezados (decisión global de estilo).
