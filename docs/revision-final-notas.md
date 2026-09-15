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
