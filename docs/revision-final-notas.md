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
