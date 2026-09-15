# Revisión final de `docs/tesis.typ`

## Contexto

`docs/tesis.typ` es el manuscrito final de mi tesis de maestría en Estadística Matemática. Ya pasó por todas las revisiones que puedo hacer sin perder la cabeza. Esta es la última pasada, granular y secuencial. Pasadas anteriores con LLM fueron ciegas a faltas obvias (por ejemplo, no había introducción); eso ya está corregido, así que no busques problemas estructurales grandes: busca errores concretos.

Ejecutá esta tarea vos mismo, en esta sesión, de forma secuencial, un bloque por turno. No delegues en subagentes: la continuidad de contexto vale más que el paralelismo. Si el contexto se compacta entre bloques, el archivo de notas descrito abajo es el estado.

## Reglas

1. **Prioridad de criterios**, en este orden: (a) corrección del español según la RAE y las reglas de `CLAUDE.md`; (b) corrección matemática al nivel de un curso de posgrado en estadística, precisa pero sin volverse pedante en detrimento del flujo del argumento; (c) consistencia interna del texto tras muchas revisiones.
2. **Tres categorías de hallazgo, tres tratamientos:**
   - *Erratas y giros claramente defectuosos* (ortografía, concordancia, coma entre sujeto y predicado, «sólo» con tilde, extranjerismos sin cursiva, puntuación): corregilos directamente.
   - *Errores de fondo* (un enunciado matemático incorrecto, una afirmación cualitativa que excede lo que los resultados sostienen, una inconsistencia de notación que cambia el significado): aplicá **la edición más corta posible que lo corrija** y anotala en el archivo de notas con la versión anterior citada. Si la corrección requiere conocimiento que no está en el texto ni en la bibliografía citada, **no la apliques**: anotala como pregunta y seguí.
   - *Sugerencias de mejora* que no son errores: solo en el archivo de notas, nunca en el texto. Sé selectivo: únicamente mejoras claras.
3. **No introduzcas conocimiento nuevo.** Ninguna referencia, ecuación, resultado o cita que no esté ya en el texto o en `docs/references.bib`. Ese archivo lo exporta Zotero automáticamente: **no lo edites**.
4. **Preservá mi voz y mi tono.** Primera persona singular en la introducción, en la nota sobre IA y en conjeturas u opiniones; plural de modestia en el resto. No reescribas párrafos; no toques el humor salvo que oscurezca el sentido.
5. **Mecánica de edición.** Hacé cada reemplazo anclado en texto exacto (fallar si no hay coincidencia única), nunca reescrituras de secciones enteras. Compilá con `typst compile --font-path docs/fonts docs/tesis.typ docs/tesis.pdf` al terminar cada bloque y no pases al siguiente si no compila.
6. **Sin commits.** No commitees nada, ni el manuscrito, ni las notas, ni este prompt: todo queda en el árbol de trabajo para que yo lo revise y commitee. No corras `git add`, `git commit` ni `git stash`.

## Procedimiento

Recorré el manuscrito en estos bloques, en orden, uno por turno. Los límites son encabezados, no cantidades de caracteres; apuntan a unas 2500 palabras cada uno.

1. Carátula, Resumen, Abstract e Introducción.
2. «Vocabulario y Notación» (ver nota B más abajo antes de tocarla).
3. Preliminares 2.1–2.3: problema de clasificación, KDE, KDE multivariado y maldición de la dimensionalidad, hipótesis de la variedad.
4. Preliminares 2.4: variedades de Riemann, geodésicas, probabilidad en variedades, KDE en variedades, densidad de volumen.
5. Preliminares 2.5–2.6: clasificación en variedades y aprendizaje de distancias hasta el final de la distancia de Fermat.
6. Propuesta Original completa (3.1 a 3.5, incluida la regla de parsimonia).
7. Resultados 4.1–4.2: In Totis y curvas en el plano (hasta `anteojos` inclusive).
8. Resultados 4.3: datasets 3D y el efecto de la distancia de Fermat en los vecindarios de KN.
9. Resultados 4.4: ruido 12D, datasets orgánicos, alta dimensión.
10. Conclusiones y Trabajo futuro.
11. Anexo A, nota sobre IA, listados; más una pasada de consistencia global (nota H) leyendo solo las notas acumuladas y grepeando el archivo.

Antes del bloque 1, verificá con `git status` que el árbol está limpio; si no lo está, detenete y preguntame. Leé `docs/revision-editorial.md` una sola vez: son observaciones ya conocidas; no las repitas, pero si alguna cae bajo la categoría *error de fondo* y sigue vigente, aplicá la regla 2.

Para cada bloque:

- Leé el bloque completo antes de editar. Editá y compilá.
- Agregá al archivo `docs/revision-final-notas.md` una sección con el nombre del bloque y tres listas: **Aplicado** (solo ediciones de fondo, con el texto anterior citado; las erratas no se listan), **Para decidir** (preguntas y sugerencias, cada una con la ubicación y una frase), **Estado** (qué convención adoptaste que el bloque siguiente debe respetar, p. ej. «puntuación de ecuaciones: dentro del `$ $`»).
- Cerrá el turno con un resumen de tres líneas y esperá mi «seguí» para el bloque siguiente.

## Notas del autor, con la decisión que ya tomé para cada una

**A. «In Totis».** No es latín; la locución es *in toto*. Cambiá el título de 4.1 a «In toto» y nada más.

**B. Vocabulario y Notación.** Reordená las entradas por orden de primera aparición en el cuerpo (grepeá cada símbolo). Eliminá las que no aparecen en el texto o son triviales para un lector de posgrado; candidatas: la medida de Lebesgue, la notación de flecha `a ↦ b`, el símbolo de proporcionalidad, «c.s.», el conjunto vacío. Conservá toda entrada que se use al menos una vez con un significado que un lector podría no conocer. Listá en las notas qué borraste.

**C. Nombres cortos de datasets antes de presentarlos.** Es aceptable en 4.1 para contar «victorias», siempre que se diga que se presentan en detalle más adelante. Si esa frase ya existe, no toques nada; si no, agregá a lo sumo una oración.

**D. Índice del Anexo A.** Hay una solución limpia: los encabezados de las fichas están declarados con `outlined: false`; ponelos en `outlined: true` (el índice general tiene `depth: 2`, así que no los mostrará) y agregá inmediatamente después del párrafo introductorio del anexo `#outline(title: none, target: selector(heading.where(level: 4)).after(<anexo-fichas>))`. Si eso no compila o el resultado no es limpio, dejalo como está y anotalo. No conviertas el anexo en documento aparte.

**E. Bibliografía.** No edites `references.bib`. Verificá y anotá: si hay entradas de tipo `@misc`, con `arxiv` en la URL o DOI, o con `wikipedia`, listalas; y si alguna clave citada en el `.typ` no existe en el `.bib` (Typst lo marca como error de compilación).

**F. Puntuación de notas al pie.** Regla RAE: la nota al pie es un enunciado: empieza con mayúscula y termina con punto, aun cuando sea un fragmento («Del inglés *kernel*, "núcleo".»). Aplicala en todas las notas del bloque; es mecánico y cuenta como errata.

**G. Puntuación de ecuaciones.** Regla: una ecuación en línea aparte forma parte de la oración y lleva el signo de puntuación que le corresponda (coma si la oración sigue, punto si termina), escrito al final de la ecuación, dentro del `$ ... $` (el texto ya usa `thin ,` en varios lugares). Tras una coma, la línea siguiente sigue en minúscula («donde…»); tras un punto, con mayúscula. Aplicala en todas las ecuaciones destacadas del bloque.

**H. Consistencia de notación**, a verificar en cada bloque y globalmente en el último: `d` dimensión ambiente, `d_MM` dimensión intrínseca, `K` número de clases, `k` número de vecinos (y pliegos de CV, avisado), `N` tamaño muestral (cuidado con `n` suelto), `h` ventana, `α`, `μ^⋆` maximizador del *score* de entrenamiento, `μ^{1σ}` elección de la regla de parsimonia, `D_{Q,α}` distancia muestral de Fermat, `𝒟_{f,β}` distancia macroscópica. Se admite desviarse al reproducir la notación de un paper concreto, siempre con aviso en nota o en el texto (ya ocurre para Bijral et al. y para los autocodificadores). Toda desviación sin aviso es un error de fondo: corregila con el cambio mínimo o agregá el aviso.

## Cierre

Al terminar el bloque 11, el manuscrito debe contener ya todas las correcciones preacordadas (reglas 1–2 y notas A–H) y compilar. Devolveme en el chat un mensaje final con: (1) la confirmación de que compila y el conteo de bloques procesados; (2) la lista **Para decidir** consolidada, ordenada por sección, sin repetir lo aplicado, o bien la indicación de leerla en `docs/revision-final-notas.md` si supera las veinte entradas; (3) `git status` resumido, para que yo revise y commitee.
