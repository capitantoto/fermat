# Revisión editorial de `tesis.typ` (rama `recorte-resultados`)

Lectura completa del cuerpo (Vocabulario → Trabajo futuro), hecha con dos sombreros: corrector de estilo (norma RAE) y director/revisor de contenido. Lo mecánico e inequívoco ya está aplicado en el mismo commit (45 erratas: tildes, «aun/aún», «solo», concordancias, palabras pegadas, un ítem perdido en la definición de espacio topológico, tres anglicismos verbales). Lo que sigue son decisiones del autor.

## A. Urgente antes de cualquier entrega

1. **Resumen y Abstract están vacíos.** Solo quedaron las palabras clave. Un párrafo de ~200 palabras cada uno: problema (clasificación en variedades desconocidas), propuesta (distancia de Fermat muestral en KDC y k-NN, implementación `scikit-learn`), diseño (20 datasets × 25 semillas × 9 clasificadores), hallazgos (competitivo por R²; ventaja real en hélices/hueveras; sensible a ruido y a escala), y una frase de alcance.
2. **Tabla «In Totis» y datos crudos.** La tabla de podios sigue contando `pinguinos` (s-LR) y `vino` (GBT) sobre atributos crudos. El pie de tabla ya lo aclara; falta decidir si se recuenta con las variantes `_std` (recomendado) o se deja con la salvedad. Si se recuenta, actualizar el «7 / 5 / 3 / 2» del texto y del pie.
3. **Conteos.** Verificados contra los CSV regenerados: por R², f-KDC 7, GBT 5, f-KN 3, KDC 2, KN 2, s-LR 1 (familia K: 14 de 20); por exactitud, SVC 6, GBT 4, KDC 3, f-KN 2, KN 2, s-LR 2, f-KDC 1 (familia K: 8 de 20). El texto de 4.1 coincide. Ojo: `helices_12` es un empate a R² ≈ 0.002 entre f-KN y KN que el desempate del código adjudica a KN; no apoyar ninguna afirmación en ese dataset.
4. **Variantes `_std`.** Solo para iris, vino, pinguinos y digitos (las de mnist y los `_12` se descartaron: estandarizar componentes principales o ruido de varianza controlada no tiene sentido). Resumen: `pinguinos` f-KDC 0.42 → 0.96 (a 0.005 de s-LR); `vino` familia K 0.43 → 0.84–0.88 (GBT y s-LR 0.90 no se mueven); `iris` K baja levemente; `digitos` K baja 0.98 → 0.87. Texto de 4.4 y Conclusiones alineados con estos números.

5. **s-LR eliminado del análisis.** La regresión logística con escalador previo era el único algoritmo con pre-tratamiento; ahora el efecto de la escala se lee en las variantes `_std`. Quedan 8 clasificadores y 4000 tareas; podios, fichas y tablas regenerados sin s-LR (en `pinguinos` el podio pasa a LR, 0.963).

## B. Estilo y norma (RAE)

- **Anglicismos evitables** (los verbales ya se corrigieron): _scatter plot_ → «gráfico de dispersión» (se usan ambos; unificar); _boxplot_ está aceptado como extranjerismo en cursiva, pero aparece a veces sin cursiva; _splits_ → «particiones»; _out-of-sample_ ya tiene nota, bien; _dataset_ en cursiva es consistente, mantener.
- **«In Totis»** no es latín: es *in toto* («en conjunto»). Sugiero «En conjunto» o «Panorama general» como título de 4.1.
- **Muletillas.** «Hete aquí» aparece dos veces en tres páginas; «Pues bien» y «Ahora bien» abren cinco párrafos del capítulo 2. Una vez cada una alcanza.
- **Comillas.** Las comillas dobles se convierten en «» por Typst, bien; pero hay términos entrecomillados que ya no lo necesitan a la segunda aparición («suficientemente grande», «bien muestreado», «ruido blanco»). Regla: comillas la primera vez o cuando se usa en sentido no literal; después, sin ellas.
- **Rayas de inciso.** El texto abusa de «--- inciso ---» dentro de oraciones ya largas (hay párrafos con tres). Convertir uno de cada tres en oración aparte mejora la cadencia sin perder nada.
- **Mayúsculas.** «Distancia de Fermat» y «Regla de Bayes» van en minúscula (distancia de Fermat, regla de Bayes) salvo en títulos. «Naive Bayes» puede quedar como nombre propio.
- **Notas al pie.** Hay más de 130. Muchas son incisos de una línea que caben en el texto entre comas, y varias son chistes («distancia de San Telmo», «en criollo, un montón»). Sugiero conservar el humor en dos o tres lugares elegidos y absorber o eliminar el resto; el tribunal las lee todas.
- **Notación.** (i) `vero` se define como `op("vero")` pero la log-verosimilitud usa `op("L")`: unificar. (ii) En la definición de verosimilitud la matriz de probabilidades es `RR^(n times k)` con `k` clases; en todo el texto las clases son `K`. (iii) Ídem `hat(bu(Y))_(i, y_i)` con `y_i` por `g_i`. (iv) `#clf` como macro para «clasificador» aparece solo en esas definiciones; considerar `hat(G)` como en el resto.

## C. Contenido y argumentación (director / revisor)

### Capítulo 1–2 (Vocabulario, Preliminares)
- La sección 2.3.4 («maldición de la dimensionalidad») está muy bien lograda; el ejemplo del audio a 8 kHz es memorable. Sugiero cerrar el ejercicio con la frase de la figura («un d-cubo de lado h») para que el lector no tenga que reconstruir la conexión.
- La nota sobre empatía y conciencia como «variedades» (Gallese; Bengio 2019) es simpática pero larga y desvía. Reducir a una oración con las dos citas.
- Definición de espacio topológico: el tercer axioma había perdido su viñeta (corregido). Revisar además que «base numerable» esté definida o referenciada en la definición de variedad topológica: se usa sin introducirla.
- «Radio de inyectividad»: la nota al pie de 8 líneas contiene la definición más clara (cut locus). Invertir: cut locus en el cuerpo, la definición por bolas normales en nota.

### Capítulo 2.4–2.6 (variedades, KDE, aprendizaje de distancias)
- Teorema de Pelletier: se enuncia con `h_n < h_0 < iny M` y la nota dice «esta restricción no es catastrófica». Bien. Pero más adelante (Trabajo futuro) se afirma que en espacios ralos «no queda otra que tomar h > iny M, violando el supuesto»: eso es una hipótesis del autor, no un hecho medido. Marcar como conjetura (primera persona singular, como manda el CLAUDE.md).
- Densidad de volumen: la Definición 2.4.18 y su observación están ahora bien acotadas. La frase «En general, su cómputo resulta sumamente complejo» necesita una cita (Besse §6 o Berenfeld et al., ya citados más abajo).
- Brand (2002): la sección está bien tras la revisión; la conclusión «complejiza en lugar de simplificar» es opinión razonable, pero conviene decir en una línea *por qué* nos importa: no hay estimador de θ_p disponible sin atlas.
- Bijral et al.: la derivación `J_r ≈ ||b−a||^q` mezcla `≈` y `∝` en la misma cadena y termina en «=» con proporcionalidad implícita. Escribir `∝` en toda la cadena o introducir la constante.
- Chu et al.: el ejemplo trivial está correcto. Falta una frase que diga qué se gana con la equivalencia *para esta tesis*: justifica usar el grafo completo con `k = O(2^d log n)` vecinos sin perder exactitud, que es lo que hace `fkdc/fermat.py`.
- Definición de distancia muestral de Fermat: `K` se usa como longitud del camino y como número de clases en la misma página. Cambiar a `m` o `n_γ`.

### Capítulo 3 (Propuesta y metodología)
- La lista de cinco objetivos es clara. Sugiero cerrarla con una frase que anticipe la respuesta («encontramos que…»), porque el lector llega al capítulo 4 sin saber qué esperar.
- Regla de parsimonia (R1SD): la definición usa «pérdida» y `arg min`, pero el entrenamiento maximiza un `score`; la observación «unidades de la pérdida» lo aclara recién en 4.2. Adelantar una línea a la definición.
- «Grilla cuadrada» es un neologismo; en español se dice «grilla completa» o «producto cartesiano de grillas».
- Métricas: la exactitud se define con `n` y la muestra con `N`; unificar.
- Pre-tratamiento: la sección es ahora el ancla de §4.4.1. Vale agregar una oración: «la familia K, como veremos, es igual de sensible a la escala que la regresión logística».

### Capítulo 4 (Resultados)
- 4.1: «casi imbatible» (SVC) y «rindió frutos» son lenguaje de prensa; «obtuvo la mayor exactitud en 6 de 20» dice lo mismo sin adjetivos.
- La nota al pie que invita al lector a hacer un *pull request* es encantadora pero no es tono de tesis; moverla a un README o al anexo.
- 4.2 «lunas_lo»: el cálculo «1/3 · 50 % + 2/3 · 100 % ≈ 86.7 %» es una heurística visual presentada como derivación. Decir «a ojo».
- 4.2 «Estudio de ablación»: el título define ablación por la RAE, gracioso pero largo; basta «(quitar la distancia de Fermat)».
- 4.2 «Hiperparámetros óptimos en lunas_lo»: es la sección más honesta del capítulo (la ventaja viene de la grilla y de la validación cruzada). Sugiero que esa conclusión aparezca también en el resumen de 4.1, para que no parezca escondida.
- 4.3 Hélices: el argumento de k = 1 y la robustez a k es el mejor resultado. Merece una oración en el Resumen.
- 4.3 Hueveras: la tabla 7 muestra un caso con Δ = −0.092 y α = 1.25 que contradice la frase «Δ > 0 en casi todos los casos». Decir «en 7 de 8 casos con α > 1».
- 4.4 12D: el párrafo explicativo (ruido alarga aristas; GBT interroga una variable a la vez) es bueno. Cuando lleguen `pionono_12_std` y `hueveras_12_std`, actualizar la frase «no cambia el cuadro» con los cuatro datasets.
- 4.4.1 Pingüinos: verificar los rangos citados (13–230 mm, 2700–6300 g) contra el dataset; son de memoria.
- 4.4.2 Alta dimensión: «Esperábamos alguna ventaja más notable… que no se comprobó» es la actitud correcta; dejarla.

### Conclusiones y trabajo futuro
- Reescritas en esta rama; releerlas con el Resumen a la vista para que digan lo mismo con las mismas palabras clave.
- «Ningún algoritmo evaluado fue universalmente óptimo» es un lugar común (*no free lunch*); si se mantiene, citar Wolpert (1996) o quitarlo.
- Nueva sección 3.3 «La densidad de volumen, omitida»: verificar contra Besse §6.3 el desarrollo θ_p(exp_p v) = 1 − Ric_p(v,v)/6 + O(‖v‖³) antes de defenderlo; la afirmación «del mismo orden que el sesgo» es correcta si se acepta ese desarrollo. Demšar (2006) se agregó a mano a `references.bib`: falta cargarlo en Zotero.

## D. Estructura y extensión

- Con el Anexo A, el cuerpo termina en la página ~94 y el documento en ~128. Si hace falta recortar más, los candidatos por orden: (1) notas al pie (ver B), (2) la sección de Brand (2002) a la mitad, (3) la historia de von Mises/Fisher a un párrafo, (4) la observación «riesgos computacionales» a un párrafo en Metodología.
- Falta un puente de dos líneas al inicio del capítulo 4 que enumere las cuatro familias de datasets y diga qué pregunta responde cada una.
