# Declaración de uso de IA en esta tesis

Este archivo documenta de forma transparente el uso de herramientas de IA
(Claude, de Anthropic) en la elaboración de esta tesis de maestría en
Estadística Matemática.

## Principios éticos de uso

### Lo que la IA **SÍ** hace en este proyecto:
- Corrección de errores gramaticales y ortográficos
- Sugerencias de claridad y fluidez en oraciones
- Detección de inconsistencias de estilo
- Formateo y estructura del documento Typst
- Resolución de TODOs mecánicos (referencias, formato de figuras)

### Lo que la IA **NO** hace en este proyecto:
- Generar contenido matemático original
- Escribir demostraciones o pruebas
- Formular hipótesis o conclusiones
- Crear argumentos o análisis nuevos
- Redactar secciones completas de contenido original

## Flujo de trabajo

1. El autor escribe todo el contenido intelectual
2. La IA revisa y sugiere correcciones de forma
3. El autor acepta, rechaza o modifica cada sugerencia
4. Todos los cambios quedan registrados en el historial de git

## Verificabilidad

El historial de commits de este repositorio permite verificar:
- Qué cambios se hicieron en cada sesión
- La naturaleza de las modificaciones (forma vs. fondo)
- La evolución del documento a lo largo del tiempo

---

# Instrucciones técnicas para Claude

## Idioma
- El texto de la tesis debe estar en español
- Podemos comunicarnos en inglés o español
- Términos técnicos en inglés (como "k-NN", "kernel", "bandwidth")
  son aceptables pero deben aclararse en nota al pie si no son
  ampliamente conocidos

## Formato Typst
- Usar `#figure` para imágenes
- Usar `#defn` para definiciones
- Etiquetar todos los bloques `#theorem`, `#definition`, `#obs`
- El documento principal está en `docs/tesis.typ`
- Todos los gráficos y tablas que se usan en el informe se generaron con `fkdc/viz.py` o desde `docs/figuras-y-tablas.ipynb`.

## Modo de trabajo
- Proponer cambios explicando el motivo
- No reescribir párrafos enteros sin justificación
- Preservar siempre la voz y estilo del autor
- Ante dudas de contenido, preguntar en lugar de asumir
- Editar `docs/tesis.typ` directamente sin pedir confirmación previa para cada edición
- **SIEMPRE** pedir autorización antes de hacer un commit

## Reglas de estilo (según RAE)

**Regla general**: Ante dudas de estilo no cubiertas explícitamente en este
documento, buscar la mejor práctica según la RAE (Diccionario panhispánico
de dudas, Libro de estilo de la lengua española, u otras fuentes oficiales).

### Extranjerismos y términos técnicos
- **Cursiva** para extranjerismos no adaptados: _kernel_, _bandwidth_, _overfitting_
- En Typst usar `#emph[término]` o `_término_`
- Excepción: en textos técnicos, términos de uso muy asentado en la disciplina
  pueden ir en redonda (ej: software, hardware en informática)
- Si no se puede usar cursiva, usar comillas dobles: "kernel"

### Traducciones de términos extranjeros
- **SIEMPRE** ofrecer traducción o equivalente español en nota al pie
  la **primera vez** que aparece un extranjerismo
- En apariciones subsiguientes, usar el término sin nota
- Formato: `#footnote[del inglés _term_, "traducción"]`
- El término original en cursiva (según RAE), la traducción entre comillas
- Ejemplo: _overfitting_ #footnote[del inglés _overfitting_, "sobreajuste"]

### Citas textuales

#### Citas breves (integradas en el texto)
- Entre comillas: «texto citado» o "texto citado"
- Van en redonda (no cursiva), las comillas son suficiente marca
- Ejemplo: Como señala Silverman, «la elección del ancho de banda es crucial»

#### Citas en bloque (más de 40 palabras o varios párrafos)
- Párrafo separado con sangría
- Sin comillas (la separación visual es suficiente)
- Cuerpo de letra menor o cursiva (opcional)
- En Typst usar `#quote[...]` o sangría manual

### Puntuación y sintaxis

#### Coma entre sujeto y predicado
- **NUNCA** se escribe coma entre sujeto y predicado
- Incorrecto: «El estimador de densidad kernel, converge...»
- Correcto: «El estimador de densidad kernel converge...»
- Excepción: cuando hay un inciso entre ambos: «El estimador, como veremos, converge»

#### Voz del autor
- Usar el **"nosotros" de modestia** como norma general: «proponemos», «observamos», «concluimos»
- **Excepción**: usar primera persona singular («propongo», «creo», «considero») cuando se
  afirma una hipótesis no verificada o una opinión personal, para señalar la singular
  responsabilidad del autor de estar equivocado
- Ejemplo nosotros: «Demostramos que el estimador converge en probabilidad»
- Ejemplo singular: «Conjeturo que esta cota puede mejorarse, aunque no tengo una prueba»

#### Sujetos tácitos
- El español permite y favorece el sujeto tácito cuando es recuperable del contexto
- Evitar redundancia: «Calculamos el valor» mejor que «Nosotros calculamos el valor»
- Mantener consistencia dentro de cada párrafo

#### Orden de la oración
- El español es flexible; el orden canónico es Sujeto-Verbo-Objeto pero no obligatorio
- Preferir el orden que favorezca la claridad y el flujo del argumento
- Evitar hipérbatos innecesarios que dificulten la comprensión

## Control de versiones

### Commits
- **SIEMPRE** pedir autorización al usuario antes de hacer cualquier commit
- Hacer commits lo más atómicos posibles (un cambio lógico por commit)
- Todos los commits deben identificar claramente que fueron asistidos por IA

### Formato de mensajes de commit
- Preferir mensajes cortos (≤50 caracteres) cuando sea posible
- Solo agregar cuerpo del mensaje para la marca de IA
- NO agregar descripciones detalladas innecesarias

```
<tipo>: <descripción breve>

🤖 Asistido por IA (Claude)
```

### Tipos de commit permitidos
- `errata:` correcciones gramaticales u ortográficas
- `estilo:` cambios de formato, estructura Typst
- `docs:` actualizaciones a documentación (como este archivo)
- `refactor:` reorganización de texto sin cambiar contenido
- `todo:` reemplaza un comentario TODO por la tarea pendiente correspondiente
- `bib:` completa marcadores `@` o `at` sueltos con la referencia bibliográfica correcta, o agrega citas donde el texto las requiera


### Ejemplos
```
errata: corrige concordancia de género en sección 2.3

🤖 Asistido por IA (Claude)
```

```
estilo: reformatea figuras para usar #figure consistentemente

🤖 Asistido por IA (Claude)
```

## Procesamiento del texto
Los archivos `docs/tesis.typ` y `docs/figuras-y-tablas.ipynb` son extensos. Leerlos por secciones usando offset/limit en lugar de cargar el archivo completo. Buscar patrones específicos con Grep.
