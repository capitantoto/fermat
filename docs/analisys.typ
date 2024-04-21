= FKDC - Fermat Kernel Density Classifier

== Análisis

A fines de evaluar las bondades de la técnica propuesta, evaluaremos diferentes _estimadores_ a través de una _métrica de performance_ en distintas _tareas_ de clasificación. La métrica más obvia, por caso, sería la _exactitud_ ("accuracy") de un estimador: el % de observaciones de evaluación clasificadas en la categoría correcta. Otras podrán tomar su lugar.

Para comparar equitativamente diferentes _algoritmos_ de clasificación, elegiremos un _estimador_ mediante la búsqueda de un set de _hiperparámetros óptimos_ por _validaci´øn cruzada_ en un _espacio de búsqueda_ de hiperparámetros acorde a las necesidades del algoritmo.

A priori, quisiéramos comparar FKDC con su primo hermano, el GKDC, clasificador for KD gaussiano, caso particular del FKDC, cuando $alpha == 1$

A fin de obtener resultados de utilidad en el contexto más amplio de las tareas de clasificación, incluiremos entre los algoritmos a evaluar algunos de uso bien extendido en el ámbito industrial y académico, y otros cercanos a FKDC en el conjunto de supuestos y técnicas utilizadas:
- regresión linear
- SVC
- GBTs
- NNs simples
- Naive Bayes

Para tener una idea "sistémica" de la performance de los algoritmos, evaluaremos su performance con diferentes _datasets_. Muchos factores en la definición de un dataset pueden afectar la exactitud de la clasificación; nos interesará explorar en particular 3 que a su vez figuran en el cálculo de la densidad en variedades:
- $n$, el tamaño de la muestra,
- $d$, la dimensión de las observaciones y
- $k$, la cantidad de categorías.

