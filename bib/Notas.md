# Notas
## on Weighted Geodesic ...
### Exponente primer-último término para $D_{X_n}$ en $p, q$ arbitrarios
> (p. 2 parr. 2) The minimization is done over all K ≥ 2 and all finite sequences of data points with $x_1 = \text{argmin}_{i \in N} l(x_i, p)$ ...

Debería decir $l(x_i, p)^d$?

### $D_{X_n}$ es una distancia en $\mathcal{M}$

> (p.2 parr. 2) Also observe that $D_{X_n}$ verifies triangular inequality and so, it is indeed a distance.

Dem:
1. Id. Indisc. : requiere que $l(p, p) = 0$, pero no puedo definir l en $R^p$ sin especificar de qué variedad $M \sim R^p \subseteq R^d$ estoy hablando. Luego, $l$ tiene que ser distancia euclídea en $R^d$ en lugar de $R^P$, puede ser?
2. Simetría: asumo trivial cuando $l$ es simétrica y el grafo es no-dirigido.
3. Desig. triangular:

 Sea $geod_{D}(p, q)$ la geodésica  inducida por la distancia $D$, y $a + b$ el camino que resulta de concatenar los caminos $a = (a_1, \dots, a_n, x)$  y $b = (x, b_1, \dots, b_m)$. Cuando

 - $geod_D(p, r) = geod_D(p, q) + geod_D(q, r)$, resulta que $D(p, r) =  D(p, q) + D(q, r)$ y la desigualdad se cumple con igualdad.
 - $geod_D(p, r) \ne geod_D(p, q) + geod_D(q, r)$, el RHS es _un_ camino entre $p$ y $r$, por lo cual está incluido en el conjunto sobre el que se minimiza la expresión que define $D_{X_n}(p, r)$, y la desigualdad se cumple con desigualdad (?).

 QED. Es así, casi tautológico? O me falta algo?

 ### Elección de $d$
 No es razonable suponerlo atado a la dimensión $d_{\mathcal{M}}$ de la variedad?

 ## Weighted Geodesic Distance Following Fermat's Principle

 ### 3 Clustering Properties
 En la descripción del experimento, no se hace mención directa al método de validación: ¿las métricas reportadas se midieron sobre el conjunto de entrenamiento o de validación?

 De hecho, asumo que sobre el de entrenamiento, porque no veo cómo predecir fuera-de-muestra con la implementación actual. `FKMeans` no tiene `predict`! Vaya estimador...

 TODO: Escribir un predictor basado en distancia de Fermat, pero que se pueda computar en nuevas partículas $p \notin Q_n$.

 ```
landmarks = np.random.RandomState(seed=self.seed).choice(
    range(distances.shape[0]), self.landmarks
)
```

Es una elección muy poco informativa de _hitos_ para elegir. Probemos con los KMedoides que devuelve FermatKMeans?
