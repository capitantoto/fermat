# DEFENSA — 20/11

## PT 1: Los personajes
- efecto en la locacion/dispersión de la medida (euclidea, coseno) en n->inf AKA "maldicion dimensionalidad"
- pero existe ML en img, video y ahora h/ texto (LLMs!). ¿cómo? la hipotesis de la variedad
- detalle: hip var como claim, del esp. prob. (Ω, A, F)
- y hay teoría! se peude hacer estadística en variedad "riemannianas compactas sin frontera" adaptando métodos euclídeos
- pero hay que conocer, o estimar la variedad. manifold learning ⊆ repr. learning; link to frontier archs as repr. learners
    - PCA como aprendizaje de variedades lineales,
    - Isomap: el grafo [k/eps]NN como aproximación del la variedad,
    - Fermat: distancias euclidas alfa-elevadas como aproximadores de la integral 1/f^beta, (Bijral: se puede encontrar alfa sin saber d por CV!)
- Fermat by visuals:
    - surface plots for 2d toy datasets of single class dentisities 0, 1 and the overlap (vs euclid)
    -  kNN (or qNN) by euclid and fermat dist on espirales, maybe some others, as gifs.
- q: would it help clf then?

## PT 2: Plan
- (fast!) KDC on unknown riemannian manifolds: Pelletier 2005 gives KDE, Pelletier & ? gives binary KDC, henry & rodriguex prove better bounds and try S^2. we oiffer multi-class KDC based on same rule as binary KDC,
- build a sklearn-compatible implementation
- and test sample-based micro-Fermat vs Euclid and a broad panel.
- setting: 25 "randomised" trials on synth + top 7 [?] dsets against SOTA
- 1 slide on code & experimentation framework
  - TODO: Release as library in public pypi! Spruce up before doing so

## PT 3: Results
Make this as visually appleaing as possible, and brief
- w/o scaling, F does not hold (pinguinos debacle)
- it extends the neighborhood quite cleanly (kNN @ espirales), might reuse previous animation
- on sparse sampling and/or large curvature there are marginal gains (does this hold?)

## PT 4: The future
- review extended comparison set from post-handing in with claude, bring up 1-3 really interesting obs or nothing
  - tabfpn merits a mention? is this learned on the weights or we f-ed up the methodology and leaked the eval set?
- and AI? can we use riemannian geometry top better understanmd the representations of domains that are achievable by the multi-layer NN architectures empirically so impressive today?

## TONE & STYLE
Revised: the light version came out too watered down, so the double audience is inverted. Each slide's body is technical without overwhelming (a couple of slides per idea, 30 minutes total); the box at the foot, `#intuicion`, gives the intuition for whoever doesn't read math. What follows is the earlier brief.
Potentially super layman audience as guests mixed in with jury and peers. Keep light, visual heavy, never a dense slide, even if that mean move faster b/w ém. When math is relevant, cover quickly and to the chase, as adog whistle. The jury will understand, the rest don't get mired in a diatribe.
Let's recover the nicest cahrt from the thesis as loss contours, decision frontiers, and even come up with better visuals for givbing a feeling of the distance, even animated.
I don;'t know if slides are really the way to go, but we do have only 3 weeks on maybe 4hs/week, so let's not throw the house off the window. Typst is for sure a stay I think, maybe we can do a fancy html website? Or am i trying to fit a square peg in a round hole?
