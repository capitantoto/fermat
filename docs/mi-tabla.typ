#show table.cell.where(y: 0): set text(weight: "bold")
#table(
  columns: 3,
  stroke: none,
  align: (x, y) => if y == 0 { center } else { if x == 0 { bottom } else { top } },
  table.header[clf][$R^2$][exac],
  table.hline(stroke: 1pt),
  table.vline(x: 1, start: 1, stroke: .5pt),
  [FKDC], [1.0], [1.0],
  [KDC], [1.0], [1.0],
  [GNB], [1.0], [1.0],
  [KN], [1.0], [1.0],
  [FKN], [1.0], [1.0],
  [LR], [0.99994], [1.0],
  [SLR], [0.99952], [1.0],
  [GBT], [0.9995], [1.0],
  [SVC], [--], [1.0],
)
