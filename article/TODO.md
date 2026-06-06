# TODO

## Abstract

(write last): one sentence stating the headline result and conclusion.

## Things we need to agree on

### The Genome

- _Inputs/Sensors:_ How many rays from car? 5?
  - e.g. at −90°, −45°, 0°, +45°, +90° relative to heading
- _Network:_ a small MLP, e.g. 5 inputs → 6 hidden (tanh) → 2 outputs.
- _Outputs:_ Steering signal [-1, 1]? What about throttle
- _Genome:_ the flattened vector of all weights + biases

Related to lecture 20:

- the genotype is the weight vector;
- the phenotype is the driving behaviour it produces on the track

- Fitness formula? Equation = happy scientist
- "We initialise a population of N" - what is N? 10, 20, 30?
- Crossover variant / mutation rate / noise scale
  - Elitism?
