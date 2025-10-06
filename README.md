# Evolve

‼️ WASM Powered Demo at: https://arnavbista.com/projects/constraints

## What is it?

`Evolve` is a personal project of mine inspired by my IB CS final year
case study about Genetic Algorithms, and Constraints Programmings @
The University of St Andrews. It was interesting to see how we can
solve (or approximate the solution) for problems where orthodox approaches fail
due to computational inefficiency.

For example, a brute force attempt at solving the Travelling Salesman Problem
would have a time complexity of O(n!) which is computationally infeasible for
large n

This is the result of multiple trials and errors. The initial program was
written in Julia, rewritten in Java then finaly in Rust as a
complete and generic algorithm. 


## What does it have? 

So far, this project contains two algorithms: 
- Genetic Algorithm
- Simulated Annealing

And it currently solves the [Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

### Genetic Algorithm

To be a bit technical, a genetic algorithm is a method for solving both
constrained and unconstrained optimization problems based on a natural
selection process that mimics biological evolution. It can be used in problems
where the number of possible solutions increases rapidly and where primitive
solutions such as brute force methods might be computationally infeasible.

The selection of the fittest individuals from a population begins the natural
selection process. They generate offspring who inherit the parents' qualities
and are passed down to the next generation. If parents have high fitness, their
children will be fitter than they are and have a better chance of surviving.

This process is known as natural selection and this can be used to find the
solution for a search problem. We consider the best solutions and select the
set of the fittest solutions.

These offsprings have a chance to undergo mutation where some chromosomes in
their DNA are changed. It helps promote biodiversity and helps the algorithm
take a break from any local optimas. 

This process is then repeated for a certain number of generations or until an
end criterion is met. 


### Simulated Annealing

Simulated Annealing is taken from the metallurgical process of annealing where
a metal is heated and then slowly cooled to remove defects and help its molecules
reach into a lower energy state.

The process allows molecues to naturally form a lattice structure as it cools
which is much stronger than a metal that does not have such a structure.

In the context of optimization, the algorithm starts with a random solution
with some temperature `T`. It then gets a neighbour of the solution (some
answer that is similar to the current solution) and calculates the energy of
the neighbour (Think of this as fitness). If the neighbour has a better energy
than the current solution, then the neighbour becomes the current solution.
If not, the neighbour becomes the current solution with a probability of
`exp((current_energy - neighbour_energy) / T)`.

This process is repeated for a certain number of iterations, meets a criteria
or until the temperature reaches a certain threshold.

This method allows the algorithm to escape local optimas and explore the larger
solution space. But due to its sequential nature it may perform slower in some
cases.


## Running it

```bash
cargo run --release
```

That's it!


It stores the city data in a file called `data` where you run the code from.


The interactive CLI will ask you about the parameters of the algorithm, it will
then present a GUI with the Fitness and the Current Best Solution.
