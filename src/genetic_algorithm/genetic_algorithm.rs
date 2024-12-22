use std::marker::PhantomData;

use rand::Rng;

use crate::candidate::Candidate;

use super::{
    genetic_algorithm_candidate::GaCandidate,
    selection::{roulette_wheel_selection, tournament_selection, SelectionMethod},
};


pub struct GA<T, U> {
    chromosome_type: PhantomData<U>,
    population: Vec<T>,
    population_size: usize,
    mutation_rate: f64,
    selection_target: usize,
    elitism_target: usize,
}

impl<T, U> GA<T, U>
where
    T: Candidate<U> + GaCandidate + Clone,
{
    /// Create a new Genetic Algorithm with a population and mutation rate
    /// Offshore the initialization of the population to the user
    /// That depends on the specific problem
    pub fn new(
        population: Vec<T>,
        mutation_rate: f64,
        selection_target: f64,
        elitism: f64,
    ) -> Self {
        let population_size = population.len();
        GA {
            chromosome_type: PhantomData,
            population,
            population_size,
            mutation_rate,
            selection_target: (population_size as f64 * selection_target) as usize,
            elitism_target: (population_size as f64 * elitism) as usize,
        }
    }

    pub fn select(&self, method: SelectionMethod) -> Vec<T> {
        match method {
            SelectionMethod::Tournament(k) => tournament_selection(
                k,
                self.selection_target,
                &self.population,
                0.8,
                self.elitism_target,
            ),
            SelectionMethod::RouletteWheel => roulette_wheel_selection(
                self.selection_target,
                &self.population,
                self.elitism_target,
            ),
        }
    }

    /// Generational Replacement + Elitism
    pub fn repopulate(&mut self, parents: Vec<T>) {
        let mut new_population: Vec<T> = Vec::with_capacity(self.population_size);
        // Prep the probabilities based on fitness
        // May not be good for larger population...
        let total = parents.iter().map(|p| p.get_fitness()).sum::<f64>();
        let mut probabilities: Vec<(&T, f64)> = Vec::with_capacity(self.population_size);
        let mut cumulative = 0.0;
        for parent in parents.iter() {
            let probability = parent.get_fitness() / total + cumulative;
            probabilities.push((parent, probability));
            cumulative += probability;
        }

        let mut rng = rand::thread_rng();
        for _ in 0..(self.population_size - self.elitism_target) {
            let parent_1 = GA::choose(&probabilities, rng.gen());
            let parent_2 = GA::choose(&probabilities, rng.gen());
            let mut child = parent_1.crossover(&parent_2);
            child.mutate(self.mutation_rate);
            child.self_calculate_fitness();
            new_population.push(child);
        }

        // Elitism
        // Get N best candidates and add them to the new population
        let mut sorted_parents = parents.clone();
        sorted_parents.sort_by(|a, b| b.get_fitness().partial_cmp(&a.get_fitness()).unwrap());
        for i in 0..self.elitism_target {
            new_population.push(sorted_parents[i].clone());
        }

        self.population = new_population;
    }

    fn choose(probability_population: &Vec<(&T, f64)>, value: f64) -> T {
        probability_population
            .iter()
            .find(|(_, p)| value < *p)
            .unwrap()
            .0
            .clone()
    }

    /// It's basically the standard deviation of the fitness of the population
    /// The second value is the mean
    /// Returns (standard deviation, mean)
    fn calculate_genetic_diversity(&self) -> (f64, f64) {
        let mean: f64 = self.population.iter().map(|a| a.get_fitness()).sum();
        let mean = mean / self.population_size as f64;
        let std_deviation: f64 = self
            .population
            .iter()
            .map(|a| (a.get_fitness() - mean).powi(2))
            .sum();
        let std_deviation: f64 = std_deviation / self.population_size as f64;
        let std_deviation = std_deviation.sqrt();
        (std_deviation, mean)
    }

    // pub fn step_print(&mut self, selection_method: SelectionMethod) {
    //     let parents = self.select(selection_method);
    //     self.repopulate(parents);
    //     // Calculate best fitness of the population
    //     let fitness = self
    //         .population
    //         .iter()
    //         .map(|a| a.get_fitness())
    //         .max_by(|a, b| a.total_cmp(b))
    //         .unwrap();
    //     let (dev, mean) = self.calculate_genetic_diversity();
    //     println!(
    //         "Best fitness: {}\tDiversity: {}\tMean {}",
    //         fitness, dev, mean
    //     );
    // }

    /// Step through the genetic algorithm
    pub fn step(&mut self, selection_method: SelectionMethod) -> (f64, f64, f64) {
        let parents = self.select(selection_method);
        self.repopulate(parents);
        // Calculate best fitness of the population
        let fitness = self
            .population
            .iter()
            .map(|a| a.get_fitness())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        let (dev, mean) = self.calculate_genetic_diversity();
        (fitness, dev, mean)
    }

    pub fn best(&self) -> T {
        self.population
            .iter()
            .max_by(|a, b| a.get_fitness().total_cmp(&b.get_fitness()))
            .unwrap()
            .clone()
    }
}

impl<T, U> Clone for GA<T, U>
where
    T: Candidate<U> + GaCandidate + Clone,
{
    fn clone(&self) -> Self {
        GA {
            chromosome_type: PhantomData,
            population: self.population.clone(),
            population_size: self.population_size,
            mutation_rate: self.mutation_rate,
            selection_target: self.selection_target,
            elitism_target: self.elitism_target,
        }
    }
}
