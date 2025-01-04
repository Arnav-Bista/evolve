mod candidate;
mod genetic_algorithm;
mod simualted_annealing;
mod tsp;

use candidate::Candidate;
use genetic_algorithm::{genetic_algorithm::GA, selection::SelectionMethod};
use serde::{Deserialize, Serialize};
use tsp::TspCandidate;
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct City {
    pub x: f64,
    pub y: f64,
}

#[wasm_bindgen]
impl City {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct CandidateWASM {
    #[wasm_bindgen(skip)]
    pub chromosome: Vec<City>,
    pub fitness: f64,
}

#[wasm_bindgen]
impl CandidateWASM {
    #[wasm_bindgen(getter)]
    pub fn chromosome(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.chromosome).unwrap()
    }
}

#[wasm_bindgen]
pub struct GaWasm {
    ga: GA<TspCandidate, Vec<(f64, f64)>>,
}

#[wasm_bindgen]
impl GaWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(population_size: usize, cities: Vec<City>) -> Self {
        let mut initial_population: Vec<TspCandidate> = Vec::with_capacity(population_size);
        let flat: Vec<(f64, f64)> = cities.iter().map(|c| (c.x, c.y)).collect();
        for _ in 0..population_size {
            initial_population.push(TspCandidate::new(flat.clone()));
        }
        let ga = GA::new(initial_population, 0.7, 0.8, 0.01);
        Self { ga }
    }

    #[wasm_bindgen]
    pub fn step(
        &mut self,
        mutation_rate: f64,
        selection_target: f64,
        elitism: f64,
    ) -> CandidateWASM {

        self.ga.set_mutation_rate(mutation_rate);
        self.ga.set_elitism_target(elitism);
        self.ga.set_selection_target(selection_target);

        self.ga.step(SelectionMethod::RouletteWheel);
        let best = self.ga.best();
        CandidateWASM {
            fitness: best.get_fitness(),
            chromosome: best
                .get_chromosome()
                .iter()
                .map(|(x, y)| City { x: *x, y: *y })
                .collect(),
        }
    }
}
