use std::{thread::sleep, time::Duration};

use egui::Vec2;
use egui_plot::{Corner, Legend, Line, Plot, PlotItem, PlotPoints};

use crate::{
    candidate::Candidate,
    genetic_algorithm::{genetic_algorithm::GA, selection::SelectionMethod},
    simualted_annealing::simulated_annealing::SimulatedAnnealing,
    tsp::TspCandidate,
};

pub struct RealTimePlotTSP {
    ga: GA<TspCandidate, Vec<(f64, f64)>>,
    iterations: usize,
    last_iteration: usize,
    ga_data: Vec<(f64, f64, f64)>,
    selection_method: SelectionMethod,
}

impl RealTimePlotTSP {
    const SIZE: (f32, f32) = (1900.0, 1000.0);
    const PLOT_SIZE: (f32, f32) = (
        Self::SIZE.0 / 2.0 - Self::SIZE.0 * 0.03,
        Self::SIZE.1 - Self::SIZE.1 * 0.2,
    );

    pub fn new(
        ga: GA<TspCandidate, Vec<(f64, f64)>>,
        iterations: usize,
        selection_method: SelectionMethod,
    ) -> Self {
        RealTimePlotTSP {
            ga,
            iterations,
            last_iteration: 0,
            ga_data: Vec::with_capacity(iterations),
            selection_method,
        }
    }
}

impl eframe::App for RealTimePlotTSP {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let data = self.ga.step(SelectionMethod::RouletteWheel);
        self.last_iteration += 1;
        self.ga_data.push(data);
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.heading(format!("Generation: {}", self.last_iteration));
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Grid::new("grid").show(ui, |ui| {
                ui.label("Fitness & Mean Fitness");
                ui.label("Best Solution");
                ui.end_row();

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        let plot = Plot::new("fitnessplot");
                        plot.legend(Legend::default().position(Corner::RightBottom))
                            .allow_drag(false)
                            .allow_drag(false)
                            .show_axes(true)
                            .min_size(Vec2::new(Self::PLOT_SIZE.0, Self::PLOT_SIZE.1))
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(
                                        self.ga_data
                                            .iter()
                                            .enumerate()
                                            .map(|(generation, data)| [generation as f64, data.0])
                                            .collect(),
                                    ))
                                    .name("Best Fitness"),
                                );
                                plot_ui.line(
                                    Line::new(PlotPoints::new(
                                        self.ga_data
                                            .iter()
                                            .enumerate()
                                            .map(|(generation, data)| [generation as f64, data.2])
                                            .collect(),
                                    ))
                                    .name("Mean Fitness"),
                                );
                            });
                    });
                });

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        let best = self.ga.best();
                        let mut best_chromo = best.get_chromosome().clone();
                        best_chromo.push(best_chromo[0]);
                        let plot = Plot::new("bestplot");
                        plot.allow_drag(false)
                            .allow_drag(false)
                            .show_axes(true)
                            .min_size(Vec2::new(Self::PLOT_SIZE.0, Self::PLOT_SIZE.1))
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(
                                        best_chromo.iter().map(|(x, y)| [*x, *y]).collect(),
                                    ))
                                    .name("Best Solution"),
                                );
                            });
                    });
                });
            });
        });


        if !(self.last_iteration >= self.iterations) {
            ctx.request_repaint();
        }
    }
}

pub struct RealTimePlotSA {
    sa: SimulatedAnnealing<TspCandidate, Vec<(f64, f64)>>,
    iterations: usize,
    last_iteration: usize,
    sa_data: Vec<f64>,
    candidate: TspCandidate,
}

impl RealTimePlotSA {
    const SIZE: (f32, f32) = (800.0, 400.0);
    const PLOT_SIZE: (f32, f32) = (
        Self::SIZE.0 / 2.0 - Self::SIZE.0 * 0.03,
        Self::SIZE.1 - Self::SIZE.1 * 0.2,
    );

    pub fn new(
        sa: SimulatedAnnealing<TspCandidate, Vec<(f64, f64)>>,
        iterations: usize,
        candidate: TspCandidate,
    ) -> Self {
        RealTimePlotSA {
            sa,
            iterations,
            last_iteration: 0,
            sa_data: Vec::with_capacity(iterations),
            candidate,
        }
    }
}

impl eframe::App for RealTimePlotSA {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let data = self.sa.step(self.candidate.clone());
        self.candidate = data.clone();
        self.last_iteration += 1;
        self.sa_data.push(data.get_fitness());
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.heading(format!("Generation: {}", self.last_iteration));
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Grid::new("grid").show(ui, |ui| {
                ui.label("Fitness");
                ui.label("Best Solution");
                ui.end_row();

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        let plot = Plot::new("fitnessplot");
                        plot.legend(Legend::default().position(Corner::RightBottom))
                            .allow_drag(false)
                            .allow_drag(false)
                            .show_axes(true)
                            .min_size(Vec2::new(Self::PLOT_SIZE.0, Self::PLOT_SIZE.1))
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(
                                        self.sa_data
                                            .iter()
                                            .enumerate()
                                            .map(|(generation, data)| [generation as f64, *data])
                                            .collect(),
                                    ))
                                    .name("Best Fitness"),
                                );
                            });
                    });
                });

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        let best = self.candidate.clone();
                        let mut best_chromo = best.get_chromosome().clone();
                        best_chromo.push(best_chromo[0]);
                        let plot = Plot::new("bestplot");
                        plot.allow_drag(false)
                            .allow_drag(false)
                            .show_axes(true)
                            .min_size(Vec2::new(Self::PLOT_SIZE.0, Self::PLOT_SIZE.1))
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(PlotPoints::new(
                                        best_chromo.iter().map(|(x, y)| [*x, *y]).collect(),
                                    ))
                                    .name("Best Solution"),
                                );
                            });
                    });
                });
            });
        });

        ctx.request_repaint();
    }
}
