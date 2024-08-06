// Released under MIT License.
// Copyright (c) 2023 Ladislav Bartos

use rand::rngs::ThreadRng;
use rand::Rng;

use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

use crate::bond::Bond;
use crate::diffusion::Diffusion;
use crate::particle::Particle;
use crate::statistics::MoveStatistics;
use crate::DIFFUSION_MULTIPLIER;

#[derive(Clone, Copy, PartialEq)]
pub enum Dimensionality {
    ONE,
    TWO,
}

/// Allows the usage of print* macros for Dimensionality.
impl fmt::Display for Dimensionality {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Dimensionality::ONE => write!(f, "1D"),
            Dimensionality::TWO => write!(f, "2D"),
        }
    }
}

pub struct System {
    /// particles in the system
    pub particles: Vec<Particle>,
    /// bonds in the system
    pub bonds: Vec<Bond>,
    /// statistics of MC moves
    statistics: MoveStatistics,
    /// random number generator
    rng: ThreadRng,
    /// number of production sweeps per simulation repeat
    pub prod_sweeps: u32,
    /// number of equilibration sweeps per simulation repeat
    pub eq_sweeps: u32,
    /// number of simulation repeats
    pub repeats: u32,
    /// frequency of movie printing (per N sweeps)
    pub movie_freq: u32,
    /// path to output file for movie
    pub movie_file: String,
    /// frequency of total energy calculation and priting (per N sweeps)
    pub energy_freq: u32,
    /// frequency of MSD calculation (per N production sweeps)
    pub msd_freq: u32,
    /// number of repeats in one block for diffusion calculation
    pub diff_block: u32,
    /// dimensionality of the simulation
    pub dimensionality: Dimensionality,
    /// should the particles be treated like hard spheres?
    pub hard_spheres: bool,
    /// path to output file for MSD data
    pub msd_file: String,
    /// number of warnings raised during the simulation
    pub n_warnings: u32,
    /// total binding energy of the particle
    pub binding_energy: f64,
}

impl System {
    /// Creates a new System structure with fields filled by default values.
    pub fn new() -> System {
        let statistics = MoveStatistics {
            accepted: Vec::new(),
            rejected: Vec::new(),
        };

        let rng = rand::thread_rng();

        System {
            particles: Vec::new(),
            bonds: Vec::new(),
            statistics,
            rng,
            prod_sweeps: 15_000,
            eq_sweeps: 5_000,
            movie_freq: 0,
            movie_file: "movie".to_string(),
            repeats: 200,
            energy_freq: 0,
            dimensionality: Dimensionality::TWO,
            msd_freq: 100,
            diff_block: 50,
            hard_spheres: false,
            msd_file: "msd{{BLOCK_NUMBER}}.dat".to_string(),
            n_warnings: 0,
            binding_energy: 0.0,
        }
    }

    /// Adds particle to the particle list.
    pub fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);
        self.statistics.accepted.push(0);
        self.statistics.rejected.push(0);
    }

    /// Adds bonds to the bond list.
    pub fn add_bond(&mut self, bond: Bond) {
        self.bonds.push(bond);
    }

    /// Sets positions (and properties) of all particles to their initial values.
    fn set_positions(&mut self, positions: &Vec<Particle>) {
        self.particles = positions.clone();
    }

    /// Runs the simulation.
    ///
    /// ## Details
    /// This method performs R * (E + P) * (N) Monte Carlo moves, where R is the number of repeats,
    /// E is the number of equilibration sweeps, P is the number of production sweeps, and
    /// N is the number of particles in the system.
    ///
    /// Repeats are divided into a number of "blocks". In each block, MSD data are collected over the production run,
    /// and once all the simulation repeats of one block are finished, a diffusion coefficient is calculated for this block
    /// by fitting a line to the collected MSD data.
    ///
    /// Final diffusion coefficient is calculated by averaging the diffusion coefficients of the individual simulation blocks.
    ///
    pub fn run(&mut self) -> bool {
        let mut movie = match self.try_creating_movie_file() {
            Ok(x) => x,
            Err(_) => return false,
        };

        let init_positions = self.particles.clone();
        let now = Instant::now();

        println!(
            "Running simulation ({} repeats, {} equilibration sweeps, {} production sweeps)...",
            self.repeats, self.eq_sweeps, self.prod_sweeps
        );

        // prepare diffusion calculation
        let mut diffusion = if self.diff_block != 0 {
            Some(Diffusion::new(self))
        } else {
            None
        };
        let n_blocks = if self.diff_block != 0 {
            self.repeats / self.diff_block
        } else {
            0
        };

        // repeat the simulation N times, each time starting from the same configuration of particles
        for repeat in 1..=self.repeats {
            if self.diff_block == 0 || repeat % self.diff_block == 0 || self.energy_freq > 0 {
                println!(">> Running repeat {}...", repeat);
            }

            // return particles back to their initial positions
            self.set_positions(&init_positions);

            for sweep in 1..=(self.eq_sweeps + self.prod_sweeps) {
                // perform one MC sweep
                self.update();

                self.try_printing_energy(sweep);
                self.try_writing_movie(&mut movie, sweep, repeat);

                // calculate MSD for the current system configuration
                if self.diff_block != 0 && sweep % self.msd_freq == 0 {
                    if let Some(unwrapped_diff) = diffusion.as_mut() {
                        unwrapped_diff.calc_msd(&self.particles, sweep);
                    }
                }

                if sweep == (self.eq_sweeps + 1) {
                    for particle in self.particles.iter_mut() {
                        particle.closest_well = particle.get_closest_well();
                    }
                }

                if sweep > self.eq_sweeps {
                    self.binding_energy += self.energy_full_surface();
                    for particle in self.particles.iter_mut() {
                        if particle.binding != 0.0 {
                            particle.well_analyze(self.dimensionality);
                        }
                    }
                }
            }

            // calculate diffusion for the current block of simulations
            if self.diff_block != 0 && repeat % self.diff_block == 0 {
                if let Some(unwrapped_diff) = diffusion.as_mut() {
                    unwrapped_diff.normalize_msd();
                    unwrapped_diff.calc_diffusion();

                    // write MSD data into output file
                    let block_id = format!("{}", repeat / self.diff_block);
                    let curr_msd_file = self.msd_file.replace("{{BLOCK_NUMBER}}", &block_id);

                    if let Err(_) =
                        unwrapped_diff.write_msd(&curr_msd_file, repeat / self.diff_block)
                    {
                        eprintln!(
                            "Warning. Could not write MSD data into MSD file `{}`.",
                            &curr_msd_file
                        );
                        self.n_warnings += 1;
                    }

                    unwrapped_diff.clear_msd();
                }
            }
        }

        let elapsed_time = now.elapsed();

        // PRINT SIMULATION STATISTICS
        print!("\nSIMULATION FINISHED! ");
        if self.n_warnings == 0 {
            println!("No warnings have been raised.");
        } else {
            println!("{} WARNING(S) RAISED! Check the output!", self.n_warnings);
        }

        println!("Time elapsed: {:.4} s", elapsed_time.as_secs());
        println!(
            "Time per block: {:.4} s",
            elapsed_time.as_secs() as f64 / n_blocks as f64
        );
        println!(
            "Time per repeat: {:.4} ms",
            elapsed_time.as_millis() as f64 / self.repeats as f64
        );
        println!(
            "Time per sweep: {:.4} μs",
            elapsed_time.as_micros() as f64
                / ((self.eq_sweeps + self.prod_sweeps) * self.repeats) as f64
        );

        // calculating and printing total (average) diffusion
        if self.diff_block != 0 {
            println!("\nDiffusion Statistics: ");
            println!("Number of Repeats: {}", self.repeats);
            println!("Number of Blocks: {}", n_blocks);
            println!("Repeats per Block: {}", self.diff_block);

            if let Some(unwrapped_diff) = diffusion.as_mut() {
                for i in 0..n_blocks {
                    println!(
                        "Diffusion Constant in Block #{}: {:.6}",
                        i + 1,
                        unwrapped_diff.get_diff(i as usize) * DIFFUSION_MULTIPLIER
                    );
                }

                let (av, std) = unwrapped_diff.get_average_diffusion();

                println!(
                    "Mean Diffusion: {:.6} ± {:.6} [~68%]",
                    av * DIFFUSION_MULTIPLIER,
                    std * DIFFUSION_MULTIPLIER
                );
            }
        } else {
            println!("\nDiffusion calculation not requested.");
        }

        true
    }

    /// Performs one simulation sweep.
    pub fn update(&mut self) {
        // perform N Monte Carlo moves where N is the number of particles
        for _ in 0..self.particles.len() {
            // randomly select a particle to move
            let index = self.rng.gen_range(0..self.particles.len());
            self.move_particle(index);
        }
    }

    /// Calculates energy of a single particle.
    ///
    /// ## Details
    ///
    /// Energy of a particle consists of:
    ///
    /// a) interaction with the surface
    ///
    /// b) bonded interactions with other particles
    ///
    /// c) non-bonded interactions with other particles*
    ///
    /// *Particles interact like hard spheres.
    fn energy_particle(&self, particle_index: usize) -> f64 {
        let particle = &self.particles[particle_index];

        // calculate energy of this particle interacting with the surface
        let mut energy = particle.energy_surface();

        // calculate energy of bonds
        for bond in &self.bonds {
            if bond.particles.contains(&particle_index) {
                energy += bond.energy(&self.particles);
            }
        }

        // calculate energy of pairwise interactions
        if self.hard_spheres {
            for other_index in 0..self.particles.len() {
                if particle_index == other_index {
                    continue;
                }

                energy += System::energy_pairwise(
                    &self.particles[particle_index],
                    &self.particles[other_index],
                );
            }
        }

        energy
    }

    /// Calculates the total energy of particles interacting with the surface.
    pub fn energy_full_surface(&self) -> f64 {
        let mut energy = 0.0;
        for particle in &self.particles {
            energy += particle.energy_surface();
        }

        energy
    }

    /// Calculates energy of the entire system.
    pub fn energy_full(&self) -> f64 {
        // calculate energy originating from particle-surface interaction
        let mut energy = self.energy_full_surface();

        // calculate energy of bonds
        for bond in &self.bonds {
            energy += bond.energy(&self.particles);
        }

        // calculate energy of pairwise interactions
        if self.hard_spheres {
            for i in 0..self.particles.len() {
                for j in (i + 1)..self.particles.len() {
                    energy += System::energy_pairwise(&self.particles[i], &self.particles[j]);
                }
            }
        }

        energy
    }

    /// Calculates non-bonded interaction between two particles.
    /// Particles interact like hard spheres.
    fn energy_pairwise(p1: &Particle, p2: &Particle) -> f64 {
        let dist = System::distance(p1, p2);

        if dist < (p1.size + p2.size) {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Calculates distance between two particles.
    pub fn distance(p1: &Particle, p2: &Particle) -> f64 {
        ((p1.position[0] - p2.position[0]) * (p1.position[0] - p2.position[0])
            + (p1.position[1] - p2.position[1]) * (p1.position[1] - p2.position[1]))
            .sqrt()
    }

    /// Returns center of geometry for the particles.
    pub fn center(particles: &Vec<Particle>) -> [f64; 2] {
        let mut center = [0.0, 0.0];

        for particle in particles {
            center[0] += particle.position[0];
            center[1] += particle.position[1];
        }

        center[0] /= particles.len() as f64;
        center[1] /= particles.len() as f64;

        center
    }

    /// Performs a single Monte Carlo move consisting of particle translation.
    fn move_particle(&mut self, particle_index: usize) {
        // calculate the original energy of the particle
        let old_energy = self.energy_particle(particle_index);

        // save the old position of the particle
        let old_x = self.particles[particle_index].position[0];
        let old_y = self.particles[particle_index].position[1];

        // propose a move and update the position of the particle
        match self.dimensionality {
            Dimensionality::ONE => self.particles[particle_index].propose_move_1d(&mut self.rng),
            Dimensionality::TWO => self.particles[particle_index].propose_move_2d(&mut self.rng),
        }

        // calculate the new energy of the particle
        let new_energy = self.energy_particle(particle_index);

        // accept or reject the move based on Metropolis criterion
        if !System::metropolis(new_energy - old_energy, &mut self.rng) {
            self.particles[particle_index].position[0] = old_x;
            self.particles[particle_index].position[1] = old_y;
            self.statistics.rejected[particle_index] += 1;
        } else {
            self.statistics.accepted[particle_index] += 1;
        }
    }

    /// Based on energy_difference, decides whether an MC move should be accepted or rejected.
    fn metropolis(energy_difference: f64, rng: &mut ThreadRng) -> bool {
        if energy_difference < 0.0 {
            return true;
        }

        // kT is set to 1
        if rng.gen::<f64>() < (-energy_difference).exp() {
            return true;
        }

        false
    }

    /// Displays basic information about the system.
    pub fn display(&self) {
        println!("\nSIMULATION SETTINGS:");
        println!("Simulation Repeats: {}", self.repeats);
        println!("Simulation Repeats per Block: {}", self.diff_block);
        println!("Equilibration Sweeps: {}", self.eq_sweeps);
        println!("Production Sweeps: {}", self.prod_sweeps);
        println!("Movie Printing Frequency: {}", self.movie_freq);
        println!("Movie File: {}", self.movie_file);
        println!("Energy Printing Frequency: {}", self.energy_freq);
        println!("MSD Calculation Frequency: {}", self.msd_freq);
        println!("Dimensionality: {}", self.dimensionality);
        println!("Particles Hard Spheres?: {}", self.hard_spheres);
        println!("MSD File Pattern: {}", self.msd_file);

        println!("\nPARTICLES:");
        for (i, part) in self.particles.iter().enumerate() {
            println!("Particle {}", i);
            println!("{}", part);
        }

        let center = System::center(&self.particles);
        println!(
            "\nInitial center of geometry: {:.4} {:.4}",
            center[0], center[1]
        );

        println!("\nBONDS:");
        for bond in &self.bonds {
            println!("{}", bond);
        }
    }

    /// Writes a movie frame into an open file.
    pub fn write_movie_frame(
        &self,
        output: &mut File,
        repeat: u32,
        sweep: u32,
    ) -> Result<(), io::Error> {
        write!(
            output,
            "@ Repeat {} ; Sweep {} ; Particles {} ; Dimensionality {}\n",
            repeat,
            sweep,
            self.particles.len(),
            self.dimensionality
        )?;

        for (i, p) in self.particles.iter().enumerate() {
            write!(output, "{} {} {}\n", i, p.position[0], p.position[1])?;
        }

        Ok(())
    }

    /// Prints move statistics collected over the simulation
    pub fn print_statistics(&self) {
        self.statistics.report();
    }

    /// Prints average binding energy of the particle.
    pub fn print_energy(&self) {
        println!(
            "\nAverage binding energy: {}",
            self.binding_energy / (self.repeats * self.prod_sweeps) as f64
        );
    }

    pub fn print_residence_times(&self) {
        println!("\nAVERAGE RESIDENCE \"TIME\":");

        let mut total_time = 0.0f64;
        let mut n_binding = 0;
        for (i, particle) in self.particles.iter().enumerate() {
            print!("Particle #{}: ", i);

            if particle.binding == 0.0 {
                println!("NOT BINDING");
            } else {
                n_binding += 1;
                let time = particle.average_residence_time();
                total_time += time;
                println!("{:.4} sweeps", time)
            }
        }

        println!("Average: {:.4} sweeps", total_time / n_binding as f64);
    }

    /// Create a movie file, if requested.
    fn try_creating_movie_file(&self) -> Result<Option<File>, ()> {
        if self.movie_freq != 0 {
            // create movie file
            let movie = match File::create(&self.movie_file) {
                Ok(x) => x,
                Err(_) => {
                    eprintln!("\nError. File `{}` could not be created.", &self.movie_file);
                    return Err(());
                }
            };

            return Ok(Some(movie));
        }

        return Ok(None);
    }

    /// Calculates and prints total energy of the system if target sweep matches energy_freq.
    fn try_printing_energy(&self, sweep: u32) {
        if self.energy_freq != 0 && sweep % self.energy_freq == 0 {
            println!(
                ">>>> Sweep: {}. Total Energy: {:.4}",
                sweep,
                self.energy_full()
            );
        }
    }

    /// Write movie step, if target sweep matches movie_freq.
    fn try_writing_movie(&mut self, movie: &mut Option<File>, sweep: u32, repeat: u32) {
        if self.movie_freq != 0 && sweep % self.movie_freq == 0 {
            if let Some(m) = movie.as_mut() {
                if let Err(_) = self.write_movie_frame(m, repeat, sweep) {
                    eprintln!(
                        "Warning. Could not write sweep `{}` into movie file `{}`",
                        sweep, self.movie_file
                    );
                    self.n_warnings += 1;
                }
            }
        }
    }

    /// Checks that the simulation settings makes sense
    pub fn sanity_check(&mut self) -> bool {
        // check that the number of repeats is positive
        if self.repeats <= 0 {
            eprintln!(
                "Error. `repeats` is {} but it must be larger than zero.",
                self.repeats
            );
            return false;
        }

        // check that the number of equilibration sweeps is not zero
        if self.eq_sweeps == 0 {
            eprintln!("Warning. `eq_sweeps` is 0 which means that the simulation would have no equilibration phase.");
            eprintln!("That seems incorrect.");
            self.n_warnings += 1;
        }

        // check that the number of production sweeps is not zero
        if self.prod_sweeps == 0 {
            eprintln!("Error. `prod_sweeps` is 0 which means that the simulation would have no production phase.");
            eprintln!("That is not supported.");
            return false;
        }

        // check that the number of repeats is divisible by diff_block
        if self.diff_block != 0 && self.repeats % self.diff_block != 0 {
            eprintln!(
                "Error. `repeats` ({}) is not divisible by `diff_block` ({}).",
                self.repeats, self.diff_block
            );
            return false;
        }

        // check that msd frequency is sufficiently low to obtain at least some data
        if self.diff_block != 0 && self.msd_freq > self.prod_sweeps {
            eprintln!(
                "Warning. `msd_freq` ({}) is higher than `prod_sweeps` ({}).",
                self.msd_freq, self.prod_sweeps
            );
            eprintln!("Turning diffusion calculation off. No msd data will be collected.");
            self.diff_block = 0;
            self.n_warnings += 1;
        }

        // check that the particles forming the bonds actually exist
        for (i, bond) in self.bonds.iter().enumerate() {
            for j in 0..2 {
                if bond.particles[j] >= self.particles.len() {
                    eprintln!(
                        "Error. Bond {} defines a particle {} which does not exist.",
                        i, bond.particles[j]
                    );
                    return false;
                }
            }
        }

        // check that particles in 1D simulations have zero y-coordinates
        // if this is not the case, modify their coordinates and print a warning
        if self.dimensionality == Dimensionality::ONE {
            for (i, particle) in self.particles.iter_mut().enumerate() {
                if particle.position[1] != 0.0 {
                    eprintln!("Warning. Dimensionality is 1D and particle {} has non-zero y-coordinate ({}).", i, particle.position[1]);
                    eprintln!("Setting the coordinate to 0.");
                    particle.position[1] = 0.0;
                    self.n_warnings += 1;
                }
            }
        }

        // sanity checking hard spheres
        let mut all_point_like = true;
        for (i, particle) in self.particles.iter().enumerate() {
            if !self.hard_spheres {
                if particle.size != 0.0 {
                    eprintln!("Warning. `hard_spheres` is `no`, but particle {} has a defined size of {}.", i, particle.size);
                    eprintln!("Particle {} will be treated as point-like object.", i);
                    self.n_warnings += 1;
                }
            } else {
                if particle.size != 0.0 {
                    all_point_like = false;
                }
            }
        }

        if self.hard_spheres && all_point_like {
            eprintln!(
                "Warning. `hard_spheres` is `yes`, but all particles have a defined size of 0."
            );
            eprintln!(
                "Turning `hard_spheres` off. All particles will be treated as point-like objects."
            );
            self.hard_spheres = false;
            self.n_warnings += 1;
        }

        // check that msd file pattern contains {{BLOCK_NUMBER}}
        if self.diff_block != 0 && !self.msd_file.contains("{{BLOCK_NUMBER}}") {
            eprintln!(
                "Error. `msd_file` pattern ({}) does not contain '{{BLOCK_NUMBER}}'.",
                self.msd_file
            );
            eprintln!("Therefore, it can't be used.");
            return false;
        }

        // check that `wells_distance` is positive for all particles
        // check that `size` is non-negative for all particles
        for (i, particle) in self.particles.iter().enumerate() {
            if particle.wells_distance <= 0.0 {
                eprintln!("Error. Particle {} has `wells_distance` of {}. `wells_distance` must be positive.", i, particle.wells_distance);
                return false;
            }

            if particle.size < 0.0 {
                eprintln!(
                    "Error. Particle {} has `size` of {}. `size` must be positive or zero.",
                    i, particle.size
                );
                return false;
            }
        }

        // check that msd_freq is not zero
        if self.diff_block != 0 && self.msd_freq <= 0 {
            eprintln!(
                "Error. `msd_freq` is {} but it must be larger than zero.",
                self.msd_freq
            );
            return false;
        }

        true
    }
}

/*
*************************************
            UNIT TESTS
*************************************
*/

#[cfg(test)]
mod tests {

    use crate::particle::Particle;
    use crate::simulation::{Dimensionality, System};

    use crate::parser::parse_input;

    const INPUT_FILE: &str = "test_files/test_input_energy";
    const INPUT_FILE_MOVEMENT: &str = "test_files/test_input_movement";

    #[test]
    fn test_distance() {
        let coordinates = [
            [0.7654, 0.2763],
            [1.453, -1.0],
            [-0.298465, -0.652],
            [0.00001, 0.0],
            [0.48735, 12.02983],
            [5.0, 4.0],
            [-9.234, 9.321],
        ];

        let expected = [
            0.0000, 1.4497, 1.4119, 0.8137, 11.7568, 5.6390, 13.4831, 1.4497, 0.0000, 1.7857,
            1.7639, 13.0656, 6.1304, 14.8572, 1.4119, 1.7857, 0.0000, 0.7171, 12.7062, 7.0509,
            13.3905, 0.8137, 1.7639, 0.7171, 0.0000, 12.0397, 6.4031, 13.1205, 11.7568, 13.0656,
            12.7062, 12.0397, 0.0000, 9.2110, 10.0917, 5.6390, 6.1304, 7.0509, 6.4031, 9.2110,
            0.0000, 15.1960, 13.4831, 14.8572, 13.3905, 13.1205, 10.0917, 15.1960, 0.0000,
        ];

        let mut i = 0;
        for p1 in &coordinates {
            let particle1 = Particle::new(p1.clone(), 0.0, 0.1, 0.0, 1.0, 0.0, 0.0);

            for p2 in &coordinates {
                let particle2 = Particle::new(p2.clone(), 0.0, 0.1, 0.0, 1.0, 0.0, 0.0);

                let distance1 = System::distance(&particle1, &particle2);
                let distance2 = System::distance(&particle1, &particle2);

                assert!((distance1 - expected[i]).abs() < 0.0001);
                assert!((distance2 - expected[i]).abs() < 0.0001);

                i += 1;
            }
        }
    }

    #[test]
    fn test_center() {
        let system = parse_input(INPUT_FILE).expect("Could not find input file.");

        let expected = [0.26, -0.12];
        let calculated = System::center(&system.particles);

        assert!((expected[0] - calculated[0]).abs() < 0.0001);
        assert!((expected[1] - calculated[1]).abs() < 0.0001);
    }

    #[test]
    fn test_energy_pairwise() {
        let mut system = parse_input(INPUT_FILE).expect("Could not find input file.");

        // modify sizes of particles
        system.particles[0].size = 0.3;
        system.particles[2].size = 0.2;
        system.particles[3].size = 0.5;

        let expected = [
            f64::INFINITY,
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            f64::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let mut i = 0;

        for particle1 in &system.particles {
            for particle2 in &system.particles {
                assert_eq!(System::energy_pairwise(particle1, particle2), expected[i]);
                assert_eq!(System::energy_pairwise(particle2, particle1), expected[i]);

                i += 1;
            }
        }
    }

    #[test]
    fn test_set_positions() {
        let mut system = parse_input(INPUT_FILE).expect("Could not find input file.");

        system.hard_spheres = false;

        let orig_pos = system.particles.clone();

        for _ in 0..100 {
            system.update();
        }

        for i in 0..system.particles.len() {
            assert_ne!(system.particles[i].position[0], orig_pos[i].position[0]);
            assert_ne!(system.particles[i].position[1], orig_pos[i].position[1]);
        }

        system.set_positions(&orig_pos);

        for i in 0..system.particles.len() {
            assert_eq!(system.particles[i].position[0], orig_pos[i].position[0]);
            assert_eq!(system.particles[i].position[1], orig_pos[i].position[1]);
        }
    }

    #[test]
    fn test_metropolis() {
        let mut rng = rand::thread_rng();
        let energy_differences = [0.0, -1.5, -15.0, 0.001, 100.0];

        assert!(System::metropolis(energy_differences[0], &mut rng));
        assert!(System::metropolis(energy_differences[1], &mut rng));
        assert!(System::metropolis(energy_differences[2], &mut rng));
        // these two may in very rare cases fail
        assert!(System::metropolis(energy_differences[3], &mut rng));
        assert!(!System::metropolis(energy_differences[4], &mut rng));
    }

    #[test]
    fn test_energy_particle() {
        let mut system = parse_input(INPUT_FILE).expect("Could not find input file.");

        // hard spheres on
        system.hard_spheres = true;
        let manual_part0 = f64::INFINITY;
        assert_eq!(manual_part0, system.energy_particle(0));
        let manual_part1 = f64::INFINITY;
        assert_eq!(manual_part1, system.energy_particle(1));
        let manual_part2 =
            system.particles[2].energy_surface() + system.bonds[1].energy(&system.particles);
        assert!((manual_part2 - system.energy_particle(2)).abs() < 0.0001);
        let manual_part3 =
            system.particles[3].energy_surface() + system.bonds[2].energy(&system.particles);
        assert!((manual_part3 - system.energy_particle(3)).abs() < 0.0001);
        let manual_part4 =
            system.particles[4].energy_surface() + system.bonds[3].energy(&system.particles);
        assert!((manual_part4 - system.energy_particle(4)).abs() < 0.0001);

        // hard spheres off
        system.hard_spheres = false;
        let manual_part0 = system.particles[0].energy_surface()
            + system.bonds[0].energy(&system.particles)
            + system.bonds[1].energy(&system.particles)
            + system.bonds[3].energy(&system.particles);
        assert!((manual_part0 - system.energy_particle(0)).abs() < 0.0001);
        let manual_part1 = system.particles[1].energy_surface()
            + system.bonds[0].energy(&system.particles)
            + system.bonds[2].energy(&system.particles);
        assert!((manual_part1 - system.energy_particle(1)).abs() < 0.0001);
        let manual_part2 =
            system.particles[2].energy_surface() + system.bonds[1].energy(&system.particles);
        assert!((manual_part2 - system.energy_particle(2)).abs() < 0.0001);
        let manual_part3 =
            system.particles[3].energy_surface() + system.bonds[2].energy(&system.particles);
        assert!((manual_part3 - system.energy_particle(3)).abs() < 0.0001);
        let manual_part4 =
            system.particles[4].energy_surface() + system.bonds[3].energy(&system.particles);
        assert!((manual_part4 - system.energy_particle(4)).abs() < 0.0001);
    }

    #[test]
    fn test_energy_full() {
        let expected_energy = -1.1892628;

        let mut system = parse_input(INPUT_FILE).expect("Could not find input file.");

        // hard spheres on
        system.hard_spheres = true;
        assert_eq!(system.energy_full(), f64::INFINITY);

        // hard spheres on but particle 1 is moved away
        system.particles[1].position[1] = 0.3;

        let mut full_energy = 0.0;
        for (i, particle) in system.particles.iter().enumerate() {
            full_energy += particle.energy_surface();

            for j in (i + 1)..system.particles.len() {
                full_energy += System::energy_pairwise(particle, &system.particles[j]);
            }
        }

        for bond in &system.bonds {
            full_energy += bond.energy(&system.particles);
        }

        assert!((full_energy - system.energy_full()).abs() < 0.0001);
        assert!((expected_energy - system.energy_full()).abs() < 0.0001);

        println!("{full_energy}");

        // hard spheres off
        system.hard_spheres = false;

        let mut full_energy = 0.0;
        for particle in &system.particles {
            full_energy += particle.energy_surface();
        }

        for bond in &system.bonds {
            full_energy += bond.energy(&system.particles);
        }

        assert!((full_energy - system.energy_full()).abs() < 0.0001);
        assert!((expected_energy - system.energy_full()).abs() < 0.0001);
    }

    #[test]
    fn test_move_particle_1d() {
        let mut system = parse_input(INPUT_FILE_MOVEMENT).expect("Could not find input file.");

        system.dimensionality = Dimensionality::ONE;

        let original_positions = system.particles.clone();

        // particle 0 should move freely as it doesn't interact with anything
        system.move_particle(0);
        assert_ne!(
            system.particles[0].position[0],
            original_positions[0].position[0]
        );
        assert_eq!(
            system.particles[0].position[1],
            original_positions[0].position[1]
        );
        assert_eq!(system.statistics.accepted[0], 1);
        assert_eq!(system.statistics.rejected[0], 0);

        // particle 1 shouldn't move as it interacts strongly and is in local minimum
        // this may however still fail, albeit rarely
        system.move_particle(1);
        assert_eq!(
            system.particles[1].position[0],
            original_positions[1].position[0]
        );
        assert_eq!(
            system.particles[1].position[1],
            original_positions[1].position[1]
        );
        assert_eq!(system.statistics.accepted[1], 0);
        assert_eq!(system.statistics.rejected[1], 1);
    }

    #[test]
    fn test_move_particle_2d() {
        let mut system = parse_input(INPUT_FILE_MOVEMENT).expect("Could not find input file.");

        system.dimensionality = Dimensionality::TWO;

        let original_positions = system.particles.clone();

        // particle 0 should move freely as it doesn't interact with anything
        system.move_particle(0);
        assert_ne!(
            system.particles[0].position[0],
            original_positions[0].position[0]
        );
        assert_ne!(
            system.particles[0].position[1],
            original_positions[0].position[1]
        );
        assert_eq!(system.statistics.accepted[0], 1);
        assert_eq!(system.statistics.rejected[0], 0);

        // particle 1 shouldn't move as it interacts strongly and is in local minimum
        // this may however still fail, albeit rarely
        system.move_particle(1);
        assert_eq!(
            system.particles[1].position[0],
            original_positions[1].position[0]
        );
        assert_eq!(
            system.particles[1].position[1],
            original_positions[1].position[1]
        );
        assert_eq!(system.statistics.accepted[1], 0);
        assert_eq!(system.statistics.rejected[1], 1);
    }

    #[test]
    fn test_update() {
        let mut system = parse_input(INPUT_FILE).expect("Could not find input file.");

        system.update();

        // check that N MC moves have been performed
        assert_eq!(
            system.statistics.accepted.iter().sum::<u64>()
                + system.statistics.rejected.iter().sum::<u64>(),
            system.particles.len() as u64
        );
    }
}
