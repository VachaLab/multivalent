// Released under MIT License.
// Copyright (c) 2023 Ladislav Bartos

use core::f64::consts::PI;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::fmt;

/// Particle is a hard sphere or a point in space that can:
///
/// a) interact with the surface,
///
/// b) bond with other particles,
///
/// c) repulse other particles.
#[derive(Clone)]
pub struct Particle {
    /// coordinates of the particle position: x and y
    pub position: [f64; 2],
    /// binding strength of the particle to surface;
    /// corresponds to A/n
    pub binding: f64,
    /// maximal displacement of the particle per MC move
    max_disp: f64,
    /// size of the particle (radius of the sphere)
    pub size: f64,
    /// distance between the potential wells of the surface
    pub wells_distance: f64,
    /// shift of the sine term of the surface potential
    sin_shift: f64,
    /// shift of the cosine term of the surface potential
    cos_shift: f64,
}

impl Particle {
    /// Creates a new particle.
    pub fn new(
        position: [f64; 2],
        binding: f64,
        max_disp: f64,
        size: f64,
        wells_distance: f64,
        sin_shift: f64,
        cos_shift: f64,
    ) -> Particle {
        Particle {
            position,
            binding,
            max_disp,
            size,
            wells_distance,
            sin_shift,
            cos_shift,
        }
    }

    /// Calculates energy of particle-surface interaction.
    pub fn energy_surface(&self) -> f64 {
        self.binding
            * (1.0
                + (2.0 * PI * (self.position[0] / self.wells_distance + self.sin_shift)).sin()
                + (2.0 * PI * (self.position[1] / self.wells_distance + self.cos_shift)).cos())
    }

    /// Calculates energy of particle-surface interaction as if the binding affinity of the particle were 1.0.
    /// Used inside `position`.
    fn fake_energy_surface(&self) -> f64 {
        1.0 * (1.0
            + (2.0 * PI * (self.position[0] / self.wells_distance + self.sin_shift)).sin()
            + (2.0 * PI * (self.position[1] / self.wells_distance + self.cos_shift)).cos())
    }

    /// Calculate the position of the particle along the virtual z-axis.
    pub fn position(&self) -> f64 {
        self.fake_energy_surface()
    }

    /// Proposes a translation move for a particle in 2D space.
    ///
    /// ## Details
    /// MODIFIES THE COORDINATES OF THE PARTICLE.
    /// If the move is subsequently rejected, particle must be returned to the original position.
    pub fn propose_move_2d(&mut self, rng: &mut ThreadRng) {
        // get random displacement smaller than max_disp (maximal displacement)
        // this uniformly generates points inside a circle of selected radius (radius = max_disp)
        let r = self.max_disp * (rng.gen::<f64>()).sqrt();
        let theta = rng.gen::<f64>() * 2.0 * PI;

        // update the position of the particle
        self.position[0] += r * theta.cos();
        self.position[1] += r * theta.sin();
    }

    /// Proposes a translation move for a particle in 1D space.
    pub fn propose_move_1d(&mut self, rng: &mut ThreadRng) {
        // get random displacement in one dimension that is smaller than max_disp
        let mut dx = self.max_disp * rng.gen::<f64>();
        if rng.gen::<bool>() {
            dx *= -1.0;
        }

        self.position[0] += dx;
    }
}

/// Allows the usage of print* macros for Particle.
impl fmt::Display for Particle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Position: {}, {}\nBinding: {}\nMaximal Displacement: {}\nSize: {}\nPotential Wells Distance: {}\nSine shift: {}\nCosine shift: {}\n", 
                   self.position[0], self.position[1], self.binding, self.max_disp, self.size, self.wells_distance, self.sin_shift, self.cos_shift)
    }
}

/*
*************************************
            UNIT TESTS
*************************************
*/

#[cfg(test)]
mod tests {

    use crate::parser::parse_input;
    use crate::particle::Particle;

    const INPUT_FILE: &str = "test_files/test_input_energy";

    #[test]
    fn test_particle_surface_energy() {
        let system = parse_input(INPUT_FILE).expect("Could not find input file.");
        let expected = [-1.0, 3.0, -0.0, 0.25, 0.08244];
        for (i, particle) in system.particles.iter().enumerate() {
            assert!((particle.energy_surface() - expected[i]).abs() < 0.0001);
        }
    }

    #[test]
    fn test_particle_propose_move_1d() {
        let mut rng = rand::thread_rng();
        let n_moves = 10_000usize;

        let displacements = [0.1, 0.5, 1.0];

        for i in 0..3 {
            let mut particle = Particle::new([0.0, 0.0], 0.0, displacements[i], 0.0, 1.0, 0.0, 0.0);

            let orig_pos_x = particle.position[0];

            for _ in 0..n_moves {
                let old_pos_x = particle.position[0];

                particle.propose_move_1d(&mut rng);

                assert!((particle.position[0] - old_pos_x).abs() < particle.max_disp);
                assert_eq!(particle.position[1], 0.0);
            }

            //println!("Total difference: {}", particle.position[0] - orig_pos_x);
            // this may in very rare cases fail
            assert!((particle.position[0] - orig_pos_x).abs() < particle.max_disp * 200.0);
        }
    }

    #[test]
    fn test_particle_propose_move_2d() {
        let mut rng = rand::thread_rng();
        let n_moves = 10_000usize;

        let displacements = [0.1, 0.5, 1.0];

        for i in 0..3 {
            let mut particle = Particle {
                position: [0.0, 0.0],
                binding: 0.0,
                max_disp: displacements[i],
                size: 0.0,
                wells_distance: 1.0,
                sin_shift: 0.0,
                cos_shift: 0.0,
            };

            let orig_pos_x = particle.position[0];
            let orig_pos_y = particle.position[1];

            for _ in 0..n_moves {
                let old_pos_x = particle.position[0];
                let old_pos_y = particle.position[1];

                particle.propose_move_2d(&mut rng);

                let dx = particle.position[0] - old_pos_x;
                let dy = particle.position[1] - old_pos_y;

                assert!((dx * dx + dy * dy).sqrt().abs() < particle.max_disp);
            }

            //println!("Total difference: {} {}", particle.position[0] - orig_pos_x, particle.position[1] - orig_pos_y);
            // these two may in very rare cases fail
            assert!((particle.position[0] - orig_pos_x).abs() < particle.max_disp * 200.0);
            assert!((particle.position[1] - orig_pos_y).abs() < particle.max_disp * 200.0);
        }
    }
}
