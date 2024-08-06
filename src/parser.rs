// Released under MIT License.
// Copyright (c) 2023 Ladislav Bartos

use crate::bond::Bond;
use crate::particle::Particle;
use crate::simulation::{Dimensionality, System};
use std::fs::File;
use std::io::{BufRead, BufReader};

enum ParseBlock {
    None,
    System,
    Particles,
    Bonds,
}

/// Parses an input line that corresponds to system parameters.
fn parse_system_line(line: &str, system: &mut System) -> bool {
    let split: Vec<&str> = line.split_whitespace().collect();

    if split.len() != 2 {
        return false;
    };

    match split[0] {
        "repeats" => match split[1].parse::<u32>() {
            Ok(x) => system.repeats = x,
            Err(_) => return false,
        },

        "prod_sweeps" => match split[1].parse::<u32>() {
            Ok(x) => system.prod_sweeps = x,
            Err(_) => return false,
        },

        "eq_sweeps" => match split[1].parse::<u32>() {
            Ok(x) => system.eq_sweeps = x,
            Err(_) => return false,
        },

        "movie_freq" => match split[1].parse::<u32>() {
            Ok(x) => system.movie_freq = x,
            Err(_) => return false,
        },

        "movie_file" => system.movie_file = split[1].to_string(),

        "energy_freq" => match split[1].parse::<u32>() {
            Ok(x) => system.energy_freq = x,
            Err(_) => return false,
        },

        "msd_freq" => match split[1].parse::<u32>() {
            Ok(x) => system.msd_freq = x,
            Err(_) => return false,
        },

        "diff_block" => match split[1].parse::<u32>() {
            Ok(x) => system.diff_block = x,
            Err(_) => return false,
        },

        "dimensionality" => match split[1] {
            "1D" => system.dimensionality = Dimensionality::ONE,
            "2D" => system.dimensionality = Dimensionality::TWO,
            _ => return false,
        },

        "hard_spheres" => match split[1] {
            "no" => system.hard_spheres = false,
            "yes" => system.hard_spheres = true,
            _ => return false,
        },

        "msd_file" => system.msd_file = split[1].to_string(),

        _ => return false,
    }

    true
}

/// Parses an input line that corresponds to a particle. Note that the particle number is actually ignored.
fn parse_particle_line(line: &str) -> Result<Particle, ()> {
    let split: Vec<&str> = line.split_whitespace().collect();

    if split.len() != 9 {
        return Err(());
    }

    let mut values = [0.0f64; 8];

    for i in 0..8 {
        values[i] = match split[i + 1].parse::<f64>() {
            Ok(x) => x,
            Err(_) => return Err(()),
        };
    }

    Ok(Particle::new(
        [values[0], values[1]],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
    ))
}

/// Parses an input line that corresponds to a bond.
fn parse_bond_line(line: &str) -> Result<Bond, ()> {
    let split: Vec<&str> = line.split_whitespace().collect();

    if split.len() != 4 {
        return Err(());
    }

    let p1 = match split[0].parse::<usize>() {
        Ok(x) => x,
        Err(_) => return Err(()),
    };

    let p2 = match split[1].parse::<usize>() {
        Ok(x) => x,
        Err(_) => return Err(()),
    };

    let fc = match split[2].parse::<f64>() {
        Ok(x) => x,
        Err(_) => return Err(()),
    };

    let eq = match split[3].parse::<f64>() {
        Ok(x) => x,
        Err(_) => return Err(()),
    };

    Ok(Bond::new([p1, p2], fc, eq))
}

/// Parses input file.
pub fn parse_input(filename: &str) -> Result<System, ()> {
    let file = match File::open(filename) {
        Ok(x) => x,
        Err(_) => {
            eprintln!("\nError. Input file `{}` could not be read.", filename);
            return Err(());
        }
    };

    let reader = BufReader::new(file);

    let mut system = System::new();

    let mut block = ParseBlock::None;

    for raw_line in reader.lines() {
        let line = match raw_line {
            Ok(x) => x,
            Err(_) => {
                eprintln!("\nError. Could not read line in `{}`.", filename);
                return Err(());
            }
        };

        if line.trim() == "" || line.trim().chars().next().unwrap() == '#' {
            continue;
        }

        if line.contains("[system]") {
            block = ParseBlock::System;
            continue;
        }

        if line.contains("[particles]") {
            block = ParseBlock::Particles;
            continue;
        }

        if line.contains("[bonds]") {
            block = ParseBlock::Bonds;
            continue;
        }

        match block {
            ParseBlock::None => {
                eprintln!(
                    "\nError. Could not understand line `{}` in `{}`.",
                    line, filename
                );
                return Err(());
            }
            ParseBlock::Particles => {
                let particle = match parse_particle_line(&line) {
                    Ok(x) => x,
                    Err(_) => {
                        eprintln!("\nError. Could not parse line `{}` as a particle.", line);
                        return Err(());
                    }
                };

                system.add_particle(particle);
            }
            ParseBlock::Bonds => {
                let bond = match parse_bond_line(&line) {
                    Ok(x) => x,
                    Err(_) => {
                        eprintln!("\nError. Could not parse line `{}` as a bond.", line);
                        return Err(());
                    }
                };

                system.add_bond(bond);
            }
            ParseBlock::System => {
                if !parse_system_line(&line, &mut system) {
                    eprintln!(
                        "\nError. Could not parse line `{}` as system information.",
                        line
                    );
                    return Err(());
                }
            }
        }
    }

    Ok(system)
}
