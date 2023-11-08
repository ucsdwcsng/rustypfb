use bessel_fun_sys::bessel_func;

use num::Complex;
use rustfft::{Fft, FftPlanner};

use std::iter::Chain;
use std::slice::Iter;
use std::sync::Arc;

fn channel_fn(ind: usize, nchannel: usize, nproto: usize, kbeta: f32) -> f32 {
    let ind_arg = ind as f32;
    let arg = -((nproto / 2) as f32) + (ind_arg + 1.0) / (nchannel as f32);
    let darg = (2.0 * ind_arg) / ((nchannel * nproto) as f32) - 1.0;
    let carg = kbeta * (1.0 - darg * darg).sqrt();
    (unsafe { bessel_func(carg) }) / (unsafe { bessel_func(kbeta) })
        * (if arg != 0.0 { arg.sin() / arg } else { 1.0 })
}

fn create_filter<const TWICE_TAPS: usize>(channels: usize) -> Vec<[f32; TWICE_TAPS]> {
    let mut result = vec![[0.0; TWICE_TAPS]; channels];
    let taps = TWICE_TAPS / 2;
    for chann_id in 0..channels {
        let buffer = &mut result[chann_id];
        for tap_id in 0..taps {
            let ind = tap_id * channels + chann_id;
            if chann_id < channels / 2 {
                buffer[2 * tap_id] = channel_fn(ind, channels, taps, 10.0);
            } else {
                buffer[2 * tap_id + 1] = channel_fn(ind, channels, taps, 10.0);
            }
        }
    }
    result
}

#[derive(Copy, Clone, Debug)]
struct Ring<T, const CAPACITY: usize> {
    head: usize,
    full: bool,
    buffer: [T; CAPACITY],
}

impl<T: Default + Copy, const CAPACITY: usize> Ring<T, CAPACITY> {
    fn new() -> Self {
        Self {
            head: 0,
            full: false,
            buffer: [T::default(); CAPACITY],
        }
    }

    #[inline]
    fn add(&mut self, element: T) {
        self.buffer[self.head] = element;
        self.head += 1;
        if self.head >= CAPACITY {
            self.head = 0;
            self.full = true;
        }
    }

    fn inner_iter(&self) -> Chain<Iter<'_, T>, Iter<'_, T>> {
        let initial = self.buffer[..self.head].iter();
        if self.full {
            return initial.chain(self.buffer[self.head..].iter());
        }
        initial.chain(self.buffer[0..0].iter())
    }
}

pub struct Channelizer<const TWICE_TAPS: usize> {
    channels: usize,
    fft: Arc<dyn Fft<f32>>,
    coeff: Vec<[f32; TWICE_TAPS]>,
    state: Vec<Ring<Complex<f32>, TWICE_TAPS>>,
    scratch: Vec<Complex<f32>>,
}

impl<const TWICE_TAPS: usize> Channelizer<TWICE_TAPS> {
    pub fn new(channels: usize) -> Self {
        Self {
            fft: FftPlanner::new().plan_fft_inverse(channels),
            coeff: create_filter::<TWICE_TAPS>(channels),
            state: vec![Ring::new(); channels / 2],
            scratch: vec![Complex::new(0.0, 0.0); channels],
            channels,
        }
    }

    pub fn process(&mut self, samples: &[Complex<f32>], output: &mut [Complex<f32>]) {
        self.state
            .iter_mut()
            .zip(samples.iter().rev())
            .for_each(|(ring, sample)| ring.add(*sample));

        output[..self.channels]
            .iter_mut()
            .zip(self.state.iter().chain(self.state.iter()))
            .zip(self.coeff.iter())
            .for_each(|((out, ring), coeff)| {
                *out = ring
                    .inner_iter()
                    .zip(coeff.iter())
                    .fold(Complex::new(0.0, 0.0), |accum, (state, coeff)| {
                        accum + state * coeff
                    })
            });

        self.fft
            .process_with_scratch(&mut output[..self.channels], &mut self.scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process() {
        const CHANNELS: usize = 1024;
        let mut channelizer = Channelizer::<256>::new(CHANNELS);

        // Example input signal: Let's just use a bunch of 1's for simplicity
        let input_signal = vec![Complex::new(1.0 as f32, 0.0); CHANNELS / 2];

        // Buffer for the channelizer output
        let mut output_buffer = vec![Complex::new(0.0 as f32, 0.0); CHANNELS];

        // Process the input signal
        let now = std::time::Instant::now();
        for _ in 1..1000 {
            channelizer.process(&input_signal, &mut output_buffer);
        }
        println!("{:?}", now.elapsed());

        println!("{:?}", &output_buffer[..2]);
    }
}
