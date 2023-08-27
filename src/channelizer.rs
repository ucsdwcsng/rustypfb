use libm::cos;
use libm::erfc;
use libm::log;
use libm::sin;
use libm::sqrt;
use num::complex;
use num::Complex;
use rayon::prelude::*;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/*
 Streaming version of the polyphase channelizer.
 Updates outputs one slice across channels at a time.
 This is maximally decimating.
*/
pub struct streaming_maximally_decimated_channelizer {
    m: usize, // This is the number of channels.

    q: usize, // This is the number of coloumns in the polyphase form of the filter.

    // Filter coefficients in polyphase form
    coeff: Vec<Complex<f64>>,

    // FFT Plan
    fft_inverse_plan: Box<dyn Fft<f64>>,

    // Buffer to hold polyphase products
    buffer: Vec<Complex<f64>>,
}

impl streaming_maximally_decimated_channelizer {
    pub fn process(&mut self, inp: &Vec<Complex<f64>>) {
        let w = inp.clone();
        let s: Vec<Complex<f64>> = w.into_iter().rev().collect();
        for item in &mut self.buffer {
            *item = Complex{re:0.0, im:0.0};
        }
        self.buffer
            .par_chunks_mut(self.q)
            .enumerate()
            .for_each(|(ind, item)| {
                multiply(
                    &s[ind * self.m..(ind + 1) * self.m],
                    &self.coeff[(self.q - 1 - ind) * self.m..(self.q - ind) * self.m],
                    item,
                )
            });
        self.fft_inverse_plan.process(&mut self.buffer); 
    }
}

pub fn multiply(lhs: &[Complex<f64>], rhs: &[Complex<f64>], prod: &mut [Complex<f64>]) {
    for (ind, item) in prod.iter_mut().enumerate() {
        *item += lhs[ind] * rhs[ind];
    }
}
