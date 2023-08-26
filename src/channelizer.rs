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
This is the struct that preserves all the state that is required for channelizing a given 1D input IQ array.

fft_plan and fft_inverse_plan are FFT plans that will be initialized in the beginning, for
forward and inverse fft computation.

nsamples is the number of samples that this channelizer will analyze. This simplifies
internal buffer memory allocations and fft planning that need to happen. Set this
to a large number like 1000000.

nchannel is the number of channels.

ntaps is the number of filter taps per channel.

When performing convolutions, this will be doing it in full mode by default.
*/
pub struct polyphase_channelizer {
    nsamples: usize,
    nchannel: u128,
    nslice: u128,
    sample_rate: f64,
    fft_plan: Box<dyn Fft<f64>>,
    fft_inverse_plan: Box<dyn Fft<f64>>,
    output_buffer: Vec<Complex<f64>>,
    coeff: Vec<Complex<f64>>,
}

impl polyphase_channelizer {
    pub fn process(&mut self, inp: &Vec<Complex<f64>>) {
        self.output_buffer[0..self.nsamples].clone_from_slice(inp);
        self.output_buffer[self.nsamples..2 * self.nsamples].clone_from_slice(inp);
        add_sign(&mut self.output_buffer[self.nsamples..2 * self.nsamples]);
        add_phase(
            (self.nchannel / 2) as usize,
            2 * self.nsamples / (self.nchannel as usize),
            &mut self.output_buffer[self.nsamples..2 * self.nsamples],
        );

    }
}

/* This is the lookup table that is supposed to give optimal K for a given L. */
static LOOKUP_TABLE: [(f64, f64); 19] = [
    (8.0, 4.853),
    (10.0, 4.775),
    (12.0, 5.257),
    (14.0, 5.736),
    (16.0, 5.856),
    (18.0, 7.037),
    (20.0, 6.499),
    (22.0, 6.483),
    (24.0, 7.410),
    (26.0, 7.022),
    (28.0, 7.097),
    (30.0, 7.755),
    (32.0, 7.452),
    (48.0, 8.522),
    (64.0, 9.396),
    (96.0, 10.785),
    (128.0, 11.5),
    (192.0, 11.5),
    (256.0, 11.5),
];

/* Linearly interpolate between two values. */
pub fn interp_linear(x: (f64, f64), y: (f64, f64), val: f64) -> f64 {
    x.1 + (val - x.0) * (y.1 - x.1) / (y.0 - x.0)
}

/* Binary lookup over the LOOKUP_TABLE, followed by interpolation.
Port of numpy's linear interp function. */
pub fn lookup(key: f64) -> f64 {
    let mut low: usize = 0;
    let mut high: usize = 18;
    let mut mid: usize = 9;
    let mut diff: bool = (high - low == 1) || (high - low == 0);
    while !diff {
        if key < log(LOOKUP_TABLE[mid].0) {
            high = mid;
        } else if key > log(LOOKUP_TABLE[mid].0) {
            low = mid;
        }
        mid = (low + high) / 2;
        diff = (high - low == 1) || (high - low == 0);
    }
    if low == high {
        2.5 * LOOKUP_TABLE[low].1
    } else {
        let tup_1 = (log(LOOKUP_TABLE[low].0), LOOKUP_TABLE[low].1);
        let tup_2 = (log(LOOKUP_TABLE[high].0), LOOKUP_TABLE[high].1);
        2.5 * interp_linear(tup_1, tup_2, key)
    }
}

/* Port of get_ij_vector. */
pub fn get_ij_vector(
    height: usize,
    width: Option<usize>,
    pixel_size: (Option<f64>, Option<f64>),
    ij_vector: &mut Vec<Complex<f64>>,
) {
    let width = match width {
        None => height,
        Some(val) => val,
    };

    let pixel_size: (f64, f64) = match pixel_size {
        (None, Some(val)) => (val, val),
        (Some(val), None) => (val, val),
        (Some(val_1), Some(val_2)) => (val_1, val_2),
        (None, None) => (1.0, 1.0),
    };

    get_vector(height, ij_vector);

    for item in ij_vector.iter_mut() {
        *item *= pixel_size.0;
    }
}

/* Port of get_vector */
pub fn get_vector(n: usize, output: &mut Vec<Complex<f64>>) {
    for ind in 0..n {
        output.push(Complex {
            re: ((ind - output.len() / 2) as f64),
            im: 0.0,
        });
    }
}

/*
 * coeff should be a mutable borrow from a float vector of size m*l where m = n/2
 * reference should be a mutable borrow from a complex float vector of size m*l where m=n/2
 * This will be called once when the Channelizer is initialized, so don't bother
 * optimizing this very much.
 */
pub fn npr_coeff(
    n: u128,
    l: u128,
    shiftpix: f64,
    k: Option<f64>,
    coeff: &mut Vec<Complex<f64>>,
    reference: &mut Vec<Complex<f64>>,
    ij_reference: &mut Vec<Complex<f64>>,
) {
    let k: f64 = match k {
        None => {
            let ind = l as f64;
            let key = log(ind);
            lookup(key)
        }
        Some(val) => val,
    } as f64;

    let m: u128 = n / 2;
    let size = (m * l) as usize;

    for val in 0..size {
        let inter = (val as f64) / ((size) as f64);
        reference[val] = Complex {
            re: sqrt(0.5 * erfc(2.0 * (k as f64) * (m as f64) * inter - 0.5)),
            im: 0.0,
        };
    }

    let new_size = (size / 2) as usize;

    for val in 0..new_size {
        reference[size - val] = reference[1 + val];
    }

    get_ij_vector(size, None, (None, None), ij_reference);

    for item in ij_reference.iter_mut() {
        *item *= Complex {
            re: cos(2.0 * PI * shiftpix / (size as f64)),
            im: -sin(2.0 * PI * shiftpix / (size as f64)),
        };
    }

    ij_reference.rotate_right(size / 2);

    for (index, item) in reference.iter_mut().enumerate() {
        *item *= ij_reference[index];
    }

    let mut planner = FftPlanner::new();
    let inverse_plan = planner.plan_fft_inverse(size);

    inverse_plan.process(reference);

    for (index, item) in coeff.iter_mut().enumerate() {
        *item = Complex {
            re: reference[index].re,
            im: 0.0,
        };
    }

    coeff.rotate_right(size / 2);

    let norm: Complex<f64> = coeff.iter().sum();

    for item in coeff.iter_mut() {
        *item /= norm;
    }
}

// Implments flip_lr over the whole array.
pub fn flip_lr<T>(inp: &mut Vec<T>, m: usize, l: usize) {
    for ind in 0..m {
        let _ = &inp[ind * l..(ind + 1) * l].reverse();
    }
}

/*
 * Calculates the filter function for a given Polyphase channelizer instance.
 * Stores the FFT of the coefficient function, to help with the convnfft procedure.
 */
pub fn calc_fiter(n: u128, l: u128, k: Option<f64>, shift_px: f64, coeff: &mut Vec<Complex<f64>>) {
    let size = (l * n / 2) as usize;

    let reference = &mut vec![Complex { re: 0.0, im: 0.0 }; size];
    let ij_reference = &mut vec![Complex { re: 0.0, im: 0.0 }; size];

    npr_coeff(n, l, shift_px, k, coeff, reference, ij_reference);

    let sum_coeff: Complex<f64> = coeff.iter().sum();

    for item in coeff.iter_mut() {
        (*item) /= sum_coeff;
    }

    flip_lr(coeff, (n / 2) as usize, l as usize);

    let mut planner = FftPlanner::new();
    let plan = planner.plan_fft_forward(l as usize);

    coeff
        .par_chunks_mut(l as usize)
        .for_each(|x| plan.process(x));
}

/* Tacks in the sign for the input. Needs to be called everytime the input is processed. */
pub fn add_sign(x: &mut [Complex<f64>]) {
    x.par_iter_mut().enumerate().for_each(|(ind, item)| {
        (*item) *= Complex {
            re: (2 * ind - 1) as f64,
            im: 0.0,
        }
    });
}

/* Tacks in the phase for the input. Needs to be called everytime the input is processed. */
pub fn add_phase(chann: usize, chunk: usize, x: &mut [Complex<f64>]) {
    x.par_chunks_mut(chunk).enumerate().for_each(|(ind, item)| {
        item.par_iter_mut().for_each(|x| {
            *x *= Complex {
                re: cos(2.0 * (ind as f64) * PI / (chann as f64)),
                im: sin(2.0 * (ind as f64) * PI / (chann as f64)),
            }
        })
    });
}
