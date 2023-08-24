use libm::cos;
use libm::erfc;
use libm::log;
use libm::sin;
use libm::sqrt;
use num::Complex;
use rustfft::Fft;
use rustfft::FftPlanner;
use std::f64::consts::PI;

pub struct polyphase_channelizer {
    fft_inverse_plan: Box<dyn Fft<f64>>,
    fft_plan: Box<dyn Fft<f64>>,
    reference: Vec<Complex<f64>>,
    ij_reference: Vec<Complex<f64>>,
}

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

pub fn interp_linear(x: (f64, f64), y: (f64, f64), val: f64) -> f64 {
    x.1 + (val - x.0) * (y.1 - x.1) / (y.0 - x.0)
}

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
 */
pub fn npr_coeff(
    n: u128,
    l: u128,
    shiftpix: f64,
    k: Option<f64>,
    coeff: &mut Vec<f64>,
    channel: &mut polyphase_channelizer,
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
        channel.reference[val] = Complex {
            re: sqrt(0.5 * erfc(2.0 * (k as f64) * (m as f64) * inter - 0.5)),
            im: 0.0,
        };
    }

    let new_size = (size / 2) as usize;

    for val in 0..new_size {
        channel.reference[size - val] = channel.reference[1 + val];
    }

    get_ij_vector(size, None, (None, None), &mut channel.ij_reference);

    for item in channel.ij_reference.iter_mut() {
        *item *= Complex {
            re: cos(2.0 * PI * shiftpix / (size as f64)),
            im: -sin(2.0 * PI * shiftpix / (size as f64)),
        };
    }

    channel.ij_reference.rotate_right(size / 2);

    for (index, item) in channel.reference.iter_mut().enumerate() {
        *item *= channel.ij_reference[index];
    }

    (*(channel.fft_inverse_plan)).process(&mut channel.reference);

    for (index, item) in coeff.iter_mut().enumerate() {
        *item = channel.reference[index].re;
    }

    coeff.rotate_right(size / 2);

    let norm: f64 = coeff.iter().sum();

    for item in coeff.iter_mut() {
        *item /= norm;
    }
}
