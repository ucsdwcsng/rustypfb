use libm::log;
use libm::erfc;
use libm::sqrt;

static LOOKUP_TABLE: [(f64, f64);19] = [
        (8.0, 4.853),
        (10.0, 4.775),
        (12.0,   5.257),
        (14.0,   5.736),
        (16.0,   5.856),
        (18.0,   7.037),
        (20.0,   6.499),
        (22.0,   6.483),
        (24.0,   7.410),
        (26.0,   7.022),
        (28.0,   7.097),
        (30.0,   7.755),
        (32.0,   7.452),
        (48.0,   8.522),
        (64.0,   9.396),
        (96.0,   10.785),
        (128.0,  11.5),
        (192.0,  11.5),
        (256.0,  11.5)
    ];

pub fn interp_linear(x:(f64, f64), y:(f64, f64), val:f64) -> f64 {
    x.1 + (val - x.0) * (y.1 - x.1) / (y.0 - x.0)
}

pub fn lookup(key: f64) -> f64 {
    let mut low: usize = 0;
    let mut high: usize = 18;
    let mut mid: usize = 9;
    let mut diff:bool = (high - low == 1) || (high - low == 0);
    while !diff
    {
        if key < log(LOOKUP_TABLE[mid].0){
            high = mid;
        }else if key > log(LOOKUP_TABLE[mid].0){
            low  = mid;
        }
        mid = (low + high) / 2;
        diff = (high - low == 1) || (high - low == 0);
    }
    if low == high {
        2.5 * LOOKUP_TABLE[low].1
    }else{
        let tup_1 = (log(LOOKUP_TABLE[low].0), LOOKUP_TABLE[low].1);
        let tup_2 = (log(LOOKUP_TABLE[low].0), LOOKUP_TABLE[low].1);
        interp_linear(tup_1, tup_2, key)
    }
}

pub fn npr_coeff(n: u128, l: u128, shiftpix: u64, k: Option<f64>, coeff: &mut Vec<Vec<f64>>) {

    let k:f64 = match k {
        None => {
            let ind = l as f64;
            let key = log(ind);
            lookup(key)         
        }, 
        Some(val) => val
    } as f64;

    let m:u128 = n / 2;

    let f:Vec<f64> = (0..m*l-1).map(|x| (x as f64) / ((m*l) as f64)).collect();    
    let g:Vec<f64> = f.iter().map(|x| sqrt(0.5 * erfc(2.0*(k as f64)*(m as f64)*x - 0.5))).collect();
}
