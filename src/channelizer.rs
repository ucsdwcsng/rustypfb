use libm::log;

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

pub fn npr_coeff(n: u128, l: u128, shiftpix: u128, k: Option<f64>, coeff: &mut Vec<Vec<f64>>) {

    let k: f64 = match k {
        None => {
            let ind = l as f64;
            let key = log(ind);
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
                2.5 * (LOOKUP_TABLE[low].1 + (LOOKUP_TABLE[high].1 - LOOKUP_TABLE[low].1)*(key - log(LOOKUP_TABLE[low].0)) / 
                (log(LOOKUP_TABLE[high].0) - log(LOOKUP_TABLE[low].0)))
            }
        }, 
        Some(val) => val
    };
    
}
