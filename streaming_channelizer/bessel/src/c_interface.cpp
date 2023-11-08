#include <cmath>
using std::cyl_bessel_if;

extern "C"{
    float bessel_func(float x)
    {
        return cyl_bessel_if(0.0, x);
    }
}