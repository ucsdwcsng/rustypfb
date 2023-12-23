#ifndef _SYNTHESIZER
#define _SYNTHESIZER_
#include <cufft.h>


struct box {
    int start_time;
    int stop_time;
    int start_chann;
    int stop_chann;
    int box_id;

    public:
    box(int, int, int, int, int);
    box();
};

class synthesizer
{
    public:
    int nchannel;
    int ntaps;
    int nslice;

    cufftHandle *input_plans;
    cufftHandle *downconvert_plans;

    synthesizer(int, int, int);
    void revert(cufftComplex*, box*, cufftComplex*, cufftComplex*, int, int);
    ~synthesizer();
};

void __global__ synthesize(cufftComplex*, box*, cufftComplex*, int);

#endif

