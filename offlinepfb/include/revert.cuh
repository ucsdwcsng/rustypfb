#ifndef _SYNTHESIZER
#define _SYNTHESIZER_
#include <cufft.h>
#include <vector>
using std::vector;

struct box {
    int time_start;
    int time_stop;
    int chann_start;
    int chann_stop;
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

    cufftComplex* scratch_space;
    cufftComplex* multiply_scratch_space;
    cufftComplex* output;
    cufftComplex** filters;

    vector<int> output_start_times;

    synthesizer(int, int, int);
    void reconstruct(cufftComplex*, box);
    void revert(cufftComplex*, box*, cufftComplex*, cufftComplex*, int);
    ~synthesizer();
};

#endif

