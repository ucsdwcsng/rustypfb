#include "offline_channelizer.cuh"
#include <cufft.h>


struct box {
    int start_time;
    int stop_time;
    int start_chann;
    int stop_chann;
    int box_id;

    box(int, int, int, int, int);
    box();
};

class synthesizer
{
    public:
    int nchannel;
    int ntaps;
    int nslice;

    cufftHandle *plans;

    synthesizer(int, int, int);
    void revert(cufftComplex*, box*, cufftHandle*, cufftComplex*, cufftComplex*, int);
    ~synthesizer();
};

void __global__ synthesize(cufftComplex*, box*, cufftHandle*, cufftComplex*, cufftComplex*, int);

