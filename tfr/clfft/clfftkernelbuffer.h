#ifndef CLFFTKERNELBUFFER_H
#include "clFFT.h"
#include "openclcontext.h"
#include "HasSingleton.h"
#include <utility>
#include <map>

class CLFFTKernelBuffer: public HasSingleton<CLFFTKernelBuffer>
{
public:
    ~CLFFTKernelBuffer();

    clFFT_Plan getPlan(cl_context c, unsigned int n, cl_int& error);

protected:
    typedef std::map<unsigned int, clFFT_Plan> PlanMap;
    PlanMap kernels;

private:
    friend class HasSingleton<CLFFTKernelBuffer>;
    CLFFTKernelBuffer(); // CL_DEVICE_TYPE_GPU is required for clFFT!
};

#endif //CLFFTKERNELBUFFER_H
