#ifndef CLFFTKERNELBUFFER_H
#include "clFFT.h"
#include "openclcontext.h"
#include "HasSingleton.h"
#include <utility>
#include <vector>

class CLFFTKernelBuffer: public HasSingleton<CLFFTKernelBuffer>
{
public:
    ~CLFFTKernelBuffer();

	clFFT_Plan getPlan(cl_context c, unsigned int n, cl_int *error);
	static CLFFTKernelBuffer *initialize();

protected:
	typedef std::pair<unsigned int, clFFT_Plan> int_pair__;
	std::vector<int_pair__> kernels;

private:
    friend class HasSingleton<CLFFTKernelBuffer>;
    CLFFTKernelBuffer(); // CL_DEVICE_TYPE_GPU is required for clFFT!
};

#endif //CLFFTKERNELBUFFER_H
