#include "clfftkernelbuffer.h"

#include "TaskTimer.h"


CLFFTKernelBuffer::CLFFTKernelBuffer()
{

}


CLFFTKernelBuffer::~CLFFTKernelBuffer()
{
    for (PlanMap::iterator i=kernels.begin(); i!=kernels.end(); ++i)
	{
        clFFT_DestroyPlan(i->second);
	}
}


clFFT_Plan CLFFTKernelBuffer::getPlan(cl_context c, unsigned int n, cl_int& error)
{
    if (kernels.find(n) != kernels.end())
    {
        error = CL_SUCCESS;
        return kernels[n];
	}

    TaskTimer tt("Creating an OpenCL FFT compute plan for n=%u", n);

    clFFT_Dim3 ndim = { n, 1, 1 };
    clFFT_Plan plan = clFFT_CreatePlan(c, ndim, clFFT_1D, clFFT_InterleavedComplexFormat, &error);

    if (error == CL_SUCCESS)
        kernels[n] = plan;

	return plan;
}
