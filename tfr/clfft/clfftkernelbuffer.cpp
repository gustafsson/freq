#include "clfftkernelbuffer.h"

CLFFTKernelBuffer::CLFFTKernelBuffer()
{

}

CLFFTKernelBuffer::~CLFFTKernelBuffer()
{
	int len = kernels.size();
	for(int i = 0; i < len; i++)
	{
		clFFT_DestroyPlan(kernels[i].second);
	}
}
clFFT_Plan CLFFTKernelBuffer::getPlan(cl_context c, unsigned int n, cl_int *error)
{
	int len = kernels.size();
	for(int i = 0; i < len; i++)
	{
		if(kernels[i].first == n)
		{
			*error = CL_SUCCESS;
			return kernels[i].second;
		}
	}
	clFFT_Dim3 ndim = { n, 1, 1 };
	clFFT_Plan plan = clFFT_CreatePlan(c, ndim, clFFT_1D, clFFT_InterleavedComplexFormat, error);
	kernels.push_back(std::make_pair(n, plan));

	return plan;
}
CLFFTKernelBuffer *CLFFTKernelBuffer::initialize()
{
	return SingletonP().get();
}