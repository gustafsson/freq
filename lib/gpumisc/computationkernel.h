#ifndef COMPUTATIONKERNEL_H
#define COMPUTATIONKERNEL_H


#ifdef USE_CUDA
    #include "CudaException.h"
    #include "cudaUtil.h"
    #include "cudaglobalstorage.h"
    #include "cuda_runtime.h"


#elif defined(USE_OPENCL)
    #include "openclcontext.h"

    inline unsigned availableMemoryForSingleAllocation()
    {
        unsigned MB = 1 << 20;
        return 256 * MB;
    }


#else
    inline unsigned availableMemoryForSingleAllocation()
    {
        unsigned MB = 1 << 20;
        return 256 * MB;
    }
#endif


#ifdef USE_CUDA
    #define ComputationSynchronize() CudaException_ThreadSynchronize()
    #define ComputationCheckError() CudaException_CHECK_ERROR()
#elif defined(USE_OPENCL)
    #define ComputationSynchronize() clEnqueueBarrier(OpenCLContext::Singleton().getCommandQueue())
    #define ComputationCheckError() while(false){}
#else
    #define ComputationSynchronize() while(false){}
    #define ComputationCheckError() while(false){}    
#endif


#endif // COMPUTATIONKERNEL_H
