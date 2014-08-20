#include "CudaException.h"
#include "stringprintf.h"

#include <cuda.h>

#include <sstream>

CudaException::CudaException( cudaError namedCudaError )
:	runtime_error((const char*)cudaGetErrorString( namedCudaError )),
	namedCudaError(namedCudaError)
{}

CudaException::CudaException(cudaError namedCudaError, const char* const &message )
:	runtime_error(message),
	namedCudaError(namedCudaError)
{}

cudaError CudaException::getCudaError() const {
	return namedCudaError;
}

void CudaException::check_error( ) {
	check_error( cudaGetLastError() );
}

void CudaException::check_error( cudaError resultCode ) {
	if (cudaSuccess != resultCode) {
		throw CudaException(resultCode );
        }
}

void CudaException::check_error( const char* functionMacro, 
	                     const char* fileMacro, int lineMacro,
						 const char* callerMessage)
{
	check_error( cudaGetLastError(), functionMacro, fileMacro, 
		              lineMacro, callerMessage );
}

void CudaException::check_error( cudaError errorCode, const char* functionMacro, 
	                     const char* fileMacro, int lineMacro, 
						 const char* callerMessage ) {
	if (cudaSuccess != errorCode) {
		int c=0;
		cudaGetDeviceCount(&c); // reset cudaGetLastError

		// Invalid Device Function
		//  the graphics driver is not compatible with the cuda version that the application was compiled for.
		//
		// Fill in...

        // Reset cuda error state as we handle this error with exceptions from here on
        cudaError cleared_error = cudaGetLastError();
		std::stringstream context_error;

        CUdevice current_device;
        cudaError_enum ctx_err = cuCtxGetDevice( &current_device );
        if( CUDA_ERROR_INVALID_CONTEXT == ctx_err)
            context_error << "No valid CUDA context is currently active.\n";
		else if (cleared_error != cudaSuccess)
		{
			context_error << "Couldn't clear cuda error, got " << cudaGetErrorString(cleared_error) << " (Code " << cleared_error << ")\n";
		}


		if(callerMessage) {
			throw CudaException(errorCode, printfstring(
                                "%s\n%sCuda error: %s (Code %d)\nFunction  : %s\nFile      : %s (Line %d)",
					callerMessage, context_error.str().c_str(), cudaGetErrorString(errorCode), errorCode, functionMacro, fileMacro, lineMacro ).c_str());
		} else {
			throw CudaException(errorCode, printfstring(
                                "%sCuda error: %s (Code %d)\nFunction  : %s\nFile      : %s (Line %d)",
					context_error.str().c_str(), cudaGetErrorString(errorCode), errorCode, functionMacro, fileMacro, lineMacro ).c_str());
		}
	}
}

CufftException::CufftException(cufftResult_t namedCufftError )
    :   CudaException((cudaError)namedCufftError, getErrorString( namedCufftError ))
{}

CufftException::CufftException(cufftResult_t namedCufftError, const char* const &message )
    :   CudaException((cudaError)namedCufftError, message)
{}

const char* CufftException::
         getErrorString(cufftResult_t t)
{
    switch(t) {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    default:        return "CUFFT, UNKNOWN ERROR";
    }
}


void CufftException::
        check_error( cufftResult_t errorCode, const char* functionMacro,
                             const char* fileMacro, int lineMacro,
                                                 const char* callerMessage ) {
        if ((cufftResult_t)cudaSuccess != errorCode) {

            // Reset cuda error state as we handle this error with exceptions from here on
            cudaGetLastError();

                if(callerMessage) {
                        throw CufftException(errorCode, printfstring(
                                "%s\nCufft error: %s (Code %d)\nFunction  : %s\nFile      : %s (Line %d)",
                                        callerMessage, getErrorString(errorCode), errorCode, functionMacro, fileMacro, lineMacro ).c_str());
                } else {
                        throw CufftException(errorCode, printfstring(
                                "Cufft error: %s (Code %d)\nFunction  : %s\nFile      : %s (Line %d)",
                                        getErrorString(errorCode), errorCode, functionMacro, fileMacro, lineMacro ).c_str());
                }
        }
}
