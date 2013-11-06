/**
See class comment #GlException.
*/

#ifndef OPENCLEXCEPTION_H
#define OPENCLEXCEPTION_H


#include <stdexcept> // std::domain_error
#include "computationkernel.h"

/**
*/
class OpenClException : public std::domain_error {
public:
    OpenClException( cl_int fft_error );

    OpenClException( cl_int fft_error, const char* const &message );

    cl_int getOpenClError() const;

    static std::string getErrorString( cl_int fft_error );

    static void check_error( cl_int fft_error );

    static void check_error( cl_int fft_error, const char* functionMacro,
                             const char* fileMacro, int lineMacro,
                             const char* callerMessage = 0);

protected:
    cl_int fft_error;
};

/**
OpenClException_SAFE_CALL() takes the return value from some OpenCL call
and passes it on to <code>OpenClException::check_error</code> with the
arguments filled with the macros __FUNCTION__, __FILE__ and __LINE__.

@see #OpenClException_CHECK_ERROR()
*/
#define OpenClException_SAFE_CALL( call ) \
    OpenClException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, #call)

#define OpenClException_CHECK_ERROR( call, message ) \
    OpenClException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, (message))

#define OpenClException_SYNC_CALL( call ) do { \
        OpenClException::check_error( ComputationSynchronize(), __FUNCTION__, __FILE__, __LINE__, "cudaThreadSynchronize() before " #call); \
        OpenClException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, #call); \
        OpenClException::check_error( ComputationSynchronize(), __FUNCTION__, __FILE__, __LINE__, "cudaThreadSynchronize() after " #call); \
    } while(0)


/**
OpenClException_ThreadSynchronize(...) calls clEnqueueBarrier() as a
#OpenClException_SAFE_CALL. Any extra parameters passed as ... should be string
literals, they are appended to an eventual exception message.

@see #OpenClException_SAFE_CALL( call )
*/
#define OpenClException_ThreadSynchronize( ... ) \
    OpenClException::check_error( ComputationSynchronize(), __FUNCTION__, __FILE__, __LINE__, "cudaThreadSynchronize() " __VA_ARGS__)


#endif // OPENCLEXCEPTION_H
