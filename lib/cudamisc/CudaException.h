/**
See class comment #CudaException.
*/

#pragma once

#include <stdexcept> // std::domain_error
#include <cuda_runtime_api.h> // cudaThreadSynchronize() and cudaError
#include <cufft.h> // cufftResult_t

/**
CudaException is a C++ way of handling errors from Cuda. The cutil
macros <code>CUT_CHECK_ERROR</code> and <code>CUDA_SAFE_CALL</code>
are neat, but they execute an <code>exit(EXIT_FAILURE)</code> which
is not a very nice way to exit an application. It is sometimes nice 
to notify the user in other manners. There is also a possibility that
the application may continue execution even after a cuda error.
CudaException gives an application these possibilities by throwing
exceptions when a <code>cudaError</code> value is different from
<code>cudaSuccess</code>.
<p>
To throw CudaExceptions on errors you can use the static method 
#check_error() without any arguments, which calls <code>
cudaGetLastError()</code> to find out if an error has occured.
You may also specify where in your code the value was returned from 
a cuda function with <code>check_error(const char* fileMacro, 
int lineMacro)</code>. Or use the macro CudaException_CHECK_ERROR()
which uses <code>check_error(const char* fileMacro, int lineMacro)
</code> and fills the arguments with __FILE__ and __LINE__.
<p>
CudaException becomes even more practial if you, in your debugging 
environment, put a breakpoint on the <code>throw CudaException</code>
rows in the check_error methods in CudaException.cpp. That will make 
it easier to find and fix errors when you can back trace and check 
values of instance variables.

@see GlException Provides a similiar mechanism for OpenGl.

@author johan.b.gustafsson@gmail.com
*/
class CudaException : public std::runtime_error {
public:
	/**
	Creates a CudaException from a named <code>cudaError</code>.
	Then, <code>cudaGetErrorString(cudaError)</code> is used to get a
	textual description of the error.

	@param namedCudaError A cudaError different from cudaSuccess.
	*/
	CudaException( cudaError namedCudaError );

	/**
	Creates a CudaException from a named <code>cudaError</code> along
	with a textual message describing the error. The text may include
	info on where and why it occured. The text should preferably also
	include the result from <code>cudaGetErrorString(cudaError)
	</code> in some form.

	@param namedCudaError A cudaError different from cudaSuccess.
	@param message A textual message describing where and why this
        CudaException was created. std::domain_error is used to store
        a copy of the text.
	*/
	CudaException(cudaError namedCudaError, const char* const &message );

	/**
	@returns The cudaError which caused this CudaException to be 
	thrown.
	*/
	cudaError getCudaError() const;

	/**
	Calls <code>cudaGetLastError()</code> to see if something has 
	gone wrong in a previous call to cuda. If the result is different
	from <code>cudaSuccess</code> a CudaException is thrown.
	<code>cudaGetErrorString(cudaError)</code> is used to get a
	textual description of the error.
	
	@see #check_error( cudaError )
	@throws CudaException
	*/
	static void check_error( );

	/**
	Checks if the value of a cuda result code is different from 
	<code>cudaSuccess</code> and if it is throws a CudaException.
	<code>cudaGetErrorString(cudaError)</code> is used to get a
	textual description of the error.

	@param resultCode A cuda result code to verify.
	@see #check_error()
	@throws CudaException
	*/
	static void check_error( cudaError resultCode );

	/**
	Calls <code>cudaGetLastError()</code> to see if something has 
	gone wrong in a previous call to cuda. If the result is different
	from <code>cudaSuccess</code> a CudaException is thrown.
	<code>cudaGetErrorString(cudaError)</code> is used to get a
	textual description of the error. #CudaException_CHECK_ERROR()
	calls this method with the intented macros as arguments.
	
	@param functionMacro Should be __FUNCTION__, the name of the 
	function which called this method.
	@param fileMacro Should be __FILE__, the name of the file which
	contains the call to this method.
	@param lineMacro Should be __LINE__, the line number in the file
	spcified by <code>fileMacro</code>.
	@see #check_error( )
	@see #check_error( cudaError, const char*, int )
	@see #CudaException_CHECK_ERROR()
	@throws CudaException
	*/
	static void check_error( const char* functionMacro, 
		                     const char* fileMacro, int lineMacro,
							 const char* callerMessage = 0);

	/**
	Checks if the value of a cuda result code is different from 
	<code>cudaSuccess</code> and if it is throws a CudaException.
	<code>cudaGetErrorString(cudaError)</code> is used to get a
	textual description of the error.
	
	@param functionMacro Should be __FUNCTION__, the name of the 
	function which called this method.
	@param fileMacro Should be __FILE__, the name of the file which
	contains the call to this method.
	@param lineMacro Should be __LINE__, the line number in the file
	spcified by <code>fileMacro</code>.
	@param callerMessage If not null, precedes the exception message 
	with another caller defined text so that the exception message 
	will take the syntax of 
	<code>Could not create world {Cuda error in function 
	'worldCreation()' (file 'worldCreation.cpp' at line 123): Out of 
	memory}.</code>

	@see #check_error( )
	@see #check_error( const char*, const char*, int )
	@throws CudaException
	*/
	static void check_error( cudaError err, const char* functionMacro, 
		                     const char* fileMacro, int lineMacro,
							 const char* callerMessage = 0);

protected:
	cudaError namedCudaError;
};

class CufftException: public CudaException {
public:
    CufftException(cufftResult_t namedCufftError );
    CufftException(cufftResult_t namedCufftError, const char* const &message );

    cufftResult_t getCufftError() const { return (cufftResult_t)getCudaError(); }

    static const char* getErrorString(cufftResult_t);

    static void check_error( cufftResult_t err, const char* functionMacro,
                                 const char* fileMacro, int lineMacro,
                                                     const char* callerMessage = 0);
};


/**
CudaException_CHECK_ERROR() uses <code>cudaGetLastError(const char* 
fileMacro, int lineMacro)</code> and fills the arguments with 
__FUNCTION__, __FILE__ and __LINE__.
*/
#define CudaException_CHECK_ERROR() \
	CudaException::check_error(__FUNCTION__, __FILE__, __LINE__)

/**
As #CudaException_CHECK_ERROR with a caller defined message.
*/
#define CudaException_CHECK_ERROR_MSG( message ) \
	CudaException::check_error(__FUNCTION__, __FILE__, __LINE__, (message) )

/**
CudaException_SAFE_CALL() takes the return value from some cuda call
and passes it on to <code>check_error(cudaError err, const char*
fileMacro, int lineMacro)</code> with the arguments filled with the
macros __FUNCTION__, __FILE__ and __LINE__.

@see #CudaException_CHECK_ERROR()
*/
#define CudaException_SAFE_CALL( call ) \
	CudaException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, #call)

#define CudaException_SYNC_CALL( call ) do { \
        CudaException::check_error( cudaThreadSynchronize(), __FUNCTION__, __FILE__, __LINE__, "cudaThreadSynchronize() before " #call); \
	    CudaException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, #call); \
        CudaException::check_error( cudaThreadSynchronize(), __FUNCTION__, __FILE__, __LINE__, "cudaThreadSynchronize() after " #call); \
	} while(0)


/**
CudaException_ThreadSynchronize(...) calls cudaThreadSynchronize() as a 
#CudaException_SAFE_CALL. Any extra parameters passed as ... should be string
literals, they are appended to an eventual exception message.

@see #CudaException_SAFE_CALL( call )
*/
#define CudaException_ThreadSynchronize( ... ) \
	CudaException::check_error( cudaThreadSynchronize(), __FUNCTION__, __FILE__, __LINE__, "cudaThreadSynchronize() " __VA_ARGS__)


#define CufftException_SAFE_CALL( call ) \
        CufftException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, #call)

#define CufftException_SYNC_CALL( call ) do { \
        CudaException_ThreadSynchronize( "before " #call); \
        CufftException::check_error( (call), __FUNCTION__, __FILE__, __LINE__, #call); \
        CudaException_ThreadSynchronize( "after " #call); \
        } while(0)
