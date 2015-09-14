/**
See class comment #GlException.
*/

#pragma once

#include <stdexcept>
#include <boost/exception/all.hpp>

#ifndef __GL_H__
typedef unsigned int GLenum;
#endif

/**
GlException is a C++ way of handling errors from OpenGl. As OpenGl
works as a state machine that is not object oriented it is hard to
keep track of OpenGl errors without an external mechanism, this is
such a mechanism.
<p>
To throw GlExceptions on errors you can use the static method 
#check_error() without any arguments, which calls <code>glGetError()
</code> to find out if an error has occured. You may also specify 
where in your code the value was returned from a OpenGL function with
<code>check_error(const char* fileMacro, int lineMacro)</code>. Or 
use the macro GlException_CHECK_ERROR() which uses <code>check_error
(const char* fileMacro, int lineMacro)</code> and fills the arguments 
with __FILE__ and __LINE__.
<p>
GlException becomes even more practial if you, in your debugging 
environment, put a breakpoint on the <code>throw GlException</code>
rows in the check_error methods in GlException.cpp. That will make 
it easier to find and fix errors when you can back trace and check 
values of instance variables.


@see CudaException Provides a similiar mechanism for Nvidia Cuda.

@author johan.b.gustafsson@gmail.com
*/
class GlException : virtual public boost::exception, virtual public std::exception {
public:
    // A GLenum different from GL_NO_ERROR.
    typedef boost::error_info<struct namedGlError,GLenum> GlException_namedGlError;
    // gluErrorString(glError) is used to get a textual description of the error.
    typedef boost::error_info<struct namedGlErrorString,std::string> GlException_namedGlErrorString;
    // Arbitrary additional message.
    typedef boost::error_info<struct message,std::string> GlException_message;


	/**
	Calls <code>glGetError()</code> to see if something has gone 
	wrong in a previous call to OpenGL. If the result is different
	from <code>GL_NO_ERROR</code> a GlException is thrown.
	<code>gluErrorString(GLenum)</code> is used to get a
	textual description of the error.
	
	@see #check_error( GLenum)
	@throws GlException
	*/
	static void check_error( );

	/**
	Checks if the value of an OpenGL error code is different from
	<code>GL_NO_ERROR</code> and if it is throws a GlException.
	<code>gluErrorString(GLenum)</code> is used to get a
	textual description of the error.

	@param resultCode A GLenum error code to verify.
	@see #check_error()
	@throws GlException
	*/
	static void check_error( GLenum errorCode );

	/**
	Calls <code>glGetError()</code> to see if something has gone 
	wrong in a previous call to Open GL. If the result is different
	from <code>GL_NO_ERROR</code> a GlException is thrown.
	<code>gluErrorString(GLenum)</code> is used to get a
	textual description of the error. #GlException_CHECK_ERROR()
	calls this method with the intented macros as arguments.
	
    @param functionMacro Should be BOOST_CURRENT_FUNCTION, the name of the
	function which called this method.
	@param fileMacro Should be __FILE__, the name of the file which
	contains the call to this method.
	@param lineMacro Should be __LINE__, the line number in the file
	spcified by <code>fileMacro</code>.
	@see #check_error( )
	@see #check_error( GLenum, const char*, int )
	@see #GlException_CHECK_ERROR()
	@throws GlException
	*/
	static void check_error( const char* functionMacro, 
		                     const char* fileMacro, int lineMacro,
							 const char* callerMessage = 0);

	/**
	Checks if the value of a OpenGL error code is different from 
	<code>GL_NO_ERROR</code> and if it is throws a GlException.
	<code>gluErrorString(GLenum)</code> is used to get a
	textual description of the error.
	
    @param functionMacro Should be BOOST_CURRENT_FUNCTION, the name of the
	function which called this method.
	@param fileMacro Should be __FILE__, the name of the file which
	contains the call to this method.
	@param lineMacro Should be __LINE__, the line number in the file
	spcified by <code>fileMacro</code>.
	@param callerMessage If not null, precedes the exception message 
	with another caller defined text so that the exception message 
	will take the syntax of 
	<code>Could not create world {OpenGL error in function 
	'worldCreation()' (file 'worldCreation.cpp' at line 123): Out of 
	memory}.</code>
	@see #check_error( )
	@see #check_error( const char*, const char*, int )
	@throws GlException
	*/
	static void check_error( GLenum errorCode, const char* functionMacro, 
		                     const char* fileMacro, int lineMacro,
							 const char* callerMessage = 0);
};

#ifdef _DEBUG

/**
GlException_CHECK_ERROR() uses <code>check_error(const char* 
fileMacro, int lineMacro)</code> and fills the arguments with 
BOOST_CURRENT_FUNCTION, __FILE__ and __LINE__.

@see GlException#check_error(const char*, const char*, int)
*/
#define GlException_CHECK_ERROR() \
    GlException::check_error(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__)

/**
As #GlException_CHECK_ERROR with a caller defined message.
*/
#define GlException_CHECK_ERROR_MSG( message ) \
    GlException::check_error(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, (message) )

/**
GlException_SAFE_CALL() performs a GlException_CHECK_ERROR after
a call (to a function that may set glGetError()).

@see GlException_CHECK_ERROR
*/
#define GlException_SAFE_CALL( call ) do { \
	GlException_CHECK_ERROR_MSG( "Before " #call ); \
	call; \
	GlException_CHECK_ERROR_MSG( #call ); \
	} while(false)

#else

// don't call glGet* in release builds, not even glGetError
// OpenGL ES Programming Guide for iOS - Avoid Querying OpenGL ES State
#define GlException_CHECK_ERROR()
#define GlException_CHECK_ERROR_MSG(x)
#define GlException_SAFE_CALL(x) x

#endif
