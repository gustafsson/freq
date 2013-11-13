#include "GlException.h"

#include "stringprintf.h"
#include "gl.h"
#include "backtrace.h"

GlException::GlException( GLenum namedGlError )
:	runtime_error((const char*)gluErrorString( namedGlError )),
	namedGlError(namedGlError)
{}

GlException::GlException(GLenum namedGlError, const char* const &message )
:	runtime_error(message),
	namedGlError(namedGlError)
{}

GLenum GlException::getGlError() const {
	return namedGlError;
}

void GlException::check_error( ) {
	check_error( glGetError() );
}

void GlException::check_error( GLenum errorCode ) {
	if (GL_NO_ERROR != errorCode) {
		throw GlException(errorCode );
	}
}

void GlException::check_error( const char* functionMacro, 
	                     const char* fileMacro, int lineMacro,
						 const char* callerMessage) {
	check_error( glGetError(), functionMacro, fileMacro, 
		              lineMacro, callerMessage );
}

void GlException::check_error( GLenum errorCode, const char* functionMacro, 
		                     const char* fileMacro, int lineMacro, 
							 const char* callerMessage ) {
	if (GL_NO_ERROR != errorCode) {
        // Reset OpenGL error state as we handle this error with exceptions from here on
        GLenum cleared_error = glGetError();
		std::string context_error;
		if (cleared_error == GL_INVALID_OPERATION)
		{
			context_error = "No OpenGL context is currently active.\n";
		}

		if (callerMessage) {
			throw GlException(errorCode, printfstring(
                "%s\n%sOpenGL error: %s (Code 0x%x)\nFunction  : %s\nFile      : %s (Line %i)\nBacktrace: %s",
                callerMessage, context_error.c_str(), gluErrorString(errorCode), errorCode, functionMacro, fileMacro, lineMacro, Backtrace::make_string ().c_str ()).c_str());
		} else {
			throw GlException(errorCode, printfstring(
                "%sOpenGL error: %s (Code 0x%x)\nFunction  : %s\nFile      : %s (Line %i)\nBacktrace: %s",
                context_error.c_str(), gluErrorString(errorCode), errorCode, functionMacro, fileMacro, lineMacro, Backtrace::make_string ().c_str () ).c_str());
		}
	}
}
