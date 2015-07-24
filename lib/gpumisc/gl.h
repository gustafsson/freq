#ifndef GL_H
#define GL_H

// Must include windows.h before gl.h
#ifdef _MSC_VER
#    define NOMINMAX
#    define WIN32_LEAN_AND_MEAN
#    define VC_EXTRALEAN
#    include <windows.h>
#endif

// OSX does not use glew.h nor <gl/*>
#ifdef __APPLE__
# include <QtGui> // include <QtGui/qopengl.h> by including the QtGui framework
# ifdef GL_ES_VERSION_2_0
// assumes EXT_texture_rg which is present in for instance iOS >= 5.0
#  define GL_RED GL_RED_EXT
#  define GL_R16F GL_R16F_EXT
#  define GL_HALF_FLOAT GL_HALF_FLOAT_OES
# endif
#include "gluerrorstring.h"
#else
#    include <GL/glew.h> // glew.h includes gl.h
#endif

#endif // GL_H
