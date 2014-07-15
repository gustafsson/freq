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
# include <qgl.h>
//# include <OpenGL/gl.h>
# ifdef GL_ES_VERSION_2_0
#  define GL_RED GL_RED_EXT // assumes EXT_texture_rg which is present in for instance iOS >= 5.0
# endif
inline const char* gluErrorString(int) {
    return "(gluErrorString not implemented)";
}
#else
#    include <GL/glew.h> // glew.h includes gl.h
#endif

#endif // GL_H
