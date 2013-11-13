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
#    include <OpenGL/gl.h>
#    include <OpenGL/glu.h>
#else
#    include <GL/glew.h> // glew.h includes gl.h
#endif

#endif // GL_H
