#ifndef _GLPUSHCONTEXT_H_
#define _GLPUSHCONTEXT_H_

#pragma once

#include "gl.h"
#ifdef LEGACY_OPENGL

#include <boost/noncopyable.hpp>

class glPushMatrixContext: boost::noncopyable
{
public:
    glPushMatrixContext( GLint kind );
    ~glPushMatrixContext();

private:
    GLint kind;
};

class glPushAttribContext: boost::noncopyable
{
public:
   glPushAttribContext( GLbitfield mask = GL_ALL_ATTRIB_BITS ) { glPushAttrib(mask); }
   ~glPushAttribContext() { glPopAttrib(); }
};

#endif // LEGACY_OPENGL
#endif // _GLPUSHCONTEXT_H_
