#ifndef _GLPUSHCONTEXT_H_
#define _GLPUSHCONTEXT_H_

#pragma once

#include "gl.h"

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

#endif // _GLPUSHCONTEXT_H_
