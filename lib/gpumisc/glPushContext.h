#ifndef _GLPUSHCONTEXT_H_
#define _GLPUSHCONTEXT_H_

#pragma once

#include "GlException.h"
#include "gl.h"

class glPushMatrixContext
{
public:
    glPushMatrixContext( GLint kind )
        :   kind( kind )
    {
        GlException_CHECK_ERROR();

        glMatrixMode( kind );
        glPushMatrix();

        GlException_CHECK_ERROR();
    }

    ~glPushMatrixContext() {
        GlException_CHECK_ERROR();

        glMatrixMode( kind );
        glPopMatrix();
        glMatrixMode( GL_MODELVIEW );

        GlException_CHECK_ERROR();
    }
private:
    GLint kind;
};

class glPushAttribContext {
public:
   glPushAttribContext( GLbitfield mask = GL_ALL_ATTRIB_BITS ) { glPushAttrib(mask); }
   ~glPushAttribContext() { glPopAttrib(); }
};

#endif // _GLPUSHCONTEXT_H_
