#include "glPushContext.h"

#include "GlException.h"

glPushMatrixContext::glPushMatrixContext( GLint kind )
    :   kind( kind )
{
    glMatrixMode( kind );
    glPushMatrix();
}


glPushMatrixContext::~glPushMatrixContext() {
    glMatrixMode( kind );
    glPopMatrix();
    glMatrixMode( GL_MODELVIEW );
}

