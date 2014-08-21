#include "glPushContext.h"
#ifndef GL_ES_VERSION_2_0

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

#endif // GL_ES_VERSION_2_0
