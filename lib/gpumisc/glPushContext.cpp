#include "glPushContext.h"
#ifdef LEGACY_OPENGL

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

#else
int LEGACY_OPENGL_glPushMatrixContext;
#endif // LEGACY_OPENGL
