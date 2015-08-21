#ifndef GLSTATE_H
#define GLSTATE_H

#include "gl.h"

/**
 * @brief The GlState class should track OpenGL states and reduce the number of
 * redundant state changes being transferred to the OpenGL driver.
 *
 * The state changes are not transferred until a draw call or sync().
 */
namespace GlState
{
    void glEnableVertexAttribArray (GLuint index);
    void glDisableVertexAttribArray (GLuint index);

    void glDrawElements (GLenum mode, GLsizei count, GLenum type, const GLvoid *indices);
    void glDrawArrays (GLenum mode, GLint first, GLsizei count);

    /**
     * @brief sync isses necessary opengl state changes.
     */
    void sync ();

    /**
     * @brief lost_sync resets all state changes being tracked by GlState.
     */
    void lost_sync ();
}

#endif // GLSTATE_H
