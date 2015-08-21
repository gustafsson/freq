#include "glstate.h"

namespace GlState
{

struct S {
    bool enabledAttribArray[4]={false,false,false,false};
} next, prev;

bool is_synced=true;

void glEnableVertexAttribArray (GLuint index)
{
    if (index < 4)
    {
        if (prev.enabledAttribArray[index] == false)
            is_synced = false;
        next.enabledAttribArray[index] = true;
    }
    else
        ::glEnableVertexAttribArray (index);
}

void glDisableVertexAttribArray (GLuint index)
{
    if (index < 4)
    {
        if (prev.enabledAttribArray[index] == true)
            is_synced = false;
        next.enabledAttribArray[index] = false;
    }
    else
        ::glDisableVertexAttribArray (index);
}

void glDrawElements (GLenum mode, GLsizei count, GLenum type, const GLvoid *indices)
{
    sync();
    ::glDrawElements (mode, count, type, indices);
}

void glDrawArrays (GLenum mode, GLint first, GLsizei count)
{
    sync();
    ::glDrawArrays (mode, first, count);
}

void sync ()
{
    if (is_synced)
        return;
    is_synced = true;

    for (int i=0; i<4; i++)
        if (prev.enabledAttribArray[i] != next.enabledAttribArray[i])
        {
            prev.enabledAttribArray[i] = next.enabledAttribArray[i];
            if (next.enabledAttribArray[i])
                ::glEnableVertexAttribArray (i);
            else
                ::glDisableVertexAttribArray (i);
        }
}

void lost_sync ()
{
    for (int i=0; i<4; i++)
        if (next.enabledAttribArray[i])
            ::glEnableVertexAttribArray (i);
        else
            ::glDisableVertexAttribArray (i);

    prev = next;
}

}
