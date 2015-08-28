#include "glstate.h"
#include <set>

namespace GlState
{

struct S {
    std::set<GLenum> caps;
    bool enabledAttribArray[4]={false,false,false,false};

    S()
    {
        reset();
    }

    void reset() {
        caps.clear ();
        // GL_DITHER and GL_MULTISAMPLE are enabled by default
        // https://www.opengl.org/sdk/docs/man2/xhtml/glEnable.xml
        caps.insert (GL_DITHER);
#ifndef GL_ES_VERSION_2_0
        caps.insert (GL_MULTISAMPLE);
#endif
        for (int i=0; i<0x4;i++)
            enabledAttribArray[i]=false;
    }
} next, current;

bool is_synced=true;


void glEnable (GLenum cap, bool now)
{
    if (now)
    {
        if (current.caps.count (cap) == 0)
        {
            current.caps.insert (cap);
            ::glEnable (cap);
        }
    }
    else
    {
        if (current.caps.count (cap) == 0)
            is_synced = false;
        next.caps.insert (cap);
    }
}

void glDisable (GLenum cap, bool now)
{
    if (now)
    {
        if (current.caps.count (cap) == 1)
        {
            current.caps.erase (cap);
            ::glDisable (cap);
        }
        next.caps.erase (cap);
    }
    else
    {
        if (current.caps.count (cap) == 1)
            is_synced = false;
        next.caps.erase (cap);
    }
}

void glEnableVertexAttribArray (GLuint index)
{
    if (index < 4)
    {
        if (current.enabledAttribArray[index] == false)
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
        if (current.enabledAttribArray[index] == true)
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
        if (current.enabledAttribArray[i] != next.enabledAttribArray[i])
        {
            current.enabledAttribArray[i] = next.enabledAttribArray[i];
            if (next.enabledAttribArray[i])
                ::glEnableVertexAttribArray (i);
            else
                ::glDisableVertexAttribArray (i);
        }

    bool enabled_changed = false;
    for (const auto& v : next.caps)
    {
        if (!current.caps.count (v))
        {
            enabled_changed = true;
            ::glEnable (v);
        }
    }

    for (const auto& v : current.caps)
    {
        if (!next.caps.count (v))
        {
            enabled_changed = true;
            ::glDisable (v);
        }
    }

    if (enabled_changed)
        current.caps = next.caps;
}

void setGlIsEnabled (GLenum cap, bool v)
{
    if (v)
        current.caps.insert(cap);
    else
        current.caps.erase(cap);
}

void assume_default_gl_states ()
{
    current.reset ();
}

}
