#include "toolglbrush.h"
#include "exceptionassert.h"
#include "glstate.h"

namespace Tools {
namespace Support {

ToolGlBrush::
    ToolGlBrush(bool enabled)
{
#ifdef LEGACY_OPENGL
    enabled = false;

    GlState::glEnable (GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);
#else
    EXCEPTION_ASSERTX(false, "requires LEGACY_OPENGL");
#endif
}


ToolGlBrush::
    ~ToolGlBrush()
{
#ifdef LEGACY_OPENGL
    GlState::glDisable (GL_BLEND);
    glDepthMask(true);
#endif // LEGACY_OPENGL
}

} // namespace Support
} // namespace Tools
