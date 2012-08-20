#include "toolglbrush.h"

namespace Tools {
namespace Support {

ToolGlBrush::
    ToolGlBrush(bool enabled)
{
    enabled = false;

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(false);
    glColor4f( 0, 0, 0, enabled ? .5 : 0.2);
}


ToolGlBrush::
    ~ToolGlBrush()
{
    glDepthMask(true);
}


} // namespace Support
} // namespace Tools
