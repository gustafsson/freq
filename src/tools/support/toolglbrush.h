#ifndef TOOLS_SUPPORT_TOOLGLVIEW_H
#define TOOLS_SUPPORT_TOOLGLVIEW_H

// gpumisc
#include "glPushContext.h"

namespace Tools {
namespace Support {

class ToolGlBrush {
public:
    ToolGlBrush(bool enabled=false);
    ~ToolGlBrush();

private:
#ifdef LEGACY_OPENGL
    glPushAttribContext ac;
#endif // LEGACY_OPENGL
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_TOOLGLVIEW_H
