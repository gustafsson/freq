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
    glPushAttribContext ac;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_TOOLGLVIEW_H
