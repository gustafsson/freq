#ifndef TOOLS_SUPPORT_RENDERCAMERA_H
#define TOOLS_SUPPORT_RENDERCAMERA_H

#include "TAni.h"
#include "GLvector.h"

namespace Tools {
namespace Support {

class RenderCamera
{
public:
    RenderCamera();

    vectord  q, // camera focus point, i.e (10, 0, 0.5)
             p, // camera position relative focus point, i.e (0, 0, -6)
             r; // rotation around focus point

    float xscale;
    float zscale;
    floatAni orthoview;

    float effective_ry(); // take orthoview into account
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_RENDERCAMERA_H
