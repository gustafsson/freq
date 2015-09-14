#include "rendercamera.h"

namespace Tools {
namespace Support {

RenderCamera::RenderCamera()
    :
      xscale(0),
      zscale(0),
      orthoview(1)
{
}


float RenderCamera::
        effective_ry() const
{
    return fmod(fmod(r[1],360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(r[1],360)+360, 360)+45)/90))*orthoview;
}

} // namespace Support
} // namespace Tools
