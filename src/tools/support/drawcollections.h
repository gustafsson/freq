#ifndef TOOLS_SUPPORT_DRAWCOLLECTIONS_H
#define TOOLS_SUPPORT_DRAWCOLLECTIONS_H

#include "glframebuffer.h"

namespace Tools {
class RenderModel;

namespace Support {

class DrawCollections
{
public:
    DrawCollections(RenderModel* model);

    void drawCollections(GlFrameBuffer* fbo, float yscale);

private:
    RenderModel* model;
    std::vector<tvector<4> > channel_colors;

    void drawCollection(int channel, float yscale);
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_DRAWCOLLECTIONS_H
