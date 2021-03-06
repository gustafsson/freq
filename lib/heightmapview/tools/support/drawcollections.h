#ifndef TOOLS_SUPPORT_DRAWCOLLECTIONS_H
#define TOOLS_SUPPORT_DRAWCOLLECTIONS_H

#include "glframebuffer.h"
#include "glprojection.h"
#include "heightmap/render/renderblock.h"

namespace Heightmap { class Collection; }

class QOpenGLShaderProgram;
namespace Tools {
class RenderModel;

namespace Support {

class DrawCollections
{
public:
    DrawCollections(RenderModel* model);
    DrawCollections(const DrawCollections&) = delete;
    DrawCollections& operator=(const DrawCollections&) = delete;
    ~DrawCollections();

    void drawCollections(const glProjection& gl_projection, GlFrameBuffer* fbo, float yscale);

private:
    RenderModel* model;
    std::vector<tvector<4> > channel_colors;
    std::unique_ptr<QOpenGLShaderProgram> m_program = 0;
    Heightmap::Render::RenderBlock render_block;
    GLuint vbo_, attribVertices, attribTex;

    void drawCollection(const glProjection& gl_projection, shared_state<Heightmap::Collection> collection, int collection_i, float yscale, float L);
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_DRAWCOLLECTIONS_H
