#ifndef HEIGHTMAP_RENDER_RENDERREGION_H
#define HEIGHTMAP_RENDER_RENDERREGION_H

#include "heightmap/position.h"
#include "glprojection.h"

class QOpenGLShaderProgram;

namespace Heightmap {
namespace Render {

class RenderRegion
{
public:
    RenderRegion(glProjection gl_projection);
    virtual ~RenderRegion();
    RenderRegion(const RenderRegion&)=delete;
    RenderRegion& operator=(const RenderRegion&)=delete;

    void render(Region r, bool drawcross=true);

private:
    glProjection gl_projection_;
    QOpenGLShaderProgram* program_ = 0;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERREGION_H
