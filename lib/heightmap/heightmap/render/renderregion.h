#ifndef HEIGHTMAP_RENDER_RENDERREGION_H
#define HEIGHTMAP_RENDER_RENDERREGION_H

#include "heightmap/position.h"
#include "shaderresource.h"
#include "glprojection.h"

class QOpenGLShaderProgram;

namespace Heightmap {
namespace Render {

class RenderRegion
{
public:
    RenderRegion(const glProjecter& gl_projecter);
    RenderRegion(const RenderRegion&)=delete;
    ~RenderRegion();

    RenderRegion& operator=(const RenderRegion&)=delete;

    void render(Region r, bool drawcross=true);

private:
    const glProjecter gl_projecter_;
    GLuint vbo_=0;
    ShaderPtr program_;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERREGION_H
