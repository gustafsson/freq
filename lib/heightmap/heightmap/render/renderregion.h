#ifndef HEIGHTMAP_RENDER_RENDERREGION_H
#define HEIGHTMAP_RENDER_RENDERREGION_H

#include "heightmap/position.h"
#include "shaderresource.h"
#include "glprojection.h"
#include <QOpenGLFunctions>

class QOpenGLShaderProgram;

namespace Heightmap {
namespace Render {

class RenderRegion: QOpenGLFunctions
{
public:
    RenderRegion();
    RenderRegion(const RenderRegion&)=delete;
    ~RenderRegion();

    RenderRegion& operator=(const RenderRegion&)=delete;

    void render(const glProjecter& gl_projecter, Region r, bool drawcross=true);

private:
    GLuint vbo_=0;
    ShaderPtr program_;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERREGION_H
