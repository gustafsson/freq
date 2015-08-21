#ifndef HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
#define HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H

#include "signal/buffer.h"
#include "glprojection.h"
#include "vbo.h"
#include <vector>
#include <memory>

class QOpenGLShaderProgram;
namespace Heightmap {
namespace Update {
namespace OpenGL {

/**
 * @brief The Wave2Fbo class just draws a mono buffer as a waveform. It has nothing
 * to do with any FBO.
 */
class Wave2Fbo final
{
public:
    Wave2Fbo();
    Wave2Fbo(const Wave2Fbo&)=delete;
    Wave2Fbo& operator=(const Wave2Fbo&)=delete;

    std::function<bool(const glProjection& M)> prep(Signal::pMonoBuffer b);

private:
    struct vertex_format_xy {
        float x, y;
    };

    std::shared_ptr<QOpenGLShaderProgram> program_;
    std::vector<std::shared_ptr<Vbo>>     vbos_;
    const int                             N_;
    GLuint                                uniModelViewProjectionMatrix_,
                                          uniRgba_;

    typedef std::pair<bool,std::shared_ptr<Vbo>> NewVbo;
    NewVbo getVbo();
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
