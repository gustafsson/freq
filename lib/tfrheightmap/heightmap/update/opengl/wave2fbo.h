#ifndef HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
#define HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H

#include "signal/buffer.h"
#include "zero_on_move.h"
#include "glprojection.h"

class QOpenGLShaderProgram;
namespace Heightmap {
namespace Update {
namespace OpenGL {

/**
 * @brief The Wave2Fbo class just draws a mono buffer as a waveform. It has nothing
 * to do with any FBO.
 */
class Wave2Fbo
{
public:
    Wave2Fbo();
    Wave2Fbo(Wave2Fbo&&)=default;
    Wave2Fbo(const Wave2Fbo&)=delete;
    Wave2Fbo& operator=(const Wave2Fbo&)=delete;
    ~Wave2Fbo();

    void draw(const glProjection& glprojection, Signal::pMonoBuffer b);

private:
    struct vertex_format_xy {
        float x, y;
    };

    std::unique_ptr<QOpenGLShaderProgram> m_program;
    JustMisc::zero_on_move<unsigned>    vbo_;
    std::vector<vertex_format_xy>       dv;

    GLuint uniModelViewProjectionMatrix, uniRgba;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
