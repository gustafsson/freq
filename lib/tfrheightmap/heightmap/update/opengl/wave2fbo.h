#ifndef HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
#define HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H

#include "signal/buffer.h"
#include "zero_on_move.h"
#include "glprojection.h"

#ifndef GL_ES_VERSION_2_0
namespace Heightmap {
namespace Update {
namespace OpenGL {

/**
 * @brief The Texture2Fbo class just draws a waveform. It has nothing
 * to do with any FBO nor any texture.
 */
class Wave2Fbo
{
public:
    Wave2Fbo(Signal::pMonoBuffer b);
    Wave2Fbo(Wave2Fbo&&)=default;
    Wave2Fbo(const Wave2Fbo&)=delete;
    Wave2Fbo& operator=(const Wave2Fbo&)=delete;
    ~Wave2Fbo();

    void draw(const glProjection& glprojection);

private:
    Signal::pMonoBuffer b_;
    JustMisc::zero_on_move<unsigned>    vbo_;
    const int                           N;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
#endif

#endif // HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
