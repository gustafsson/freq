#ifndef HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
#define HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H

#include "signal/buffer.h"

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

    void draw();
private:
    Signal::pMonoBuffer b_;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_WAVE2FBO_H
