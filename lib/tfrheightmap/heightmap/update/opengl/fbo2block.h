#ifndef HEIGHTMAP_UPDATE_OPENGL_FBO2BLOCK_H
#define HEIGHTMAP_UPDATE_OPENGL_FBO2BLOCK_H

#include "heightmap/position.h"
#include "glframebuffer.h"
#include "GlTexture.h"
#include "glprojection.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

class Fbo2Block {
public:
    typedef ReleaseAfterContext<Fbo2Block> ScopeBinding;

    Fbo2Block ();
    Fbo2Block (Fbo2Block&&) = default;
    Fbo2Block (const Fbo2Block&) = delete;
    Fbo2Block& operator=(const Fbo2Block&) = delete;
    ~Fbo2Block();

    ScopeBinding begin (Region br, GlTexture::ptr fboTexture, glProjection& M);

private:
    void end();

    GlTexture::ptr blockTexture;
    GlTexture::ptr fboTexture;
    std::unique_ptr<GlFrameBuffer> fbo;
    unsigned copyfbo = 0;
};


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_FBO2BLOCK_H
