#ifndef HEIGHTMAP_UPDATE_OPENGL_FBO2BLOCK_H
#define HEIGHTMAP_UPDATE_OPENGL_FBO2BLOCK_H

#include "heightmap/render/glblock.h"
#include "heightmap/position.h"
#include "glframebuffer.h"
#include "GlTexture.h"

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

    ScopeBinding begin (Region br, Render::GlBlock::ptr glblock);

private:
    void end();

    Render::GlBlock::ptr glblock;
    GlTexture::ptr texture;
    std::unique_ptr<GlFrameBuffer> fbo;
    unsigned copyfbo;
};


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_FBO2BLOCK_H
