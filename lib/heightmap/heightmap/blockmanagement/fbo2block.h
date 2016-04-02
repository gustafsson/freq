#ifndef HEIGHTMAP_BLOCKMANAGEMENT_FBO2BLOCK_H
#define HEIGHTMAP_BLOCKMANAGEMENT_FBO2BLOCK_H

#include "heightmap/position.h"
#include "glframebuffer.h"
#include "GlTexture.h"
#include "glprojection.h"

namespace Heightmap {
namespace BlockManagement {

class Fbo2Block: private QOpenGLFunctions {
public:
    typedef ReleaseAfterContext<Fbo2Block> ScopeBinding;

    Fbo2Block ();
    Fbo2Block (Fbo2Block&&) = default;
    Fbo2Block (const Fbo2Block&) = delete;
    Fbo2Block& operator=(const Fbo2Block&) = delete;
    ~Fbo2Block();

    /**
     * @brief begin Creates a Framebuffer Object bound for drawing new contents to a block.
     * @param overlapping See Heightmap::RegionFactory::getOverlapping.
     * @param srcTexture The current block data for reading from.
     * @param drawTexture A texture in which to store new block data. This may
     * be the same texture as srcTexture. If they are different the contents of
     * srcTexture is copied into drawTexture before returning.
     * @param M [out] Orthographic projection matrices for drawing into the block.
     * @return A RAII object whose destructor unbinds the framebuffer object
     * and removes the texture binding for to the FBO.
     *
     * Disables GL_DEPTH_TEST, GL_CULL_FACE during bind and enables
     * them afterwards.
     */
    ScopeBinding begin (Region overlapping_region, GlTexture::ptr srcTexture, GlTexture::ptr drawTexture, glProjection& M);

private:
    void end();

    GlTexture::ptr drawTexture;
#ifndef GL_ES_VERSION_2_0
    unsigned readFbo = 0; // only used if srcTexture!=drawTexture
#endif
    unsigned drawFbo = 0;

#ifdef GL_ES_VERSION_2_0
    void texture2texture(GlTexture::ptr src, GlTexture::ptr dst);
#else
    void fbo2Texture(unsigned fbo, GlTexture::ptr dst);
    void blitTexture(GlTexture::ptr src, unsigned& copyfbo);
#endif
};


} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKMANAGEMENT_FBO2BLOCK_H
