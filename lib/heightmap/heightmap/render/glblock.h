#ifndef HEIGHTMAPVBO_H
#define HEIGHTMAPVBO_H

// gpumisc
#include "GlTexture.h"

//#define BLOCK_INDEX_TYPE GL_UNSIGNED_SHORT
//#define BLOCKindexType GLushort
#define BLOCK_INDEX_TYPE GL_UNSIGNED_INT
#define BLOCKindexType GLuint

namespace Heightmap {
namespace Render {

class GlBlock
{
public:
    typedef std::shared_ptr<GlBlock> ptr;

    GlBlock( GlTexture::ptr tex );

    void            draw( unsigned vbo_size );

    GlTexture::ptr  glTexture();
    void            updateTexture( float*p, int n );
    bool            has_texture() const;

private:
    GlTexture::ptr tex_;
    unsigned _tex_height;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAPVBO_H

