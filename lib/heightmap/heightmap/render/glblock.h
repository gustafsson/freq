#ifndef HEIGHTMAPVBO_H
#define HEIGHTMAPVBO_H

#include "heightmap/blocklayout.h"

// gpumisc
#include "mappedvbo.h"
#include "ThreadChecker.h"
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
    GlBlock( BlockLayout block_size, float width, float height );
    ~GlBlock();

    void                reset( float width, float height );

    GlTexture::ptr      glTexture();
    void                updateTexture( float*p, int n );

    bool has_texture() const;
    void delete_texture();

    void draw( unsigned vbo_size );

    unsigned allocated_bytes_per_element() const;

private:
    void create_texture();

    const BlockLayout block_size_;

    unsigned _tex_height;

    float _world_width, _world_height;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAPVBO_H

