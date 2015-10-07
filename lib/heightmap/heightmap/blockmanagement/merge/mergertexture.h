#ifndef HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H
#define HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H

#include "heightmap/blockcache.h"
#include "heightmap/block.h"
#include "GlTexture.h"
#include "heightmap/render/shaderresource.h"

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

/**
 * @brief The MergerTexture class should merge contents from other blocks to
 * stub the contents of a new block.
 *
 * It should use OpenGL textures to do the merge.
 *
 * If quality: 0 MergerTexture will simply rely on mipmaps to have done proper filtering in advance.
 * If quality: 1 MergerTexture will rely on mipmaps "one level down", i.e take the max out of four neightbours.
 * If quality: 2 MergerTexture will examine all individual texels underlying a target texel.
 */
class MergerTexture
{
public:
    MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout, int quality=1);
    ~MergerTexture();

    /**
     * @brief fillBlockFromOthers fills a block with data from other blocks.
     * @param block
     * @return Intervals that couldn't be merged with details from other blocks.
     */
    Signal::Intervals fillBlockFromOthers( pBlock block ) { return fillBlocksFromOthers(std::vector<pBlock>{block}); }
    Signal::Intervals fillBlocksFromOthers( const std::vector<pBlock>& blocks );

private:
    const BlockCache::const_ptr cache_;
    unsigned fbo_;
    unsigned vbo_;
    GlTexture::ptr tex_;
    const BlockLayout block_layout_;
    const int quality_;
    BlockCache::cache_t cache_clone;
    ShaderPtr programp_;
    unsigned program_;

    // glsl uniforms
    int qt_Vertex = -2,
        qt_MultiTexCoord0 = -2,
        qt_Texture0 = -2,
        invtexsize = -2,
        uniProjection = -2,
        uniModelView = -2;

    void init();
    Signal::Intervals fillBlockFromOthersInternal( Block const* block );

    /**
      Add block information from another block
      */
    void mergeBlock( const Region& ri, int texture );

    /**
     * @brief clearBlock is an alternative to glClear
     */
    void clearBlock( const Region& ri );

public:
    static void test();
};

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H
