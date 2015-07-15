#ifndef HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H
#define HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H

#include "heightmap/blockcache.h"
#include "heightmap/block.h"
#include "GlTexture.h"

namespace Heightmap {
namespace BlockManagement {
namespace Merge {

/**
 * @brief The MergerTexture class should merge contents from other blocks to
 * stub the contents of a new block.
 *
 * It should use OpenGL textures to do the merge.
 */
class MergerTexture
{
public:
    MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout, bool disable_merge=false);
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
    const bool disable_merge_;
    BlockCache::cache_t cache_clone;
    unsigned program_;

    // glsl uniforms
    int qt_Vertex = 0,
        qt_MultiTexCoord0 = 0,
        qt_Texture0 = 0,
        invtexsize = 0,
        uniProjection = 0,
        uniModelView = 0;

    void init();
    Signal::Intervals fillBlockFromOthersInternal( pBlock block );

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
