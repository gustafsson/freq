#ifndef HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H
#define HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H

#include "heightmap/blockcache.h"
#include "heightmap/block.h"

class GlFrameBuffer;

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
     */
    void fillBlockFromOthers( pBlock block ) { fillBlocksFromOthers(std::vector<pBlock>{block}); }
    void fillBlocksFromOthers( const std::vector<pBlock>& blocks );

private:
    BlockCache::const_ptr cache_;
    std::shared_ptr<GlFrameBuffer> fbo_;
    unsigned vbo_;
    BlockLayout block_layout_;
    unsigned tex_;
    const bool disable_merge_;
    BlockCache::cache_t cache_clone;
    unsigned program_;

    void init();
    void fillBlockFromOthersInternal( pBlock block );

    /**
      Add block information from another block. Returns whether any information was merged.
      */
    bool mergeBlock( const Block& inBlock );

public:
    static void test();
};

} // namespace Merge
} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H
