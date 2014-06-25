#ifndef HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H
#define HEIGHTMAP_BLOCKMANAGEMENT_MERGE_MERGERTEXTURE_H

#include "heightmap/blockcache.h"
#include "heightmap/block.h"

class ResampleTexture;

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
    MergerTexture(BlockCache::const_ptr cache, BlockLayout block_layout);
    ~MergerTexture();

    /**
     * @brief fillBlockFromOthers fills a block with data from other blocks.
     * @param block
     */
    void fillBlockFromOthers( pBlock block );

private:
    BlockCache::const_ptr cache_;
    std::shared_ptr<ResampleTexture> rt;
    unsigned tex_, pbo_;

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
