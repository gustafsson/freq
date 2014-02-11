#ifndef HEIGHTMAP_BLOCK_MERGER_H
#define HEIGHTMAP_BLOCK_MERGER_H

#include "heightmap/blockcache.h"
#include "heightmap/block.h"

namespace Heightmap {
namespace Blocks {

/**
 * @brief The Merger class should merge contents from other blocks to stub the contents of a new block.
 */
class Merger
{
public:
    Merger(BlockCache::ConstPtr cache);

    /**
     * @brief fillBlockFromOthers fills a block with data from other blocks.
     * @param block
     */
    void fillBlockFromOthers( pBlock block );

private:
    BlockCache::ConstPtr cache_;

    /**
      Add block information from another block. Returns whether any information was merged.
      */
    bool mergeBlock( const Block& outBlock,
                     const Block& inBlock,
                     const BlockData::WritePtr& poutData,
                     const BlockData::ReadPtr& pinData );

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_MERGER_H
