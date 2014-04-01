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
    Merger(BlockCache::const_ptr cache);

    /**
     * @brief fillBlockFromOthers fills a block with data from other blocks.
     * @param block
     */
    void fillBlockFromOthers( pBlock block );

private:
    BlockCache::const_ptr cache_;

    /**
      Add block information from another block. Returns whether any information was merged.
      */
    bool mergeBlock( const Block& outBlock,
                     const Block& inBlock,
                     const shared_state<BlockData>::write_ptr& poutData,
                     const shared_state<BlockData>::read_ptr& pinData );

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_MERGER_H
