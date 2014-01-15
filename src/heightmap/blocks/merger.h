#ifndef HEIGHTMAP_BLOCK_MERGER_H
#define HEIGHTMAP_BLOCK_MERGER_H

#include "heightmap/blockcache.h"
#include "heightmap/block.h"

namespace Heightmap {
namespace Blocks {

class Merger
{
public:
    Merger(BlockCache::ConstPtr cache);

    /**
     * @brief createBlockFromOthers fills a block with data from other blocks.
     * @param block
     */
    void fillBlockFromOthers( pBlock block );

private:
    BlockCache::ConstPtr cache_;

    /**
      Add block information from another block. Returns whether any information was merged.
      */
    bool mergeBlock( Block& outBlock, const Block& inBlock, BlockData& outData, const BlockData& inData );
};

} // namespace Block
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCK_MERGER_H
