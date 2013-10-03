#ifndef HEIGHTMAP_BLOCKFACTORY_H
#define HEIGHTMAP_BLOCKFACTORY_H

#include "blockcache.h"

namespace Heightmap {

/**
 * @brief The BlockFactory class should create new blocks to make them ready
 * for receiving transform data and rendering.
 *
 * TODO should take BlockCache::ConstPtr
 */
class BlockFactory
{
public:
    BlockFactory(BlockCache::Ptr cache, BlockLayout, VisualizationParams::ConstPtr, unsigned frame_counter);

    /**
      Creates a new block.
      */
    pBlock      createBlock( const Reference& ref );

private:
    /**
      Attempts to allocate a new block.
      */
    pBlock      attempt( const Reference& ref );


    /**
      Add block information from another block. Returns whether any information was merged.
      */
    bool        mergeBlock( Block& outBlock, const Block& inBlock, BlockData& outData, const BlockData& inData );


    /**
     * @brief getAllocatedBlock returns an allocated block either by new a
     * memory allocation or by reusing the data from an old block.
     */
    pBlock      getAllocatedBlock( const Reference& ref );


    /**
     * @brief setDummyValues fills a block with dummy values, used for testing.
     * @param block
     */
    void        setDummyValues( pBlock block );


    /**
     * @brief createBlockFromOthers fills a block with data from other blocks.
     * @param block
     */
    void        createBlockFromOthers(pBlock block);


    /**
     * @brief gc doesn't belong here...
     */
    void        gc();


    BlockCache::Ptr cache_;
    BlockLayout block_layout_;
    VisualizationParams::ConstPtr visualization_params_;
    unsigned _frame_counter;
    size_t _free_memory;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKFACTORY_H
