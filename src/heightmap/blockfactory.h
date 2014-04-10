#ifndef HEIGHTMAP_BLOCKFACTORY_H
#define HEIGHTMAP_BLOCKFACTORY_H

#include "blockcache.h"
#include "glblock.h"

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
    BlockFactory(BlockLayout, VisualizationParams::const_ptr);

    /**
      Creates a new block.
      */
    pBlock      createBlock( const Reference& ref, pGlBlock reuse );

private:
    /**
      Attempts to allocate a new block.
      */
    pBlock      attempt( const Reference& ref );


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


    BlockLayout block_layout_;
    VisualizationParams::const_ptr visualization_params_;
    size_t _free_memory;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKFACTORY_H
