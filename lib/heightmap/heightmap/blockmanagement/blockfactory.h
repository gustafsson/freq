#ifndef HEIGHTMAP_BLOCKFACTORY_H
#define HEIGHTMAP_BLOCKFACTORY_H

#include "heightmap/blockcache.h"
#include "heightmap/render/glblock.h"

namespace Heightmap {
namespace BlockManagement {

/**
 * @brief The BlockFactory class should create new blocks to make them ready
 * for receiving heightmap data and rendering.
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
    pBlock              createBlock( const Reference& ref, Block::pGlBlock reuse=Block::pGlBlock() );

    Signal::Intervals   recently_created();
    void                set_recently_created_all();

    bool                failed_allocation();
    void                next_frame();

private:
    /**
      Creates a new block.
      */
    pBlock              createBlockInternal( const Reference& ref, Block::pGlBlock reuse );


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


    Signal::Intervals recently_created_;
    unsigned created_count_;

    /**
     * @brief failed_allocation_ is cleared by failed_allocation() and populated by getBlock()
     */
    bool failed_allocation_;
    bool failed_allocation_prev_;

public:
    static void test();
};

} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKFACTORY_H
