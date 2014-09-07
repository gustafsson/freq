#ifndef HEIGHTMAP_BLOCKFACTORY_H
#define HEIGHTMAP_BLOCKFACTORY_H

#include "heightmap/blockcache.h"
#include "GlTexture.h"

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
    pBlock              createBlock( const Reference& ref );

    Signal::Intervals   recently_created();

    void                next_frame();

private:
    /**
     * @brief setDummyValues fills a block with dummy values, used for testing.
     * @param block
     */
    void                setDummyValues( pBlock block );


    BlockLayout block_layout_;
    VisualizationParams::const_ptr visualization_params_;
    size_t _free_memory;


    Signal::Intervals recently_created_;
    unsigned created_count_;

public:
    static void test();
};

} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKFACTORY_H
