#ifndef HEIGHTMAP_BLOCKFACTORY_H
#define HEIGHTMAP_BLOCKFACTORY_H

#include "heightmap/blockcache.h"
#include "blockupdater.h"
#include "GlTexture.h"

namespace Heightmap {
namespace BlockManagement {

/**
 * @brief The BlockFactory class should create new blocks to make them ready
 * for receiving heightmap data and rendering.
 */
class BlockFactory
{
public:
    BlockFactory();

    BlockFactory& reset(BlockLayout, VisualizationParams::const_ptr);

    /**
      Creates a new block.
      */
    pBlock              createBlock( const Reference& ref );

    BlockUpdater*       updater() { return updater_.get (); }

private:
    /**
     * @brief setDummyValues fills a block with dummy values, used for testing.
     * @param block
     */
    void                setDummyValues( pBlock block );

    BlockLayout block_layout_;
    VisualizationParams::const_ptr visualization_params_;
    BlockUpdater::ptr updater_;

public:
    static void test();
};

} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKFACTORY_H
