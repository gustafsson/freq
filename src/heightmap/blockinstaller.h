#ifndef HEIGHTMAP_BLOCKINSTALLER_H
#define HEIGHTMAP_BLOCKINSTALLER_H

#include "blocklayout.h"
#include "visualizationparams.h"
#include "block.h"
#include "blockcache.h"

namespace Heightmap {

namespace Blocks {
class MergerTexture;
}

/**
 * @brief The BlockInstaller class should create new blocks and install them
 * in a cache.
 */
class BlockInstaller
{
public:
    BlockInstaller(BlockLayout bl, VisualizationParams::ConstPtr vp, BlockCache::Ptr cache);
    BlockInstaller(BlockInstaller const&) = delete;
    BlockInstaller& operator=(BlockInstaller const&) = delete;

    void block_layout(BlockLayout v);

    Signal::Intervals recently_created();
    void set_recently_created_all();

    bool failed_allocation();
    void next_frame();

    /**
      Get a heightmap block. If the referenced block doesn't exist it is created.

      This method is used by Heightmap::Renderer to get the heightmap data of
      blocks that has been decided for rendering.

      Might return 0 if collections decides that it doesn't want to allocate
      another block.
      */
    pBlock      getBlock( const Reference& ref, int frame_counter );

private:
    BlockLayout block_layout_;
    VisualizationParams::ConstPtr visualization_params_;
    BlockCache::Ptr cache_;

    std::shared_ptr<Blocks::MergerTexture> merger_;
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

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKINSTALLER_H
