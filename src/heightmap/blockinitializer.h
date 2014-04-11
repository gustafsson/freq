#ifndef HEIGHTMAP_BLOCKINSTALLER_H
#define HEIGHTMAP_BLOCKINSTALLER_H

#include "blocklayout.h"
#include "visualizationparams.h"
#include "block.h"
#include "blockcache.h"
#include "glblock.h"

namespace Heightmap {

namespace Blocks {
class MergerTexture;
}

/**
 * @brief The BlockInstaller class should initialize new blocks with content from others.
 */
class BlockInitializer
{
public:
    BlockInitializer(BlockLayout bl, VisualizationParams::const_ptr vp, BlockCache::const_ptr cache);
    BlockInitializer(BlockInitializer const&) = delete;
    BlockInitializer& operator=(BlockInitializer const&) = delete;

    void      initBlock( pBlock );

private:
    BlockLayout block_layout_;
    VisualizationParams::const_ptr visualization_params_;
    BlockCache::const_ptr cache_;

    std::shared_ptr<Blocks::MergerTexture> merger_;
public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKINSTALLER_H
