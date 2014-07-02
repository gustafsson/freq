#ifndef HEIGHTMAP_BLOCKMANAGEMENT_BLOCKINITIALIZER_H
#define HEIGHTMAP_BLOCKMANAGEMENT_BLOCKINITIALIZER_H

#include "heightmap/blocklayout.h"
#include "heightmap/visualizationparams.h"
#include "heightmap/block.h"
#include "heightmap/blockcache.h"

namespace Heightmap {
namespace BlockManagement {

namespace Merge {
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

    void      initBlock( pBlock b ) { initBlocks( std::vector<pBlock>{b}); }
    void      initBlocks( const std::vector<pBlock>& );

private:
    BlockLayout block_layout_;
    VisualizationParams::const_ptr visualization_params_;
    BlockCache::const_ptr cache_;

    std::shared_ptr<Merge::MergerTexture> merger_;
public:
    static void test();
};

} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKMANAGEMENT_BLOCKINITIALIZER_H
