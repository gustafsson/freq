#ifndef HEIGHTMAP_BLOCKS_CHUNKMERGER_H
#define HEIGHTMAP_BLOCKS_CHUNKMERGER_H

#include "tfr/chunkfilter.h"
#include "updatequeue.h"
#include "heightmap/chunktoblockdegeneratetexture.h"
#include "heightmap/chunktoblock.h"
#include "iupdatejob.h"

namespace Heightmap {
namespace Blocks {


/**
 * @brief The BlockUpdater class should update blocks with chunk data
 */
class BlockUpdater
{
public:
    class Job: public IUpdateJob {
    public:
        Job(Tfr::pChunk chunk, float normalization_factor) : chunk(chunk), normalization_factor(normalization_factor) {}

        const Tfr::pChunk chunk;
        const float normalization_factor;

        Signal::Interval getCoveredInterval() const override;
    };

    BlockUpdater();

    void processJob( const Job& job,
                     std::vector<pBlock> intersecting_blocks );
    void sync();

private:
    ChunkToBlockDegenerateTexture chunktoblock_texture;
//    ChunkToBlock chunktoblock;
public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_CHUNKMERGER_H
