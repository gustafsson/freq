#ifndef HEIGHTMAP_BLOCKS_CHUNKMERGER_H
#define HEIGHTMAP_BLOCKS_CHUNKMERGER_H

#include "tfr/chunkfilter.h"
#include "updatequeue.h"
#include "heightmap/chunktoblockdegeneratetexture.h"
#include "heightmap/chunktoblock.h"
#include "iupdatejob.h"

#include "thread_pool.h"

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
        Job(Tfr::pChunk chunk, float normalization_factor, float largest_fs=0);

        Tfr::pChunk chunk;
        float *p;
        float normalization_factor;

        Signal::Interval getCoveredInterval() const override;
    };

    BlockUpdater();
    BlockUpdater(const BlockUpdater&) = delete;
    BlockUpdater& operator=(const BlockUpdater&) = delete;
    ~BlockUpdater();

    void processJobs( const std::vector<UpdateQueue::Job>& jobs );
    ChunkToBlockDegenerateTexture::DrawableChunk processJob(
            const Job& job,
            const std::vector<pBlock>& intersecting_blocks );

private:
    void sync();

    JustMisc::thread_pool memcpythread;
    ChunkToBlockDegenerateTexture chunktoblock_texture;
//    ChunkToBlock chunktoblock;
public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_CHUNKMERGER_H
