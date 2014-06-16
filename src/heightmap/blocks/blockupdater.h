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
 *
 *
 * Data transfers summary
 * ----------------------
 * complex chunk -> real chunk -> chunk pbo -> chunk texture -> block fbo -> block texture
 *
 *
 * complex chunk (Tfr::pChunk) -> real chunk (Blocks::IUpdateJob)
 * --------------------------------------------------------------
 * Find the largest sample rate among matching blocks: CwtBlockFilter::prepareUpdate
 * Compute squared absolute value and downsample to a smaller dataset that matches the resolution needed: BlockUpdater::Job::Job
 *
 *
 * real chunk (Blocks::IUpdateJob) -> chunk pbo (DrawableChunk)
 * ------------------------------------------------------------
 * thread: UpdateConsumer gets real chunk, pop IUpdateJob from UpdateQueue.
 * create chunk pbo: ChunkToBlockDegenerateTexture::DrawableChunk::setupPbo,
 *                   called from ChunkToBlockDegenerateTexture::prepareChunk
 * map chunk pbo: mapped_chunk_data_ in ChunkToBlockDegenerateTexture::DrawableChunk::transferData
 * copy chunk to mapped pbo: std::packaged_task<void()>(memcpy) from transferData
 *                           thread: memcpythread.addTask (d.transferData(job.p))
 * unmap chunk pbo: if(mapped_chunk_data_) in ChunkToBlockDegenerateTexture::DrawableChunk::prepareShader
 * Multiple available IUpdateJob are processed with one memcpythread.
 * done. PBO now contains chunk data.
 *
 *
 * chunk pbo -> chunk texture (DrawableChunk -> (DrawableChunk)
 * ------------------------------------------------------------
 * create and fill texture: DrawableChunk::shader_.chunk_texture_ in ChunkToBlockDegenerateTexture::ShaderTexture::prepareShader
 *                          called from ChunkToBlockDegenerateTexture::DrawableChunk::prepareShader ()
 *                          called from ChunkToBlockDegenerateTexture::DrawableChunk::draw ()
 * Multiple available chunk pbos are drawn asyncronously.
 * done. chunk texture contains chunk data
 *
 *
 * chunk texture (DrawableChunk) -> block fbo (BlockFbo)
 * -----------------------------------------------------
 * create block fbo: fbo = new GlFrameBuffer in BlockFbo::BlockFbo
 * use vbo for drawing: ChunkToBlockDegenerateTexture::DrawableChunk::setupVbo
 * setup fbo drawing: chunktoblock_texture.prepareBlock
 * begin fbo drawing: BlockFbo::begin ()
 * draw texture: ChunkToBlockDegenerateTexture::DrawableChunk::draw ()
 * done. block fbo and underlying texture contains mapped chunk data.
 *
 *
 * block fbo (BlockFbo) -> block texture (Block::glblock)
 * ------------------------------------------------------
 * block texture was created before.
 * block fbo was created so that it draws straight onto the texture.
 * function grabToTexture updates the vertex texture with glCopyTexSubImage2D.
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
