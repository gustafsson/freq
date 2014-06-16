#include "blockupdater.h"
#include "tfr/chunk.h"

#include "tasktimer.h"
#include "timer.h"
#include "log.h"
#include "neat_math.h"

#include <thread>
#include <future>

//#define INFO
#define INFO if(0)

using namespace std;

namespace Heightmap {
namespace Blocks {

class JobChunk : public Tfr::Chunk
{
public:
    JobChunk(int w, int h)
        :
          Chunk(Order_row_major),
          w(w),
          h(h)
    {
    }

    unsigned nSamples() const override { return order==Order_row_major ? w : h; }
    unsigned nScales()  const override { return order==Order_row_major ? h : w; }

private:
    int w, h;
};


float* computeNorm(Tfr::ChunkElement* cp, int n) {
    float *p = (float*)cp; // Overwrite 'cp'
    // This takes a while, simply because p is large so that a lot of memory has to be copied.
    for (int i = 0; i<n; ++i)
        p[i] = norm(cp[i]); // Compute norm here and square root in shader.
    return p;
}


BlockUpdater::Job::Job(Tfr::pChunk chunk, float normalization_factor, float largest_fs)
    :
      chunk(chunk),
      p(0),
      normalization_factor(normalization_factor)
{
    Tfr::ChunkElement *cp = chunk->transform_data->getCpuMemory ();
    int n = chunk->transform_data->numberOfElements ();
    // Compute the norm of the complex elements in the chunk prior to resampling and interpolating
    this->p = computeNorm(cp, n);

    if (!"Disable data preparation")
        return;

    int stepx = 0;
    if (0 < largest_fs)
        stepx = chunk->sample_rate / largest_fs / 4;
    if (stepx < 1)
        stepx = 1;

    int data_height, data_width, org_height, org_width;
    int offs_y = 0, offs_x = 0;
    switch (chunk->order)
    {
    case Tfr::Chunk::Order_row_major:
        org_width = chunk->nSamples ();
        org_height = chunk->nScales ();
        data_width = int_div_ceil (chunk->n_valid_samples, stepx);
        data_height = org_height;
        offs_x = chunk->first_valid_sample;
        break;

    case Tfr::Chunk::Order_column_major:
        org_width = chunk->nScales ();
        org_height = chunk->nSamples ();
        data_width = chunk->nScales ();
        data_height = int_div_ceil (chunk->nSamples (), stepx);
        offs_y = chunk->first_valid_sample;
        break;

    default:
        EXCEPTION_ASSERT_EQUALS(chunk->order, Tfr::Chunk::Order_row_major);
    }

    if (data_width == org_width && data_height == org_height)
        return;

    for (int y=0; y<data_height; ++y)
        for (int x=0; x<data_width; ++x)
          {
            float v = 0;
            int i = (y + offs_y)*org_width + offs_x + x*stepx;
            for (int j = 0; j<stepx; ++j)
                v = std::max(v, p[i + j]);
            p[y*data_width + x] = v;
          }

    this->chunk.reset (new JobChunk(data_width, data_height));
    this->chunk->order = chunk->order;
    this->chunk->freqAxis = chunk->freqAxis;
    this->chunk->transform_data = chunk->transform_data;
    this->chunk->chunk_offset = (chunk->chunk_offset + chunk->first_valid_sample)/(double)stepx;
    this->chunk->first_valid_sample = 0;
    this->chunk->n_valid_samples = data_width;
    this->chunk->sample_rate = chunk->sample_rate / stepx;
    this->chunk->original_sample_rate = chunk->original_sample_rate;
}


Signal::Interval BlockUpdater::Job::
        getCoveredInterval() const
{
    return chunk->getCoveredInterval ();
}


BlockUpdater::
        BlockUpdater()
    :
//      memcpythread(std::thread::hardware_concurrency ())
      memcpythread(2)
{
}


BlockUpdater::
        ~BlockUpdater()
{
    sync ();
}


void BlockUpdater::
        processJobs( const std::vector<UpdateQueue::Job>& jobs )
{
    typedef shared_ptr<ChunkToBlockDegenerateTexture::DrawableChunk> pDrawableChunk;
    ChunkToBlockDegenerateTexture::BlockFbos& block_fbos = chunktoblock_texture.block_fbos ();

    // 'jobs' contains all pBlock and survives longer than pDrawableChunk
    std::map<pBlock, std::queue<pDrawableChunk>> chunks_per_block;

    {
//                TaskTimer tt("Preparing %d jobs, span %s", jobs.size (), span.toString ().c_str ());

      for (const UpdateQueue::Job& job : jobs)
        {
          if (!job.updatejob)
              continue;

          if (auto bujob = dynamic_cast<const BlockUpdater::Job*>(job.updatejob.get ()))
            {
              auto drawable = processJob (*bujob, job.intersecting_blocks);
              pDrawableChunk d(new ChunkToBlockDegenerateTexture::DrawableChunk(move(drawable)));

//                        chunks_with_blocks.push_back (chunk_with_blocks_t(d, job.intersecting_blocks));
              for (pBlock b : job.intersecting_blocks)
                  chunks_per_block[b].push(d);
            }
        }
    }

  if (!chunks_per_block.empty ())
    {
//                TaskTimer tt("Updating %d blocks", chunks_per_block.size ());
      // draw all chunks who share the same block in one go
      for (map<pBlock, queue<pDrawableChunk>>::value_type& p : chunks_per_block)
        {
          shared_ptr<BlockFbo> fbo = block_fbos[p.first];
          if (!fbo)
              continue;

//                    TaskTimer tt(boost::format("Drawing %d chunks into block %s")
//                                 % p.second.size() % p.first->getRegion());

          auto fboScopeBinding = fbo->begin ();

          auto& q = p.second;
          while (!q.empty ())
            {
              pDrawableChunk c {move(q.front ())};
              q.pop ();
              c->draw ();
            }

          // unbind fbo on out-of-scope
        }
    }

//            if (!chunks_with_blocks.empty ())
//              {
//                TaskTimer tt("Updating from %d chunks", chunks_with_blocks.size ());
//                for (auto& c : chunks_with_blocks)
//                  {
//                    for (auto& b : c.second)
//                      {
//                        shared_ptr<BlockFbo> fbo = block_fbos[b];
//                        if (!fbo)
//                            continue;
//                        fbo->begin ();
//                        c.first->draw();
//                        fbo->end ();
//                      }
//                  }
//              }

//            chunks_with_blocks.clear ();

  sync (); // Wait for block textures to become updated by BlockFbo destructor.
}


ChunkToBlockDegenerateTexture::DrawableChunk BlockUpdater::
        processJob( const BlockUpdater::Job& job,
                    const std::vector<pBlock>& intersecting_blocks )
{
    EXCEPTION_ASSERT (!intersecting_blocks.empty ());
    EXCEPTION_ASSERT (job.chunk);

    pBlock example_block = intersecting_blocks.front ();
    BlockLayout block_layout = example_block->block_layout ();
    Tfr::FreqAxis display_scale = example_block->visualization_params ()->display_scale ();
    AmplitudeAxis amplitude_axis = example_block->visualization_params ()->amplitude_axis ();

    chunktoblock_texture.setParams( amplitude_axis, display_scale, block_layout, job.normalization_factor );

//    TaskTimer tt0(boost::format("processJob %s") % job.getCoveredInterval ());
    // TODO refactor to do one thing at a time
    // 1. prepare to draw from chunks (i.e copy to OpenGL texture and create vertex buffers)
    // 1.1. Same chunk_scale and display_scale for all chunks and all blocks
    // 2. For each block.
    // 2.1. prepare to draw into block
    // 2.2. draw all chunks
    // 2.3. update whatever needs to be updated
    // And rename all these "chunk block merger to merge block chunk merge"

    // Setup FBO
    for (pBlock block : intersecting_blocks)
        chunktoblock_texture.prepareBlock (block);

    // Update PBO and VBO
    auto d = chunktoblock_texture.prepareChunk (job.chunk);
    memcpythread.addTask (d.transferData(job.p));
    return d;
}


void BlockUpdater::
        sync ()
{
    chunktoblock_texture.finishBlocks ();
}

} // namespace Blocks
} // namespace Heightmap


namespace Heightmap {
namespace Blocks {

void BlockUpdater::
        test()
{
    // It should update blocks with chunk data
    {

    }
}

} // namespace Blocks
} // namespace Heightmap
