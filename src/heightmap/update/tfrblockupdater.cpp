#include "tfrblockupdater.h"

#include "tfr/chunk.h"
#include "opengl/blockupdater.h"

#include "neat_math.h"

namespace Heightmap {
namespace Update {

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


TfrBlockUpdater::Job::Job(Tfr::pChunk chunk, float normalization_factor, float largest_fs)
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


Signal::Interval TfrBlockUpdater::Job::
        getCoveredInterval() const
{
    return chunk->getCoveredInterval ();
}


class TfrBlockUpdaterPrivate: public OpenGL::BlockUpdater {};


TfrBlockUpdater::TfrBlockUpdater()
    : p(new TfrBlockUpdaterPrivate)
{
}


TfrBlockUpdater::~TfrBlockUpdater()
{
    delete p;
}


void TfrBlockUpdater::
        processJobs( const std::vector<UpdateQueue::Job>& jobs )
{
    p->processJobs(jobs);
}

} // namespace Update
} // namespace Heightmap
