#include "block.cu.h"
#include "blockfilter.h"
#include "collection.h"

#include <CudaException.h>

#include <boost/foreach.hpp>
#include <TaskTimer.h>

//#define TIME_CWTTOBLOCK
#define TIME_CWTTOBLOCK if(0)

using namespace Tfr;

namespace Heightmap
{

void BlockFilter::
        mergeChunk( Chunk& chunk )
{
    Signal::Intervals expected = _collection->expected_samples();

    if ( (expected & chunk.getInterval()).isEmpty() ) {
        // TaskTimer("Collection::put received non requested chunk [%u, %u]", chunk.getInterval().first, chunk.getInterval().last);
        return;
    }

    // TODO replace this with Tfr::Transform::displayedTimeResolution etc...
    _collection->update_sample_size( &chunk );

    BOOST_FOREACH( pBlock block, _collection->getIntersectingBlocks( chunk.getInterval() ))
    {
        if (_collection->_constructor_thread.isSameThread())
        {
            mergeChunk( block, chunk, block->glblock->height()->data );
            _collection->computeSlope( block, 0 );
        }
        else
        {
            QMutexLocker l(&block->cpu_copy_mutex);
            if (!block->cpu_copy)
                throw std::logic_error("Multi GPU code is not implemented yet");

            mergeChunk( block, chunk, block->cpu_copy );

            block->cpu_copy->getCpuMemory();
            block->cpu_copy->freeUnused();

            block->new_data_available = true;
        }
    }
}


CwtToBlock::
        CwtToBlock( Collection* collection )
            :
            BlockFilter(collection),
            complex_info(ComplexInfo_Amplitude_Weighted)
{}


void CwtToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    unsigned cuda_stream = 0;

    // Find out what intervals that match
    Signal::Interval outInterval = block->ref.getInterval();
    Signal::Interval inInterval = chunk.getInterval();

    Signal::Intervals transferDesc = inInterval;
    transferDesc &= outInterval;

    // Remove already computed intervals
    transferDesc -= block->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.isEmpty())
        return;

    std::stringstream ss;
    Position a,b;
    block->ref.getArea(a,b);
    TIME_CWTTOBLOCK TaskTimer tt("%s chunk t=[%g, %g[ into block t=[%g,%g] ff=[%g,%g]", __FUNCTION__,
                                 chunk.startTime(), chunk.endTime(), a.time, b.time, a.scale, b.scale);

    float in_sample_rate = chunk.sample_rate;
    float out_sample_rate = block->ref.sample_rate();
    float in_frequency_resolution = chunk.nScales();
    float out_frequency_resolution = block->ref.nFrequencies();

    BOOST_FOREACH( Signal::Interval transfer, transferDesc.intervals())
    {
        TIME_CWTTOBLOCK TaskTimer tt("Inserting chunk [%u,%u]", transfer.first, transfer.last);

        if (0)
        TIME_CWTTOBLOCK {
            TaskTimer("inInterval [%u,%u]", inInterval.first, inInterval.last).suppressTiming();
            TaskTimer("outInterval [%u,%u]", outInterval.first, outInterval.last).suppressTiming();
            TaskTimer("chunk.first_valid_sample = %u", chunk.first_valid_sample).suppressTiming();
            TaskTimer("chunk.n_valid_samples = %u", chunk.n_valid_samples).suppressTiming();
        }

        // Find resolution and offest for the two blocks to be merged
        float in_sample_offset = transfer.first - inInterval.first;
        float out_sample_offset = transfer.first - outInterval.first;

        // Add offset for redundant samples in chunk
        in_sample_offset += chunk.first_valid_sample;

        // Rescale out_offset to out_sample_rate (samples are originally
        // described in the chunk sample rate)
        out_sample_offset *= out_sample_rate / in_sample_rate;

        float in_frequency_offset = a.scale * in_frequency_resolution;
        float out_frequency_offset = 0;

        TaskTimer("a.scale = %g", a.scale).suppressTiming();
        TaskTimer("in_frequency_resolution = %g", in_frequency_resolution).suppressTiming();
        TaskTimer("out_frequency_resolution = %g", out_frequency_resolution).suppressTiming();

        // Invoke CUDA kernel execution to merge blocks
        ::blockMergeChunk( chunk.transform_data->getCudaGlobal(),
                           outData->getCudaGlobal(),

                           in_sample_rate,
                           out_sample_rate,
                           in_frequency_resolution,
                           out_frequency_resolution,
                           in_sample_offset,
                           out_sample_offset,
                           in_frequency_offset,
                           out_frequency_offset,
                           transfer.last - transfer.first,
                           complex_info,
                           cuda_stream);

        block->valid_samples |= transfer;
        TIME_CWTTOBLOCK CudaException_ThreadSynchronize();
    }

    TIME_CWTTOBLOCK CudaException_ThreadSynchronize();
    return;
}


void StftToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    Position a, b;
    block->ref.getArea(a,b);

    float tmin = Tfr::Cwt::Singleton().min_hz();
    float tmax = Tfr::Cwt::Singleton().max_hz( chunk.sample_rate );

    float out_min_hz = exp(log(tmin) + (a.scale*(log(tmax)-log(tmin)))),
          out_max_hz = exp(log(tmin) + (b.scale*(log(tmax)-log(tmin))));

    float out_stft_size = block->ref.sample_rate() / chunk.sample_rate;
    float out_offset = (a.time - (chunk.chunk_offset / chunk.sample_rate))
                       * block->ref.sample_rate();

    ::expandCompleteStft( chunk.transform_data->getCudaGlobal(),
                  outData->getCudaGlobal(),
                  out_min_hz,
                  out_max_hz,
                  out_stft_size,
                  out_offset,
                  chunk.min_hz,
                  chunk.max_hz,
                  chunk.nScales(),
                  0);

    block->valid_samples |= chunk.getInterval();
}

} // namespace Heightmap
