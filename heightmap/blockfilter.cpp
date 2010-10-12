#include "block.cu.h"
#include "blockfilter.h"
#include "collection.h"

#include <CudaException.h>

#include <boost/foreach.hpp>
#include <TaskTimer.h>

#define TIME_CWTTOBLOCK
// #define TIME_CWTTOBLOCK if(0)

using namespace Tfr;

namespace Heightmap
{

BlockFilter::
        BlockFilter( Collection* collection )
            :
            Filter(),
            _collection (collection)
{
    _try_shortcuts = false;
}


void BlockFilter::
        operator()( Tfr::Chunk& chunk )
{
    /*Signal::Intervals expected = _collection->invalid_samples();

    std::cout << "expected=" << expected << std::endl;
    std::cout << "chunk.getInterval()=" << chunk.getInterval() << std::endl;

    if ( !(expected & chunk.getInterval()) ) {
        int dummy = 0;
        // TaskTimer("Collection::put received non requested chunk [%u, %u]", chunk.getInterval().first, chunk.getInterval().last);
        return;
    }*/

    // TODO replace this with Tfr::Transform::displayedTimeResolution etc...
    _collection->update_sample_size( &chunk );

    BOOST_FOREACH( pBlock block, _collection->getIntersectingBlocks( chunk.getInterval() ))
    {
        if (_collection->_constructor_thread.isSameThread())
        {
            mergeChunk( block, chunk, block->glblock->height()->data );
            _collection->computeSlope( block, 0 );

            CudaException_CHECK_ERROR();
        }
        else
        {
            QMutexLocker l(&block->cpu_copy_mutex);
            if (!block->cpu_copy)
                throw std::logic_error(
                    "Multi threaded code is not usefull unless using multiple "
                    "GPUs, and multi GPU code is not implemented yet.");

            mergeChunk( block, chunk, block->cpu_copy );

            block->cpu_copy->getCpuMemory();
            block->cpu_copy->freeUnused();

            block->new_data_available = true;
        }
    }
}


Signal::Intervals BlockFilter::
        affected_samples()
{
    return Signal::Intervals::Intervals();
}


Signal::Operation* BlockFilter::
        affecting_source( const Signal::Interval& )
{
    return this;
}


Signal::Intervals BlockFilter::
        fetch_invalid_samples()
{
    _invalid_samples = _collection->invalid_samples();

    return Tfr::Filter::fetch_invalid_samples();
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
    // transferDesc -= block->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.empty())
    {
        TaskTimer tt("CwtToBlock::mergeChunk, transferDesc empty");
        tt.getStream() << "outInterval = " << outInterval;
        tt.getStream() << "inInterval = " << inInterval;
        tt.suppressTiming();

        int dummy = 0;
        return;
    }

    std::stringstream ss;
    Position a,b;
    block->ref.getArea(a,b);
    float chunk_startTime = (chunk.chunk_offset.asFloat() + chunk.first_valid_sample)/chunk.sample_rate;
    float chunk_length = chunk.n_valid_samples / chunk.sample_rate;
    TIME_CWTTOBLOCK TaskTimer tt("CwtToBlock::mergeChunk chunk t=[%g, %g) into block t=[%g,%g] ff=[%g,%g]",
                                 chunk_startTime, chunk_startTime + chunk_length, a.time, b.time, a.scale, b.scale);

    float in_sample_rate = chunk.sample_rate;
    float out_sample_rate = block->ref.sample_rate();

    float merge_first_scale = a.scale;
    float merge_last_scale = b.scale;
    float chunk_first_scale = _collection->display_scale().getFrequencyScalar( chunk.min_hz );
    float chunk_last_scale = _collection->display_scale().getFrequencyScalar( chunk.max_hz );
    merge_first_scale = std::max( merge_first_scale, chunk_first_scale );
    merge_last_scale = std::min( merge_last_scale, chunk_last_scale );
    if (merge_first_scale >= merge_last_scale)
    {
        TaskTimer("CwtToBlock::mergeChunk, merge_first_scale(%g) >= merge_last_scale(%g)", merge_first_scale, merge_last_scale).suppressTiming();
        return;
    }

    float in_frequency_resolution = chunk.nScales()/(chunk_last_scale - chunk_first_scale);
    unsigned out_frequency_resolution = block->ref.frequency_resolution();
    float in_frequency_offset = (merge_first_scale - chunk_first_scale) * in_frequency_resolution;
    float out_frequency_offset = (merge_first_scale - a.scale) * out_frequency_resolution;
    // Either out_frequency_offset or in_frequency_offset is 0
    if (0<out_frequency_offset)
    {
        // Make out_frequency_offset to an integer (by ceil) and add the
        // remainder to in_frequency_offset
        float c = ceil(out_frequency_offset);
        float d = c - out_frequency_offset;
        out_frequency_offset = c;
        in_frequency_offset += d*in_frequency_resolution/out_frequency_resolution;
    }
    if (out_frequency_offset == _collection->scales_per_block())
        out_frequency_offset--;

    TIME_CWTTOBLOCK TaskTimer("a.scale = %g", a.scale);
    TIME_CWTTOBLOCK TaskTimer("b.scale = %g", b.scale);
    TIME_CWTTOBLOCK TaskTimer("chunk_first_scale = %g", chunk_first_scale);
    TIME_CWTTOBLOCK TaskTimer("chunk_last_scale = %g", chunk_last_scale);
    TIME_CWTTOBLOCK TaskTimer("merge_first_scale = %g", merge_first_scale);
    TIME_CWTTOBLOCK TaskTimer("merge_last_scale = %g", merge_last_scale);
    TIME_CWTTOBLOCK TaskTimer("in_frequency_offset = %g", in_frequency_offset);
    TIME_CWTTOBLOCK TaskTimer("out_frequency_offset = %g", out_frequency_offset);
    TIME_CWTTOBLOCK TaskTimer("in_frequency_resolution = %g", in_frequency_resolution);
    TIME_CWTTOBLOCK TaskTimer("out_frequency_resolution = %u", out_frequency_resolution);
    TIME_CWTTOBLOCK TaskTimer("chunk.nScales() = %u", chunk.nScales());
    TIME_CWTTOBLOCK TaskTimer("_collection->scales_per_block() = %u", _collection->scales_per_block());


    BOOST_FOREACH( Signal::Interval transfer, transferDesc )
    {
        TIME_CWTTOBLOCK TaskTimer tt("Inserting chunk [%u,%u)", transfer.first, transfer.last);

        TIME_CWTTOBLOCK {
            TaskTimer("inInterval [%u,%u)", inInterval.first, inInterval.last).suppressTiming();
            TaskTimer("outInterval [%u,%u)", outInterval.first, outInterval.last).suppressTiming();
            TaskTimer("chunk.first_valid_sample = %u", chunk.first_valid_sample).suppressTiming();
            TaskTimer("chunk.n_valid_samples = %u", chunk.n_valid_samples).suppressTiming();
        }

        // Find resolution and offest for the two blocks to be merged
        float in_sample_offset = transfer.first - inInterval.first;
        float out_sample_offset = transfer.first - outInterval.first;

        // Rescale in_sample_offset to in_sample_rate (samples are originally
        // described in the original source sample rate)
        in_sample_offset *= in_sample_rate / chunk.original_sample_rate;

        // Rescale out_sample_offset to out_sample_rate (samples are originally
        // described in the original source sample rate)
        out_sample_offset *= out_sample_rate / chunk.original_sample_rate;

        // Add offset for redundant samples in chunk
        in_sample_offset += chunk.first_valid_sample;

        CudaException_CHECK_ERROR();

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
                           transfer.count() * (out_sample_rate / chunk.original_sample_rate),
                           complex_info,
                           cuda_stream);

        block->valid_samples |= transfer;

        CudaException_CHECK_ERROR();
    }

    TIME_CWTTOBLOCK CudaException_ThreadSynchronize();
    return;
}


void StftToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    Position a, b;
    block->ref.getArea(a,b);


    // float tmin = chunk.min_hz;
    // float tmax = chunk.max_hz;
    // These doesn't depent on the transform of choice but depend on what
    // parameters are set for the heightmap plot
    float tmin = 20;
    float tmax = 22050;

    float out_min_hz = exp(log(tmin) + (a.scale*(log(tmax)-log(tmin)))),
          out_max_hz = exp(log(tmin) + (b.scale*(log(tmax)-log(tmin))));

    float block_fs = block->ref.sample_rate();
    float out_stft_size = block_fs / chunk.sample_rate;
    float out_offset = (a.time - chunk.chunk_offset / this->sample_rate() )
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
