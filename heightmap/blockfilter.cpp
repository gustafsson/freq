#include "blockfilter.h"

#include "block.cu.h"
#include "collection.h"
#include "tfr/cwt.h"

#include <CudaException.h>
#include <GlException.h>
#include <TaskTimer.h>

#define TIME_BLOCKFILTER
//#define TIME_BLOCKFILTER if(0)

//#define TIME_CWTTOBLOCK
#define TIME_CWTTOBLOCK if(0)

//#define CWTTOBLOCK_INFO
#define CWTTOBLOCK_INFO if(0)

// #define DEBUG_CWTTOBLOCK
#define DEBUG_CWTTOBLOCK if(0)

using namespace Tfr;

namespace Heightmap
{

BlockFilter::
        BlockFilter( Collection* collection )
            :
            _collection (collection)
{
}


void BlockFilter::
        operator()( Tfr::Chunk& chunk )
{
    Signal::Interval chunk_interval = chunk.getInterval();
    std::vector<pBlock> intersecting_blocks = _collection->getIntersectingBlocks( chunk_interval, true );
    TIME_BLOCKFILTER TaskTimer tt("BlockFilter %s [%g %g] Hz, intersects with %u visible blocks", 
        chunk_interval.toString().c_str(), chunk.min_hz, chunk.max_hz, intersecting_blocks.size());

    // TODO Use Tfr::Transform::displayedTimeResolution somewhere...

    foreach( pBlock block, intersecting_blocks)
    {
#ifndef SAWE_NO_MUTEX
        if (_collection->constructor_thread().isSameThread())
        {
#endif
            mergeChunk( block, chunk, block->glblock->height()->data );

            TIME_BLOCKFILTER CudaException_CHECK_ERROR();
#ifndef SAWE_NO_MUTEX
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
#endif
    }

    TIME_BLOCKFILTER CudaException_ThreadSynchronize();
}


CwtToBlock::
        CwtToBlock( Collection* collection )
            :
            BlockFilterImpl<Tfr::CwtFilter>(collection),
            complex_info(ComplexInfo_Amplitude_Non_Weighted)
{
}

CwtToBlock::
        CwtToBlock( std::vector<boost::shared_ptr<Collection> >* collections )
            :
            BlockFilterImpl<Tfr::CwtFilter>(collections),
            complex_info(ComplexInfo_Amplitude_Non_Weighted)
{
}


void CwtToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    //unsigned cuda_stream = 0;

    // Find out what intervals that match
    Signal::Interval outInterval = block->ref.getInterval();
    Signal::Interval inInterval = chunk.getInterval();

    Signal::Intervals transferDesc = inInterval;
    transferDesc &= outInterval;

    DEBUG_CWTTOBLOCK TaskTimer tt("CwtToBlock::mergeChunk");
    DEBUG_CWTTOBLOCK TaskTimer("outInterval=[%g, %g)",
            outInterval.first / chunk.original_sample_rate,
            outInterval.last / chunk.original_sample_rate ).suppressTiming();
    DEBUG_CWTTOBLOCK TaskTimer("inInterval=[%g, %g)",
            inInterval.first / chunk.original_sample_rate,
            inInterval.last / chunk.original_sample_rate ).suppressTiming();
    DEBUG_CWTTOBLOCK TaskTimer("transferDesc=[%g, %g)",
            transferDesc.coveredInterval().first / chunk.original_sample_rate,
            transferDesc.coveredInterval().last / chunk.original_sample_rate ).suppressTiming();

    // Remove already computed intervals
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(transform().get());
    bool full_resolution = cwt->wavelet_time_support() >= cwt->wavelet_default_time_support();
    if (!full_resolution)
    {
        if (!(transferDesc - block->valid_samples))
        {
            TIME_CWTTOBLOCK TaskInfo("%s not accepting %s, early termination", vartype(*this).c_str(), transferDesc.toString().c_str());
            transferDesc.clear();
        }
    }
    // transferDesc -= block->valid_samples;

    // If block is already up to date, abort merge
    if (transferDesc.empty())
    {
        TIME_CWTTOBLOCK TaskInfo tt("CwtToBlock::mergeChunk, transferDesc empty");
        TIME_CWTTOBLOCK TaskInfo("outInterval = %s", outInterval.toString().c_str());
        TIME_CWTTOBLOCK TaskInfo("inInterval = %s", inInterval.toString().c_str());

        return;
    }

    Position a,b;
    block->ref.getArea(a,b);
    float chunk_startTime = (chunk.chunk_offset.asFloat() + chunk.first_valid_sample)/chunk.sample_rate;
    float chunk_length = chunk.n_valid_samples / chunk.sample_rate;
    DEBUG_CWTTOBLOCK TaskTimer tt2("CwtToBlock::mergeChunk chunk t=[%g, %g) into block t=[%g,%g] ff=[%g,%g]",
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
        DEBUG_CWTTOBLOCK TaskTimer("CwtToBlock::mergeChunk. quiting early\n"
                  "merge_first_scale(%g) >= merge_last_scale(%g)\n"
                  "a.scale = %g, b.scale = %g\n"
                  "chunk_first_scale = %g, chunk_last_scale = %g\n"
                  "chunk.min_hz = %g, chunk.max_hz = %g",
                  merge_first_scale, merge_last_scale,
                  a.scale, b.scale,
                  chunk_first_scale, chunk_last_scale,
                  chunk.min_hz, chunk.max_hz).suppressTiming();
        return;
    }

    Position chunk_a, chunk_b;
    chunk_a.scale = chunk_first_scale;
    chunk_b.scale = chunk_last_scale;
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = (inInterval.last-1)/chunk.original_sample_rate;

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

    DEBUG_CWTTOBLOCK TaskTimer("a.scale = %g", a.scale);
    DEBUG_CWTTOBLOCK TaskTimer("b.scale = %g", b.scale);
    DEBUG_CWTTOBLOCK TaskTimer("chunk_first_scale = %g", chunk_first_scale);
    DEBUG_CWTTOBLOCK TaskTimer("chunk_last_scale = %g", chunk_last_scale);
    DEBUG_CWTTOBLOCK TaskTimer("merge_first_scale = %g", merge_first_scale);
    DEBUG_CWTTOBLOCK TaskTimer("merge_last_scale = %g", merge_last_scale);
    DEBUG_CWTTOBLOCK TaskTimer("in_frequency_offset = %g", in_frequency_offset);
    DEBUG_CWTTOBLOCK TaskTimer("out_frequency_offset = %g", out_frequency_offset);
    DEBUG_CWTTOBLOCK TaskTimer("in_frequency_resolution = %g", in_frequency_resolution);
    DEBUG_CWTTOBLOCK TaskTimer("out_frequency_resolution = %u", out_frequency_resolution);
    DEBUG_CWTTOBLOCK TaskTimer("chunk.nScales() = %u", chunk.nScales());
    DEBUG_CWTTOBLOCK TaskTimer("_collection->scales_per_block() = %u", _collection->scales_per_block());


    Signal::Interval transfer = transferDesc.coveredInterval();
    
    DEBUG_CWTTOBLOCK {
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

    //CWTTOBLOCK_INFO TaskTimer("CwtToBlock [(%g %g), (%g %g)] <- [(%g %g), (%g %g)] |%g %g|",
    TIME_CWTTOBLOCK TaskTimer tt("CwtToBlock [(%.2f %.2f), (%.2f %.2f)] <- [(%.2f %g), (%.2f %g)] |%.2f %.2f|",
            a.time, a.scale,
            b.time, b.scale,
            chunk_a.time, chunk_a.scale,
            chunk_b.time, chunk_b.scale,
            transfer.first/chunk.original_sample_rate, transfer.last/chunk.original_sample_rate
        );

    BOOST_ASSERT( chunk.first_valid_sample+chunk.n_valid_samples <= chunk.transform_data->getNumberOfElements().width );

    if(0) {
        float2* p = chunk.transform_data->getCpuMemory();
        cudaExtent sz = chunk.transform_data->getNumberOfElements();
        for (unsigned y=0; y<sz.height; ++y)
        {
            for (unsigned x=0; x<=chunk.first_valid_sample+1; ++x)
            {
                p[y*sz.width + x ] = make_float2(100,100);
            }

            for (unsigned x=chunk.first_valid_sample + chunk.n_valid_samples + 1;
                 x<sz.width; ++x)
            {
                p[y*sz.width + x ] = make_float2(100,100);
            }
        }
    }

    // Invoke CUDA kernel execution to merge blocks
    ::blockResampleChunk( chunk.transform_data->getCudaGlobal(),
                     outData->getCudaGlobal(),
                     make_uint2( chunk.first_valid_sample, chunk.first_valid_sample+chunk.n_valid_samples ),
                     //make_uint2( 0, chunk.transform_data->getNumberOfElements().width ),
                     make_float4( chunk_a.time, chunk_a.scale,
                                  chunk_b.time, chunk_b.scale+(chunk_b.scale==1?0.01:0) ), // bug workaround, only affects visual
                     make_float4( a.time, a.scale,
                                  b.time, b.scale ),
                     complex_info,
                     _collection->display_scale()
                     );


    // TODO recompute transfer to the samples that have actual support
    CudaException_CHECK_ERROR();
    GlException_CHECK_ERROR();

    if( full_resolution )
    {
        block->valid_samples |= transfer;
    }
    else
    {
        block->valid_samples -= transfer;
        TIME_CWTTOBLOCK TaskInfo("%s not accepting %s", vartype(*this).c_str(), transfer.toString().c_str());
    }

    TIME_CWTTOBLOCK CudaException_ThreadSynchronize();
}


StftToBlock::
        StftToBlock( Collection* collection )
            :
            BlockFilterImpl<Tfr::StftFilter>(collection)
{
}

StftToBlock::
        StftToBlock( std::vector<boost::shared_ptr<Collection> >* collections )
            :
            BlockFilterImpl<Tfr::StftFilter>(collections)
{
}


void StftToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    Position a, b;
    block->ref.getArea(a,b);

    Position chunk_a, chunk_b;
    Signal::Interval inInterval = chunk.getInterval();
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = (inInterval.last-chunk.nScales())/chunk.original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

    ::resampleStft( chunk.transform_data->getCudaGlobal(),
                  outData->getCudaGlobal(),
                  make_float4( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  make_float4( a.time, a.scale,
                               b.time, b.scale ),
                  chunk.freqAxis(),
                  _collection->display_scale());

    block->valid_samples |= chunk.getInterval();
}


} // namespace Heightmap
