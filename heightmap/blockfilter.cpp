#include "blockfilter.h"

#include "block.cu.h"
#include "collection.h"
#include "tfr/cwt.h"

#include <CudaException.h>
#include <GlException.h>
#include <TaskTimer.h>

#include <boost/foreach.hpp>

//#define TIME_BLOCKFILTER
#define TIME_BLOCKFILTER if(0)

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
        chunk_interval.toString().c_str(), chunk.minHz(), chunk.maxHz(), intersecting_blocks.size());

    // TODO Use Tfr::Transform::displayedTimeResolution somewhere...

    BOOST_FOREACH( pBlock block, intersecting_blocks)
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

    float merge_first_scale = a.scale;
    float merge_last_scale = b.scale;
    float chunk_first_scale = _collection->display_scale().getFrequencyScalar( chunk.minHz() );
    float chunk_last_scale = _collection->display_scale().getFrequencyScalar( chunk.maxHz() );

    merge_first_scale = std::max( merge_first_scale, chunk_first_scale );
    merge_last_scale = std::min( merge_last_scale, chunk_last_scale );
    if (merge_first_scale >= merge_last_scale)
    {
        DEBUG_CWTTOBLOCK TaskTimer("CwtToBlock::mergeChunk. quiting early\n"
                  "merge_first_scale(%g) >= merge_last_scale(%g)\n"
                  "a.scale = %g, b.scale = %g\n"
                  "chunk_first_scale = %g, chunk_last_scale = %g\n"
                  "chunk.minHz() = %g, chunk.maxHz() = %g",
                  merge_first_scale, merge_last_scale,
                  a.scale, b.scale,
                  chunk_first_scale, chunk_last_scale,
                  chunk.minHz(), chunk.maxHz()).suppressTiming();
        return;
    }

    Position chunk_a, chunk_b;
    chunk_a.scale = chunk_first_scale;
    chunk_b.scale = chunk_last_scale;
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = (inInterval.last-1)/chunk.original_sample_rate;

    DEBUG_CWTTOBLOCK TaskInfo("a.scale = %g", a.scale);
    DEBUG_CWTTOBLOCK TaskInfo("b.scale = %g", b.scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk_first_scale = %g", chunk_first_scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk_last_scale = %g", chunk_last_scale);
    DEBUG_CWTTOBLOCK TaskInfo("merge_first_scale = %g", merge_first_scale);
    DEBUG_CWTTOBLOCK TaskInfo("merge_last_scale = %g", merge_last_scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk.nScales() = %u", chunk.nScales());
    DEBUG_CWTTOBLOCK TaskInfo("_collection->scales_per_block() = %u", _collection->scales_per_block());


    Signal::Interval transfer = transferDesc.coveredInterval();
    
    DEBUG_CWTTOBLOCK {
        TaskTimer("inInterval [%u,%u)", inInterval.first, inInterval.last).suppressTiming();
        TaskTimer("outInterval [%u,%u)", outInterval.first, outInterval.last).suppressTiming();
        TaskTimer("chunk.first_valid_sample = %u", chunk.first_valid_sample).suppressTiming();
        TaskTimer("chunk.n_valid_samples = %u", chunk.n_valid_samples).suppressTiming();
    }

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

    //    cuda-memcheck complains even on this testkernel when using global memory
    //    from OpenGL but not on cudaMalloc'd memory. See MappedVbo test.
#ifdef CUDA_MEMCHECK_TEST
    Block::pData copy( new GpuCpuData<float>( *outData ));
    outData.swap( copy );
#endif

    // Invoke CUDA kernel execution to merge blocks
    ::blockResampleChunk( chunk.transform_data->getCudaGlobal(),
                     outData->getCudaGlobal(),
                     make_uint2( chunk.first_valid_sample, chunk.first_valid_sample+chunk.n_valid_samples ),
                     //make_uint2( 0, chunk.transform_data->getNumberOfElements().width ),
                     make_float4( chunk_a.time, chunk_a.scale,
                                  //chunk_b.time, chunk_b.scale+(chunk_b.scale==1?0.01:0) ), // numerical error workaround, only affects visual
                                 chunk_b.time, chunk_b.scale  ), // numerical error workaround, only affects visual
                     make_float4( a.time, a.scale,
                                  b.time, b.scale ),
                     complex_info,
                     chunk.freqAxis,
                     _collection->display_scale()
                     );

#ifdef CUDA_MEMCHECK_TEST
    outData.swap( copy );
    *outData = *copy;
#endif

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
    // hack to extent first window to first sample, instead of interpolating from 0
    if (inInterval.first < chunk.nScales())
        inInterval.first = 0;
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

#ifdef CUDA_MEMCHECK_TEST
    Block::pData copy( new GpuCpuData<float>( *outData ));
    outData.swap( copy );
#endif

    ::resampleStft( chunk.transform_data->getCudaGlobal(),
                  outData->getCudaGlobal(),
                  make_float4( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  make_float4( a.time, a.scale,
                               b.time, b.scale ),
                  chunk.freqAxis,
                  _collection->display_scale());

#ifdef CUDA_MEMCHECK_TEST
    outData.swap( copy );
    *outData = *copy;
#endif

    block->valid_samples |= inInterval;
}


CepstrumToBlock::
        CepstrumToBlock( Collection* collection )
            :
            BlockFilterImpl<Tfr::CepstrumFilter>(collection)
{
    //_try_shortcuts = false;
}

CepstrumToBlock::
        CepstrumToBlock( std::vector<boost::shared_ptr<Collection> > *collections )
            :
            BlockFilterImpl<Tfr::CepstrumFilter>(collections)
{
    //_try_shortcuts = false;
}


void CepstrumToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    Position a, b;
    block->ref.getArea(a,b);

    Position chunk_a, chunk_b;
    Signal::Interval inInterval = chunk.getInterval();
    // hack to extent first window to first sample, instead of interpolating from 0
    if (inInterval.first < chunk.nScales())
        inInterval.first = 0;
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    // ::resampleCepstrum computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

#ifdef CUDA_MEMCHECK_TEST
    Block::pData copy( new GpuCpuData<float>( *outData ));
    outData.swap( copy );
#endif

    ::resampleStft( chunk.transform_data->getCudaGlobal(),
                  outData->getCudaGlobal(),
                  make_float4( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  make_float4( a.time, a.scale,
                               b.time, b.scale ),
                  chunk.freqAxis,
                  _collection->display_scale());

#ifdef CUDA_MEMCHECK_TEST
    outData.swap( copy );
    *outData = *copy;
#endif

    block->valid_samples |= inInterval;
}


} // namespace Heightmap
