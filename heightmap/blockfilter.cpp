#include "blockfilter.h"

#include "block.cu.h"
#include "collection.h"
#include "tfr/cwt.h"
#include "tfr/cwtchunk.h"
#include "tfr/drawnwaveform.h"

#include <CudaException.h>
#include <GlException.h>
#include <TaskTimer.h>

#include <boost/foreach.hpp>

#include <float.h>

//#define TIME_BLOCKFILTER
#define TIME_BLOCKFILTER if(0)

//#define TIME_CWTTOBLOCK
#define TIME_CWTTOBLOCK if(0)

//#define CWTTOBLOCK_INFO
#define CWTTOBLOCK_INFO if(0)

//#define DEBUG_CWTTOBLOCK
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
        applyFilter(Tfr::pChunk pchunk )
{
    Tfr::Chunk& chunk = *pchunk;
    Signal::Interval chunk_interval = chunk.getInterval();
    std::vector<pBlock> intersecting_blocks = _collection->getIntersectingBlocks( chunk_interval, false );
    TIME_BLOCKFILTER TaskTimer tt("BlockFilter %s [%g %g] Hz, intersects with %u visible blocks",
        chunk_interval.toString().c_str(), chunk.minHz(), chunk.maxHz(), intersecting_blocks.size());

    BOOST_FOREACH( pBlock block, intersecting_blocks)
    {
        if (((block->ref.getInterval() - block->valid_samples) & chunk_interval).empty() )
            continue;

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


void BlockFilter::
        mergeColumnMajorChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    TIME_BLOCKFILTER CudaException_ThreadSynchronize();

    Position a, b;
    block->ref.getArea(a,b);

    Position chunk_a, chunk_b;
    Signal::Interval inInterval = chunk.getInterval();
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

    cudaPitchedPtr cpp = chunk.transform_data->getCudaGlobal().getCudaPitchedPtr();

    cpp.xsize = sizeof(float2)*chunk.nScales();
    cpp.ysize = chunk.nSamples();
    cpp.pitch = cpp.xsize;

    ::resampleStft( cpp,
                  outData->getCudaGlobal(),
                  make_float4( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  make_float4( a.time, a.scale,
                               b.time, b.scale ),
                  chunk.freqAxis,
                  _collection->display_scale(),
                  _collection->amplitude_axis()
                  );

    block->valid_samples |= inInterval;

    TIME_BLOCKFILTER CudaException_ThreadSynchronize();
}


void BlockFilter::
        mergeRowMajorChunk( pBlock block, Chunk& chunk, Block::pData outData,
                            bool full_resolution, ComplexInfo complex_info )
{
    CudaException_CHECK_ERROR();

    //unsigned cuda_stream = 0;

    // Find out what intervals that match
    Signal::Interval outInterval = block->ref.getInterval();
    Signal::Interval inInterval = chunk.getInterval();

    Signal::Intervals transferDesc = inInterval;
    transferDesc &= outInterval;

    DEBUG_CWTTOBLOCK TaskTimer tt3("CwtToBlock::mergeChunk");
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
                     _collection->display_scale(),
                     _collection->amplitude_axis()
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


//////////////////////////////// CwtToBlock ///////////////////////////////

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
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(transform().get());
    bool full_resolution = cwt->wavelet_time_support() >= cwt->wavelet_default_time_support();

    Tfr::CwtChunk& chunks = *dynamic_cast<Tfr::CwtChunk*>( &chunk );

    BOOST_FOREACH( const pChunk& chunkpart, chunks.chunks )
    {
        mergeRowMajorChunk( block, *chunkpart, outData,
                            full_resolution, complex_info );
    }
}


/////////////////////////////// StftToBlock ///////////////////////////////

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
    mergeColumnMajorChunk(block, chunk, outData);
}


///////////////////////////// CepstrumToBlock /////////////////////////////

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
    mergeColumnMajorChunk(block, chunk, outData);
}


/////////////////////////// DrawnWaveformToBlock ///////////////////////////

DrawnWaveformToBlock::
        DrawnWaveformToBlock( Collection* collection )
            :
            BlockFilterImpl<Tfr::DrawnWaveformFilter>(collection)
{
    //_try_shortcuts = false;
}


DrawnWaveformToBlock::
        DrawnWaveformToBlock( std::vector<boost::shared_ptr<Collection> > *collections )
            :
            BlockFilterImpl<Tfr::DrawnWaveformFilter>(collections)
{
    //_try_shortcuts = false;
}


ChunkAndInverse DrawnWaveformToBlock::
        computeChunk( const Signal::Interval& I )
{
    std::vector<pBlock> intersecting_blocks = _collection->getIntersectingBlocks( I, false );

    Signal::Intervals missingSamples;
    BOOST_FOREACH( pBlock block, intersecting_blocks)
    {
        missingSamples |= block->ref.getInterval() - block->valid_samples;
    }

    missingSamples &= I;

    float largest_fs = 0;
    Signal::Interval toCompute = I;
    if (missingSamples)
    {
        Signal::Interval first(0, 0);
        first.first = missingSamples.coveredInterval().first;
        first.last = first.first + 1;

        BOOST_FOREACH( pBlock block, intersecting_blocks)
        {
            if (((block->ref.getInterval() - block->valid_samples) & first).empty() )
                continue;

            largest_fs = std::max(largest_fs, block->ref.sample_rate());
        }

        DrawnWaveform* wt = dynamic_cast<DrawnWaveform*>(transform().get());
        wt->block_fs = largest_fs;

        toCompute = missingSamples.fetchFirstInterval();
    }

    return Tfr::DrawnWaveformFilter::computeChunk(toCompute);
}

void DrawnWaveformToBlock::
        mergeChunk( pBlock block, Chunk& chunk, Block::pData outData )
{
    DrawnWaveformChunk* dwc = dynamic_cast<DrawnWaveformChunk*>(&chunk);

    float block_fs = block->ref.sample_rate();

    if (dwc->block_fs != block_fs)
        return;

    mergeRowMajorChunk(block, chunk, outData, true, ComplexInfo_Amplitude_Non_Weighted);
}

} // namespace Heightmap
