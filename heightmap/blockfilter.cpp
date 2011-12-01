#include "blockfilter.h"

#include "blockkernel.h"
#include "collection.h"
#include "tfr/cwt.h"
#include "tfr/cwtchunk.h"
#include "tfr/drawnwaveform.h"

#include <computationkernel.h>
#include <GlException.h>
#include <TaskTimer.h>
#include <Statistics.h>

#include <boost/foreach.hpp>

#include <float.h>

#define TIME_BLOCKFILTER
//#define TIME_BLOCKFILTER if(0)

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
        applyFilter(ChunkAndInverse& pchunk )
{
    Tfr::Chunk& chunk = *pchunk.chunk;
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

            TIME_BLOCKFILTER ComputationCheckError();
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

    TIME_BLOCKFILTER ComputationSynchronize();
}


void BlockFilter::
        mergeColumnMajorChunk( pBlock block, Chunk& chunk, Block::pData outData, float normalization_factor )
{
    TIME_BLOCKFILTER ComputationSynchronize();

    Position a, b;
    block->ref.getArea(a,b);

    Position chunk_a, chunk_b;
    Signal::Interval inInterval = chunk.getInterval();
    Signal::Interval blockInterval = block->ref.getInterval();
    Signal::Interval transfer = inInterval&blockInterval;

    if (!transfer)
        return;

    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

    ::resampleStft( chunk.transform_data,
                    chunk.nScales(),
                    chunk.nSamples(),
                  outData,
                  ResampleArea( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  ResampleArea( a.time, a.scale,
                               b.time, b.scale ),
                  chunk.freqAxis,
                  _collection->display_scale(),
                  _collection->amplitude_axis(),
                  normalization_factor
                  );

    block->valid_samples |= transfer;
    block->non_zero |= transfer;

    TIME_BLOCKFILTER ComputationSynchronize();
}


void BlockFilter::
        mergeRowMajorChunk( pBlock block, Chunk& chunk, Block::pData outData,
                            bool full_resolution, ComplexInfo complex_info,
                            float normalization_factor)
{
    ComputationCheckError();

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
            transferDesc.spannedInterval().first / chunk.original_sample_rate,
            transferDesc.spannedInterval().last / chunk.original_sample_rate ).suppressTiming();

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
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    DEBUG_CWTTOBLOCK TaskInfo("a.scale = %g", a.scale);
    DEBUG_CWTTOBLOCK TaskInfo("b.scale = %g", b.scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk_first_scale = %g", chunk_first_scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk_last_scale = %g", chunk_last_scale);
    DEBUG_CWTTOBLOCK TaskInfo("merge_first_scale = %g", merge_first_scale);
    DEBUG_CWTTOBLOCK TaskInfo("merge_last_scale = %g", merge_last_scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk.nScales() = %u", chunk.nScales());
    DEBUG_CWTTOBLOCK TaskInfo("_collection->scales_per_block() = %u", _collection->scales_per_block());


    Signal::Interval transfer = transferDesc.spannedInterval();

    DEBUG_CWTTOBLOCK {
        TaskTimer("inInterval [%u,%u)", inInterval.first, inInterval.last).suppressTiming();
        TaskTimer("outInterval [%u,%u)", outInterval.first, outInterval.last).suppressTiming();
        TaskTimer("chunk.first_valid_sample = %u", chunk.first_valid_sample).suppressTiming();
        TaskTimer("chunk.n_valid_samples = %u", chunk.n_valid_samples).suppressTiming();
    }

    ComputationCheckError();

    //CWTTOBLOCK_INFO TaskTimer("CwtToBlock [(%g %g), (%g %g)] <- [(%g %g), (%g %g)] |%g %g|",
    Position s, sblock, schunk;
    TIME_CWTTOBLOCK
    {
        Position ia, ib; // intersect
        ia.time = std::max(a.time, chunk_a.time);
        ia.scale = std::max(a.scale, chunk_a.scale);
        ib.time = std::min(b.time, chunk_b.time);
        ib.scale = std::min(b.scale, chunk_b.scale);
        s = Position ( ib.time - ia.time, ib.scale - ia.scale);
        sblock = Position( b.time - a.time, b.scale - a.scale);
        schunk = Position( chunk_b.time - chunk_a.time, chunk_b.scale - chunk_a.scale);
    }

    TIME_CWTTOBLOCK TaskTimer tt("CwtToBlock [(%.2f %.2f), (%.2f %.2f)] <- [(%.2f %.2f), (%.2f %.2f)] |%.2f %.2f, %.2f %.2f| %ux%u=%u <- %ux%u=%u",
            a.time, b.time,
            a.scale, b.scale,
            chunk_a.time, chunk_b.time,
            chunk_a.scale, chunk_b.scale,
            transfer.first/chunk.original_sample_rate, transfer.last/chunk.original_sample_rate,
            merge_first_scale, merge_last_scale,
            (unsigned)(s.time / sblock.time * block->ref.samplesPerBlock() + .5f),
            (unsigned)(s.scale / sblock.scale * block->ref.scalesPerBlock() + .5f),
            (unsigned)(s.time / sblock.time * block->ref.samplesPerBlock() *
            s.scale / sblock.scale * block->ref.scalesPerBlock() + .5f),
            (unsigned)(s.time / schunk.time * chunk.n_valid_samples + .5f),
            (unsigned)(s.scale / schunk.scale * chunk.nScales() + .5f),
            (unsigned)(s.time / schunk.time * chunk.n_valid_samples *
            s.scale / schunk.scale * chunk.nScales() + .5f)
        );

    BOOST_ASSERT( chunk.first_valid_sample+chunk.n_valid_samples <= chunk.transform_data->getNumberOfElements().width );


    // Invoke kernel execution to merge chunk into block
    ::blockResampleChunk( chunk.transform_data,
                     outData,
                     ValidInputInterval( chunk.first_valid_sample, chunk.first_valid_sample+chunk.n_valid_samples ),
                     //make_uint2( 0, chunk.transform_data->getNumberOfElements().width ),
                     ResampleArea( chunk_a.time, chunk_a.scale,
                                  //chunk_b.time, chunk_b.scale+(chunk_b.scale==1?0.01:0) ), // numerical error workaround, only affects visual
                                 chunk_b.time, chunk_b.scale  ), // numerical error workaround, only affects visual
                     ResampleArea( a.time, a.scale,
                                  b.time, b.scale ),
                     complex_info,
                     chunk.freqAxis,
                     _collection->display_scale(),
                     _collection->amplitude_axis(),
                     normalization_factor,
                     full_resolution
                     );


    ComputationCheckError();
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
    block->non_zero |= transfer;

    DEBUG_CWTTOBLOCK {
        TaskInfo ti("Block filter input and output %s", block->ref.toString().c_str());
        DataStorageSize sz = chunk.transform_data->size();
        sz.width *= 2;
        Statistics<float> o1( CpuMemoryStorage::BorrowPtr<float>( sz, (float*)CpuMemoryStorage::ReadOnly<2>(chunk.transform_data).ptr() ));
        Statistics<float> o2( outData );
    }
    TIME_CWTTOBLOCK ComputationSynchronize();
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
    Cwt* cwt = dynamic_cast<Cwt*>(transform().get());
    BOOST_ASSERT( cwt );
    bool full_resolution = cwt->wavelet_time_support() >= cwt->wavelet_default_time_support();
    float normalization_factor = cwt->nScales( chunk.original_sample_rate )/cwt->sigma();

    CwtChunk& chunks = *dynamic_cast<CwtChunk*>( &chunk );

    BOOST_FOREACH( const pChunk& chunkpart, chunks.chunks )
    {
        mergeRowMajorChunk( block, *chunkpart, outData,
                            full_resolution, complex_info, normalization_factor );
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
    StftChunk* stftchunk = dynamic_cast<StftChunk*>(&chunk);
    BOOST_ASSERT( stftchunk );
    float normalization_factor = 1.f/sqrtf(stftchunk->window_size());
    mergeColumnMajorChunk(block, chunk, outData, normalization_factor);
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
    float normalization_factor = 1.f; // already normalized when return from Cepstrum.cpp
    mergeColumnMajorChunk(block, chunk, outData, normalization_factor);
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
        first.first = missingSamples.spannedInterval().first;
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
    BOOST_FOREACH( boost::shared_ptr<Collection> c, *_collections )
    {
        Tfr::FreqAxis fa = c->display_scale();
        if (fa.min_hz != chunk.freqAxis.min_hz || fa.axis_scale != Tfr::AxisScale_Linear)
        {
            BOOST_ASSERT( fa.max_frequency_scalar == 1.f );
            fa.axis_scale = Tfr::AxisScale_Linear;
            fa.min_hz = chunk.freqAxis.min_hz;
            fa.f_step = -2*fa.min_hz;
            c->display_scale( fa );
        }
    }


    DrawnWaveformChunk* dwc = dynamic_cast<DrawnWaveformChunk*>(&chunk);

    float block_fs = block->ref.sample_rate();

    if (dwc->block_fs != block_fs)
        return;

    mergeRowMajorChunk(block, chunk, outData, true, ComplexInfo_Amplitude_Non_Weighted, 1.f);
}

} // namespace Heightmap
