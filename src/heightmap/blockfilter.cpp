#include "blockfilter.h"

#include "blockkernel.h"
#include "collection.h"
#include "renderer.h"
#include "glblock.h"
#include "tfr/cwt.h"
#include "tfr/cwtchunk.h"
#include "tfr/drawnwaveform.h"
#include "tfr/stft.h"
#include "tfr/cepstrum.h"

#include <computationkernel.h>
#include <TaskTimer.h>
#include <Statistics.h>

#include <boost/foreach.hpp>
#include <boost/format.hpp>

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
using namespace boost;

namespace Heightmap
{

BlockFilter::
        BlockFilter( Collection* collection )
{
    _collections.resize (1);
    _collections[0] = collection;
}


BlockFilter::
        BlockFilter( std::vector<boost::shared_ptr<Collection> >* collections )
{
    if (!collections)
        return;

    _collections.resize (collections->size ());
    for (unsigned c=0; c<collections->size (); ++c)
        _collections[c] = (*collections)[c].get();
}


bool BlockFilter::
        applyFilter( ChunkAndInverse& pchunk )
{
    Collection* collection = _collections[pchunk.channel];
    Tfr::Chunk& chunk = *pchunk.chunk;
    Signal::Interval chunk_interval = chunk.getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = collection->getIntersectingBlocks( chunk_interval, false );
    TIME_BLOCKFILTER TaskTimer tt(format("BlockFilter %s [%g %g] Hz, intersects with %u visible blocks")
        % chunk_interval % chunk.minHz() % chunk.maxHz() % intersecting_blocks.size());

    BOOST_FOREACH( pBlock block, intersecting_blocks)
    {
        if (((block->getInterval() - block->valid_samples) & chunk_interval).empty() )
            continue;

#ifndef SAWE_NO_MUTEX
        if (collection->constructor_thread().isSameThread())
        {
#endif
            mergeChunk( block, pchunk, block->glblock->height()->data );

            TIME_BLOCKFILTER ComputationCheckError();
#ifndef SAWE_NO_MUTEX
        }
        else
        {
            QMutexLocker l(&block->cpu_copy_mutex);
            if (!block->cpu_copy)
            {
                TaskInfo(format("%s") %
                         Heightmap::ReferenceInfo(
                             block->reference (),
                             collection->block_configuration ()
                             ));
                EXCEPTION_ASSERT( block->cpu_copy );
            }

            mergeChunk( block, pchunk, block->cpu_copy );

            block->cpu_copy->OnlyKeepOneStorage<CpuMemoryStorage>();

            block->new_data_available = true;
        }
#endif
    }


    TIME_BLOCKFILTER ComputationSynchronize();

    return false;
}


void BlockFilter::
        mergeColumnMajorChunk( pBlock block, const ChunkAndInverse& pchunk, Block::pData outData, float normalization_factor )
{
    Collection* collection = _collections[pchunk.channel];
    Heightmap::BlockConfiguration block_config = collection->block_configuration ();

    Tfr::Chunk& chunk = *pchunk.chunk;
    Region r = block->getRegion();

    Position chunk_a, chunk_b;
    //Signal::Interval inInterval = chunk.getInterval();
    Signal::Interval inInterval = chunk.getCoveredInterval();
    Signal::Interval blockInterval = block->getInterval();

    // don't validate more texels than we have actual support for
    Signal::Interval spannedBlockSamples(0,0);
    Heightmap::ReferenceInfo ri(block->reference (), block_config);
    Signal::Interval usableInInterval = ri.spannedElementsInterval(inInterval, spannedBlockSamples);

    Signal::Interval transfer = usableInInterval&blockInterval;

#ifdef _DEBUG
    if (!transfer || !spannedBlockSamples)
    {
        TaskInfo(format("Warning: !transfer || !spannedBlockSamples.~\nspannedBlockSamples=%s\ntransfer=%s\nusableInInterval=%s\nblockInterval=%s") % spannedBlockSamples % transfer % usableInInterval % blockInterval);
    }
#endif

    if (!transfer || !spannedBlockSamples)
        return;

#ifdef _DEBUG
//    blockInterval = block->getInterval();
//    Signal::Interval blockSpannedBlockSamples(0,0);
//    Signal::Interval usableBlockInterval = block->reference().spannedElementsInterval(blockInterval, blockSpannedBlockSamples);
//    blockInterval = block->getInterval();
//    usableBlockInterval = block->reference().spannedElementsInterval(blockInterval, blockSpannedBlockSamples);
//    Signal::Intervals invalidBlockInterval = blockInterval - block->valid_samples;

//    if (blockSpannedBlockSamples.count() != block->reference().samplesPerBlock())
//    {
//        Signal::Interval blockSpannedBlockSamples2(0,0);
//        Signal::Interval usableBlockInterval = block->reference().spannedElementsInterval(blockInterval, blockSpannedBlockSamples2);
//    }

//    float s1=(spannedBlockSamples.first - .5*0 - 1.5) / (float)(block->reference().samplesPerBlock()-1);
//    float s2=(spannedBlockSamples.last - .5*0 + 1.5) / (float)(block->reference().samplesPerBlock()-1);
//    float t1=(transfer.first - blockInterval.first) / (float)blockInterval.count();
//    float t2=(transfer.last - blockInterval.first) / (float)blockInterval.count();

//    Signal::Interval spannedBlockSamples3(0,0);
//    Signal::Interval usableInInterval3 = block->reference().spannedElementsInterval(inInterval, spannedBlockSamples3);

//    if( s2-s1 < t2-t1 || spannedBlockSamples.count()>4)
//    {
//        Signal::Interval spannedBlockSamples2(0,0);
//        Signal::Interval usableInInterval2 = block->reference().spannedElementsInterval(inInterval, spannedBlockSamples2);
//    }

//    EXCEPTION_ASSERT( s2 >= t2 );
//    EXCEPTION_ASSERT( s1 <= t1 );
//    EXCEPTION_ASSERT(blockSpannedBlockSamples.count() == block->reference().samplesPerBlock());
//    EXCEPTION_ASSERT(usableBlockInterval == blockInterval);
//    EXCEPTION_ASSERT(usableInInterval.first >= inInterval.first || usableInInterval.last <= inInterval.last);
#endif

    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

    {
    TIME_BLOCKFILTER TaskTimer tt("resampleStft");
    ::resampleStft( chunk.transform_data,
                    chunk.nScales(),
                    chunk.nSamples(),
                  outData,
                  ValidInterval(spannedBlockSamples.first, spannedBlockSamples.last),
                  ResampleArea( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  ResampleArea( r.a.time, r.a.scale,
                               r.b.time, r.b.scale ),
                  chunk.freqAxis,
                  block_config.display_scale (),
                  block_config.amplitude_axis (),
                  normalization_factor,
                  true);
    TIME_BLOCKFILTER ComputationSynchronize();
    }

    DEBUG_CWTTOBLOCK TaskInfo(format("Validating %s in %s (was %s)")
            % transfer
            % Heightmap::ReferenceInfo(block->reference (), collection->block_configuration ())
            % block->valid_samples);
    block->valid_samples |= transfer;
    block->non_zero |= transfer;
}


void BlockFilter::
        mergeRowMajorChunk( pBlock block, const ChunkAndInverse& pchunk, Block::pData outData,
                            bool full_resolution, ComplexInfo complex_info,
                            float normalization_factor, bool enable_subtexel_aggregation)
{
    Collection* collection = _collections[pchunk.channel];
    Heightmap::BlockConfiguration block_config = collection->block_configuration ();
    Tfr::Chunk& chunk = *pchunk.chunk;

    ComputationCheckError();

    //unsigned cuda_stream = 0;

    // Find out what intervals that match
    Signal::Interval outInterval = block->getInterval();
    Signal::Interval inInterval = chunk.getCoveredInterval();

    // don't validate more texels than we have actual support for
    //Signal::Interval usableInInterval = block->ref.spannedElementsInterval(inInterval);
    Signal::Interval usableInInterval = inInterval;

    Signal::Interval transfer = usableInInterval & outInterval;

    DEBUG_CWTTOBLOCK TaskTimer tt3("CwtToBlock::mergeChunk");
    DEBUG_CWTTOBLOCK TaskTimer("outInterval=[%g, %g)",
            outInterval.first / chunk.original_sample_rate,
            outInterval.last / chunk.original_sample_rate ).suppressTiming();
    DEBUG_CWTTOBLOCK TaskTimer("inInterval=[%g, %g)",
            inInterval.first / chunk.original_sample_rate,
            inInterval.last / chunk.original_sample_rate ).suppressTiming();
    DEBUG_CWTTOBLOCK TaskTimer("transfer=[%g, %g)",
            transfer.first / chunk.original_sample_rate,
            transfer.last / chunk.original_sample_rate ).suppressTiming();

    // Remove already computed intervals
    if (!full_resolution)
    {
        if (!(transfer - block->valid_samples))
        {
            TIME_CWTTOBLOCK TaskInfo(format("%s not accepting %s, early termination") % vartype(*this) % transfer);
            transfer.last=transfer.first;
        }
    }
    // transferDesc -= block->valid_samples;

    // If block is already up to date, abort merge
    if (!transfer)
    {
        TIME_CWTTOBLOCK TaskInfo tt("CwtToBlock::mergeChunk, transfer empty");
        TIME_CWTTOBLOCK TaskInfo(format("outInterval = %s") % outInterval);
        TIME_CWTTOBLOCK TaskInfo(format("inInterval = %s") % inInterval);

        return;
    }

    Region r = block->getRegion();
    float chunk_startTime = (chunk.chunk_offset.asFloat() + chunk.first_valid_sample)/chunk.sample_rate;
    float chunk_length = chunk.n_valid_samples / chunk.sample_rate;
    DEBUG_CWTTOBLOCK TaskTimer tt2("CwtToBlock::mergeChunk chunk t=[%g, %g) into block t=[%g,%g] ff=[%g,%g]",
                                 chunk_startTime, chunk_startTime + chunk_length, r.a.time, r.b.time, r.a.scale, r.b.scale);

    float merge_first_scale = r.a.scale;
    float merge_last_scale = r.b.scale;
    float chunk_first_scale = block_config.display_scale().getFrequencyScalar( chunk.minHz() );
    float chunk_last_scale = block_config.display_scale().getFrequencyScalar( chunk.maxHz() );

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
                  r.a.scale, r.b.scale,
                  chunk_first_scale, chunk_last_scale,
                  chunk.minHz(), chunk.maxHz()).suppressTiming();
        return;
    }

    Position chunk_a, chunk_b;
    chunk_a.scale = chunk_first_scale;
    chunk_b.scale = chunk_last_scale;
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    DEBUG_CWTTOBLOCK TaskInfo("r.a.scale = %g", r.a.scale);
    DEBUG_CWTTOBLOCK TaskInfo("r.b.scale = %g", r.b.scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk_first_scale = %g", chunk_first_scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk_last_scale = %g", chunk_last_scale);
    DEBUG_CWTTOBLOCK TaskInfo("merge_first_scale = %g", merge_first_scale);
    DEBUG_CWTTOBLOCK TaskInfo("merge_last_scale = %g", merge_last_scale);
    DEBUG_CWTTOBLOCK TaskInfo("chunk.nScales() = %u", chunk.nScales());
    DEBUG_CWTTOBLOCK TaskInfo("blockconfig.scalesPerBlock() = %u", block_config.scalesPerBlock ());


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
        ia.time = std::max(r.a.time, chunk_a.time);
        ia.scale = std::max(r.a.scale, chunk_a.scale);
        ib.time = std::min(r.b.time, chunk_b.time);
        ib.scale = std::min(r.b.scale, chunk_b.scale);
        s = Position ( ib.time - ia.time, ib.scale - ia.scale);
        sblock = Position( r.b.time - r.a.time, r.b.scale - r.a.scale);
        schunk = Position( chunk_b.time - chunk_a.time, chunk_b.scale - chunk_a.scale);
    }

    int samplesPerBlock = outData->size ().width;
    int scalesPerBlock = outData->size ().height;
    TIME_CWTTOBLOCK TaskTimer tt("CwtToBlock [(%.2f %.2f), (%.2f %.2f)] <- [(%.2f %.2f), (%.2f %.2f)] |%.2f %.2f, %.2f %.2f| %ux%u=%u <- %ux%u=%u",
            r.a.time, r.b.time,
            r.a.scale, r.b.scale,
            chunk_a.time, chunk_b.time,
            chunk_a.scale, chunk_b.scale,
            transfer.first/chunk.original_sample_rate, transfer.last/chunk.original_sample_rate,
            merge_first_scale, merge_last_scale,
            (unsigned)(s.time / sblock.time * samplesPerBlock + .5f),
            (unsigned)(s.scale / sblock.scale * scalesPerBlock + .5f),
            (unsigned)(s.time / sblock.time * samplesPerBlock *
            s.scale / sblock.scale * scalesPerBlock + .5f),
            (unsigned)(s.time / schunk.time * chunk.n_valid_samples + .5f),
            (unsigned)(s.scale / schunk.scale * chunk.nScales() + .5f),
            (unsigned)(s.time / schunk.time * chunk.n_valid_samples *
            s.scale / schunk.scale * chunk.nScales() + .5f)
        );

    EXCEPTION_ASSERT( chunk.first_valid_sample+chunk.n_valid_samples <= chunk.transform_data->size().width );

    enable_subtexel_aggregation &= full_resolution;

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    enable_subtexel_aggregation = false;
#endif

    // Invoke kernel execution to merge chunk into block
    {
    TIME_BLOCKFILTER TaskTimer tt("blockResampleChunk");
    ::blockResampleChunk( chunk.transform_data,
                     outData,
                     ValidInterval( chunk.first_valid_sample, chunk.first_valid_sample+chunk.n_valid_samples ),
                     //make_uint2( 0, chunk.transform_data->getNumberOfElements().width ),
                     ResampleArea( chunk_a.time, chunk_a.scale,
                                  //chunk_b.time, chunk_b.scale+(chunk_b.scale==1?0.01:0) ), // numerical error workaround, only affects visual
                                 chunk_b.time, chunk_b.scale  ), // numerical error workaround, only affects visual
                     ResampleArea( r.a.time, r.a.scale,
                                  r.b.time, r.b.scale ),
                     complex_info,
                     chunk.freqAxis,
                     block_config.display_scale(),
                     block_config.amplitude_axis(),
                     normalization_factor,
                     enable_subtexel_aggregation
                     );
    }

    ComputationCheckError();

    if( full_resolution )
    {
        block->valid_samples |= transfer;
    }
    else
    {
        block->valid_samples -= transfer;
        TIME_CWTTOBLOCK TaskInfo(format("%s not accepting %s") % vartype(*this) % transfer);
    }
    block->non_zero |= transfer;

    DEBUG_CWTTOBLOCK {
        TaskInfo ti(format("Block filter input and output %s") % block->reference());
        DataStorageSize sz = chunk.transform_data->size();
        sz.width *= 2;
        Statistics<float> o1( CpuMemoryStorage::BorrowPtr<float>( sz, (float*)CpuMemoryStorage::ReadOnly<2>(chunk.transform_data).ptr() ));
        Statistics<float> o2( outData );
    }
    TIME_CWTTOBLOCK ComputationSynchronize();
}


unsigned BlockFilter::
        smallestOk(const Signal::Interval& I)
{
    Collection* collection = _collections[0];
    float FS = collection->target->sample_rate();
    long double min_fs = FS;
    std::vector<pBlock> intersections = collection->getIntersectingBlocks( I?I:collection->invalid_samples(), true );
    BOOST_FOREACH( pBlock b, intersections )
    {
        if (!(b->getInterval() - b->valid_samples))
            continue;

        long double fs = b->sample_rate();
        min_fs = std::min( min_fs, fs );
    }

    unsigned r = ceil( 2 * FS/min_fs );
    return r;
}


//////////////////////////////// CwtToBlock ///////////////////////////////

CwtToBlock::
        CwtToBlock( std::vector<boost::shared_ptr<Collection> >* collections, Renderer* renderer )
            :
            BlockFilterImpl<Tfr::CwtFilter>(collections),
            complex_info(ComplexInfo_Amplitude_Non_Weighted),
            renderer(renderer)
{
}


void CwtToBlock::
        mergeChunk( pBlock block, const ChunkAndInverse& chunk, Block::pData outData )
{
    Cwt* cwt = dynamic_cast<Cwt*>(transform().get());
    EXCEPTION_ASSERT( cwt );
    bool full_resolution = cwt->wavelet_time_support() >= cwt->wavelet_default_time_support();
    float normalization_factor = cwt->nScales( chunk.chunk->original_sample_rate )/cwt->sigma();

    CwtChunk& chunks = *dynamic_cast<CwtChunk*>( chunk.chunk.get () );

    BOOST_FOREACH( const pChunk& chunkpart, chunks.chunks )
    {
        ChunkAndInverse c = chunk;
        c.chunk = chunkpart;
        mergeRowMajorChunk( block, c, outData,
                            full_resolution, complex_info, normalization_factor, renderer->redundancy() <= 1 );
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
        mergeChunk( pBlock block, const ChunkAndInverse& chunk, Block::pData outData )
{
    StftChunk* stftchunk = dynamic_cast<StftChunk*>(chunk.chunk.get ());
    EXCEPTION_ASSERT( stftchunk );
    float normalization_factor = 1.f/sqrtf(stftchunk->window_size());
    mergeColumnMajorChunk(block, chunk, outData, normalization_factor);
}


///////////////////////////// CepstrumToBlock /////////////////////////////

CepstrumToBlock::
        CepstrumToBlock( std::vector<boost::shared_ptr<Collection> > *collections )
            :
            BlockFilterImpl<Tfr::CepstrumFilter>(collections)
{
}


void CepstrumToBlock::
        mergeChunk( pBlock block, const ChunkAndInverse& chunk, Block::pData outData )
{
    float normalization_factor = 1.f; // already normalized when return from Cepstrum.cpp
    mergeColumnMajorChunk(block, chunk, outData, normalization_factor);
}


/////////////////////////// DrawnWaveformToBlock ///////////////////////////

DrawnWaveformToBlock::
        DrawnWaveformToBlock( std::vector<boost::shared_ptr<Collection> > *collections )
            :
            BlockFilterImpl<Tfr::DrawnWaveformFilter>(collections)
{
}


Signal::Interval DrawnWaveformToBlock::
        requiredInterval( const Signal::Interval& I, Tfr::pTransform t )
{
    Signal::Intervals missingSamples;
    for (unsigned c=0; c<_collections.size (); ++c)
    {
        Collection* collection = _collections[c];
        std::vector<pBlock> intersecting_blocks = collection->getIntersectingBlocks( I, false );

        BOOST_FOREACH( pBlock block, intersecting_blocks)
        {
            missingSamples |= block->getInterval() - block->valid_samples;
        }
    }

    missingSamples &= I;

    float largest_fs = 0;
    Signal::Interval toCompute = I;
    if (missingSamples)
    {
        Signal::Interval first(0, 0);
        first.first = missingSamples.spannedInterval().first;
        first.last = first.first + 1;

        for (unsigned c=0; c<_collections.size (); ++c)
        {
            Collection* collection = _collections[c];
            std::vector<pBlock> intersecting_blocks = collection->getIntersectingBlocks( I, false );

            BOOST_FOREACH( pBlock block, intersecting_blocks)
            {
                if (((block->getInterval() - block->valid_samples) & first).empty() )
                    continue;

                largest_fs = std::max(largest_fs, block->sample_rate());
            }
        }

        DrawnWaveform* wt = dynamic_cast<DrawnWaveform*>(transform().get());
        wt->block_fs = largest_fs;

        toCompute = missingSamples.fetchFirstInterval();
    }

    return Tfr::DrawnWaveformFilter::requiredInterval (toCompute, t);
}


void DrawnWaveformToBlock::
        mergeChunk( pBlock block, const ChunkAndInverse& pchunk, Block::pData outData )
{
    Collection* c = _collections[pchunk.channel];
    BlockConfiguration blockconfig = c->block_configuration ();
    Chunk& chunk = *pchunk.chunk;
    Tfr::FreqAxis fa = blockconfig.display_scale();
    if (fa.min_hz != chunk.freqAxis.min_hz || fa.axis_scale != Tfr::AxisScale_Linear)
    {
        EXCEPTION_ASSERT( fa.max_frequency_scalar == 1.f );
        fa.axis_scale = Tfr::AxisScale_Linear;
        fa.min_hz = chunk.freqAxis.min_hz;
        fa.f_step = -2*fa.min_hz;
        blockconfig.display_scale( fa );
        c->block_configuration ( blockconfig );
    }


    DrawnWaveformChunk* dwc = dynamic_cast<DrawnWaveformChunk*>(&chunk);

    float block_fs = block->sample_rate();

    if (dwc->block_fs != block_fs)
        return;

    mergeRowMajorChunk(block, pchunk, outData, true, ComplexInfo_Amplitude_Non_Weighted, 1.f, false);
}


BlockFilterDesc::
        BlockFilterDesc(std::vector<boost::shared_ptr<Collection> >* collections, Renderer* renderer, Tfr::pTransformDesc d)
    :
      FilterDesc(d, FilterKernelDesc::Ptr()),
      collections_(collections),
      renderer_(renderer)
{
    EXCEPTION_ASSERTX( false, "Not implemented");
}


Signal::Interval BlockFilterDesc::
        requiredInterval( const Signal::Interval& /*I*/, Signal::Interval* /*expectedOutput*/ ) const
{
    EXCEPTION_ASSERTX( false, "Not implemented");
    return Signal::Interval();
}



Signal::Operation::Ptr BlockFilterDesc::
        createOperation(Signal::ComputingEngine* /*engine*/) const
{
    EXCEPTION_ASSERTX(false, "Not implemented");
    return Signal::Operation::Ptr();
/*    // yeah dynamic_cast is ugly
    Signal::Operation::Ptr r;
    Tfr::pTransformDesc t = transformDesc();
    if (dynamic_cast<Tfr::DrawnWaveform*>(t.get ()))
        r.reset ( new DrawnWaveformToBlock(collections_));
    else if (dynamic_cast<Tfr::CepstrumDesc*>(t.get ()))
        r.reset ( new CepstrumToBlock(collections_));
    else if (dynamic_cast<Tfr::StftDesc*>(t.get ()))
        r.reset ( new StftToBlock(collections_));
    else if (dynamic_cast<Tfr::Cwt*>(t.get ()))
        r.reset ( new CwtToBlock(collections_, renderer_));

    Filter* f = dynamic_cast<Filter*>(r.get ());
    f->transform (t->createTransform ());
    return r;*/
}


Signal::OperationDesc::Ptr BlockFilterDesc::
        copy() const
{
    return Signal::OperationDesc::Ptr(new BlockFilterDesc(collections_, renderer_, transformDesc()));
}


void BlockFilterDesc::
        test()
{
    // ...
}

} // namespace Heightmap
