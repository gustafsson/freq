#include "blockfilter.h"

#include "blockkernel.h"
#include "collection.h"
#include "renderer.h"
#include "glblock.h"
#include "chunktoblock.h"
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
        BlockFilter( Heightmap::TfrMap::Ptr tfr_map )
    :
      tfr_map_( tfr_map )
{
    EXCEPTION_ASSERT( tfr_map );
}


bool BlockFilter::
        applyFilter( ChunkAndInverse& pchunk )
{
    Collection::Ptr collection;

    {
        Heightmap::TfrMap::ReadPtr tfr_map(tfr_map_);
        if (pchunk.channel >= tfr_map->channels()) {
            // Just ignore
            return false;
        }

        collection = tfr_map->collections()[pchunk.channel];
    }

    // pBlock is deleted from the main thread with the OpenGL context.
    // Collection makes sure that it only discards its reference to a block
    // if there is no copy. shared_ptr::unique.

    Tfr::Chunk& chunk = *pchunk.chunk;
    Signal::Interval chunk_interval = chunk.getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = read1(collection)->getIntersectingBlocks( chunk_interval, false );
    TIME_BLOCKFILTER TaskTimer tt(format("BlockFilter %s [%g %g] Hz, intersects with %u visible blocks")
        % chunk_interval % chunk.minHz() % chunk.maxHz() % intersecting_blocks.size());

    BOOST_FOREACH( pBlock block, intersecting_blocks)
    {
//        if (((block->getInterval() - block->valid_samples) & chunk_interval).empty() )
//            continue;

        // Multiple threads could create and share texture data using wglShareLists, but it's not really necessary.
        // More important would perhaps be to ensure that asynchronous transfers are used to transfer the textures,
        // and then to use an fence sync object to prevent it from being used during rendering until it's ready.
        // http://www.opengl.org/wiki/OpenGL_and_multithreading
        // http://www.opengl.org/wiki/Sync_Object
        BlockData::WritePtr bd(block->block_data());
        mergeChunk( *block, pchunk, *bd );

        bd->cpu_copy->OnlyKeepOneStorage<CpuMemoryStorage>();

        // "should" lock this access
        block->new_data_available = true;
    }


    TIME_BLOCKFILTER ComputationSynchronize();

    return false;
}


void BlockFilter::
        mergeColumnMajorChunk( const Block& blockr, const ChunkAndInverse& pchunk, BlockData& outData, float normalization_factor )
{
    Heightmap::ChunkToBlock chunktoblock;
    chunktoblock.normalization_factor = normalization_factor;
    chunktoblock.mergeColumnMajorChunk (blockr, *pchunk.chunk, outData);
}


void BlockFilter::
        mergeRowMajorChunk( const Block& blockr, const ChunkAndInverse& pchunk, BlockData& outData,
                            bool full_resolution, ComplexInfo complex_info,
                            float normalization_factor, bool enable_subtexel_aggregation)
{
    Heightmap::ChunkToBlock chunktoblock;
    chunktoblock.normalization_factor = normalization_factor;
    chunktoblock.enable_subtexel_aggregation = enable_subtexel_aggregation;
    chunktoblock.complex_info = complex_info;
    chunktoblock.full_resolution = full_resolution;
    chunktoblock.mergeRowMajorChunk (blockr, *pchunk.chunk, outData);
}


unsigned BlockFilter::
        smallestOk(const Signal::Interval& I)
{
    TfrMap::pCollection collection = read1(tfr_map_)->collections()[0];
    float FS = read1(tfr_map_)->targetSampleRate();
    long double min_fs = FS;
/*
    Signal::Intervals invalid_samples = write1(collection)->invalid_samples();
    std::vector<pBlock> intersections = write1(collection)->getIntersectingBlocks( I?I:invalid_samples, true );
*/
    std::vector<pBlock> intersections = write1(collection)->getIntersectingBlocks( I, true );
    BOOST_FOREACH( pBlock b, intersections )
    {
/*
        if (!(b->getInterval() - b->valid_samples))
            continue;
*/

        long double fs = b->sample_rate();
        min_fs = std::min( min_fs, fs );
    }

    unsigned r = ceil( 2 * FS/min_fs );
    return r;
}


//////////////////////////////// CwtToBlock ///////////////////////////////

CwtToBlock::
        CwtToBlock( TfrMap::Ptr tfr_map, Renderer* renderer )
            :
            BlockFilterImpl<Tfr::CwtFilter>(tfr_map),
            complex_info(ComplexInfo_Amplitude_Non_Weighted),
            renderer(renderer)
{
}


void CwtToBlock::
        mergeChunk( const Block& block, const ChunkAndInverse& chunk, BlockData& outData )
{
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(chunk.t.get ());
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
        StftToBlock( TfrMap::Ptr tfr_map )
            :
            BlockFilterImpl<Tfr::StftFilter>(tfr_map)
{
}


void StftToBlock::
        mergeChunk( const Block& block, const ChunkAndInverse& chunk, BlockData& outData )
{
    StftChunk* stftchunk = dynamic_cast<StftChunk*>(chunk.chunk.get ());
    EXCEPTION_ASSERT( stftchunk );
    float normalization_factor = 1.f/sqrtf(stftchunk->window_size());
    mergeColumnMajorChunk(block, chunk, outData, normalization_factor);
}


///////////////////////////// CepstrumToBlock /////////////////////////////

CepstrumToBlock::
        CepstrumToBlock( TfrMap::Ptr tfr_map )
            :
            BlockFilterImpl<Tfr::CepstrumFilter>(tfr_map)
{
}


void CepstrumToBlock::
        mergeChunk( const Block& block, const ChunkAndInverse& chunk, BlockData& outData )
{
    float normalization_factor = 1.f; // already normalized when return from Cepstrum.cpp
    mergeColumnMajorChunk(block, chunk, outData, normalization_factor);
}


/////////////////////////// DrawnWaveformToBlock ///////////////////////////

DrawnWaveformToBlock::
        DrawnWaveformToBlock( TfrMap::Ptr tfr_map )
            :
            BlockFilterImpl<Tfr::DrawnWaveformFilter>(tfr_map)
{
}


Signal::Interval DrawnWaveformToBlock::
        requiredInterval( const Signal::Interval& I, Tfr::pTransform t )
{
    Signal::Intervals missingSamples = I;

    float largest_fs = 0;
    Signal::Interval toCompute = I;
    if (missingSamples)
    {
        TfrMap::Collections collections = read1(tfr_map_)->collections();
        for (unsigned c=0; c<collections.size (); ++c)
        {
            TfrMap::pCollection collection = collections[c];
            std::vector<pBlock> intersecting_blocks = write1(collection)->getIntersectingBlocks( I, false );

            BOOST_FOREACH( pBlock block, intersecting_blocks)
            {
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
        mergeChunk( const Block& block, const ChunkAndInverse& pchunk, BlockData& outData )
{
    TfrMap::Collections collections = read1(tfr_map_)->collections();
    TfrMap::pCollection c = collections[pchunk.channel];
    TfrMapping tm = read1(tfr_map_)->tfr_mapping ();
    Chunk& chunk = *pchunk.chunk;
    Tfr::FreqAxis fa = tm.display_scale;
    if (fa.min_hz != chunk.freqAxis.min_hz || fa.axis_scale != Tfr::AxisScale_Linear)
    {
        EXCEPTION_ASSERT( fa.max_frequency_scalar == 1.f );
        fa.axis_scale = Tfr::AxisScale_Linear;
        fa.min_hz = chunk.freqAxis.min_hz;
        fa.f_step = -2*fa.min_hz;
        write1(tfr_map_)->display_scale( fa );
    }


    DrawnWaveformChunk* dwc = dynamic_cast<DrawnWaveformChunk*>(&chunk);

    float block_fs = block.sample_rate();

    if (dwc->block_fs != block_fs)
        return;

    mergeRowMajorChunk(block, pchunk, outData, true, ComplexInfo_Amplitude_Non_Weighted, 1.f, false);
}


} // namespace Heightmap
