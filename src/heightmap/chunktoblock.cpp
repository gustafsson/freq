#include "chunktoblock.h"

#include "blockkernel.h"
#include "tfr/chunk.h"

namespace Heightmap {

ChunkToBlock::
        ChunkToBlock()
    :
      tfr_mapping( BlockSize(1<<8,1<<8),2)
{

}


void ChunkToBlock::mergeColumnMajorChunk(
        const Block& block,
        Tfr::pChunk chunk,
        BlockData& outData )
{
    Region r = block.getRegion();

    Position chunk_a, chunk_b;
    //Signal::Interval inInterval = chunk.getInterval();
    Signal::Interval inInterval = chunk->getCoveredInterval();
    Signal::Interval blockInterval = block.getInterval();

    // don't validate more texels than we have actual support for
    Signal::Interval spannedBlockSamples(0,0);
    ReferenceInfo ri(block.reference (), tfr_mapping);
    Signal::Interval usableInInterval = ri.spannedElementsInterval(inInterval, spannedBlockSamples);

    Signal::Interval transfer = usableInInterval&blockInterval;

    if (!transfer || !spannedBlockSamples)
        return;

    chunk_a.time = inInterval.first/chunk->original_sample_rate;
    chunk_b.time = inInterval.last/chunk->original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

    ::resampleStft( chunk->transform_data,
                    chunk->nScales(),
                    chunk->nSamples(),
                  outData.cpu_copy,
                  ValidInterval(spannedBlockSamples.first, spannedBlockSamples.last),
                  ResampleArea( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  ResampleArea( r.a.time, r.a.scale,
                               r.b.time, r.b.scale ),
                  chunk->freqAxis,
                  tfr_mapping.display_scale,
                  tfr_mapping.amplitude_axis,
                  normalization_factor,
                  true);

}


void ChunkToBlock::mergeRowMajorChunk(
        const Block& block,
        Tfr::pChunk chunk,
        BlockData& outData )
{
    // Find out what intervals that match
    Signal::Interval outInterval = block.getInterval();
    Signal::Interval inInterval = chunk->getCoveredInterval();

    // don't validate more texels than we have actual support for
    //Signal::Interval usableInInterval = block->ref.spannedElementsInterval(inInterval);
    Signal::Interval usableInInterval = inInterval;
    Signal::Interval transfer = usableInInterval & outInterval;

    // If block is already up to date, abort merge
    if (!transfer)
        return;

    Region r = block.getRegion();
//    float chunk_startTime = (chunk.chunk_offset.asFloat() + chunk.first_valid_sample)/chunk.sample_rate;
//    float chunk_length = chunk.n_valid_samples / chunk.sample_rate;
//    DEBUG_CWTTOBLOCK TaskTimer tt2("CwtToBlock::mergeChunk chunk t=[%g, %g) into block t=[%g,%g] ff=[%g,%g]",
//                                 chunk_startTime, chunk_startTime + chunk_length, r.a.time, r.b.time, r.a.scale, r.b.scale);

    float merge_first_scale = r.a.scale;
    float merge_last_scale = r.b.scale;
    float chunk_first_scale = tfr_mapping.display_scale.getFrequencyScalar( chunk->minHz() );
    float chunk_last_scale = tfr_mapping.display_scale.getFrequencyScalar( chunk->maxHz() );

    merge_first_scale = std::max( merge_first_scale, chunk_first_scale );
    merge_last_scale = std::min( merge_last_scale, chunk_last_scale );

    if (merge_first_scale >= merge_last_scale)
        return;

    Position chunk_a, chunk_b;
    chunk_a.scale = chunk_first_scale;
    chunk_b.scale = chunk_last_scale;
    chunk_a.time = inInterval.first/chunk->original_sample_rate;
    chunk_b.time = inInterval.last/chunk->original_sample_rate;

    Position s, sblock, schunk;
    EXCEPTION_ASSERT( chunk->first_valid_sample+chunk->n_valid_samples <= chunk->transform_data->size().width );

    enable_subtexel_aggregation &= full_resolution;

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    enable_subtexel_aggregation = false;
#endif

    // Invoke kernel execution to merge chunk into block
    ::blockResampleChunk( chunk->transform_data,
                     outData.cpu_copy,
                     ValidInterval( chunk->first_valid_sample, chunk->first_valid_sample+chunk->n_valid_samples ),
                     //make_uint2( 0, chunk.transform_data->getNumberOfElements().width ),
                     ResampleArea( chunk_a.time, chunk_a.scale,
                                  //chunk_b.time, chunk_b.scale+(chunk_b.scale==1?0.01:0) ), // numerical error workaround, only affects visual
                                 chunk_b.time, chunk_b.scale  ), // numerical error workaround, only affects visual
                     ResampleArea( r.a.time, r.a.scale,
                                  r.b.time, r.b.scale ),
                     complex_info,
                     chunk->freqAxis,
                     tfr_mapping.display_scale,
                     tfr_mapping.amplitude_axis,
                     normalization_factor,
                     enable_subtexel_aggregation
                     );

}

} // namespace Heightmap

#include "tfr/filter.h"
#include "tfr/stftdesc.h"

namespace Heightmap {

class DummyKernel: public Tfr::ChunkFilter {
    virtual bool operator()( Tfr::Chunk& ) {
        return false;
    }
};

class DummyKernelDesc: public Tfr::FilterKernelDesc {
    virtual Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* =0) const {
        return Tfr::pChunkFilter(new DummyKernel);
    }
};

void ChunkToBlock::
        test()
{
    ChunkToBlock ctb;
    ctb.complex_info = ComplexInfo_Amplitude_Non_Weighted;
    ctb.enable_subtexel_aggregation = false;
    ctb.full_resolution = false;
    ctb.normalization_factor = 1;
    ctb.tfr_mapping = TfrMapping( BlockSize(1<<8,1<<8),100);
    ctb.tfr_mapping.display_scale.setLinear (1);
    ctb.tfr_mapping.amplitude_axis = AmplitudeAxis_Linear;

    Tfr::StftDesc* tfr;
    Tfr::pTransformDesc tdesc( tfr = new Tfr::StftDesc() );
    Tfr::FilterKernelDesc::Ptr fdesc( new DummyKernelDesc );
    Signal::OperationDesc::Ptr desc(new Tfr::FilterDesc(tdesc, fdesc));
    Signal::Operation::Ptr operation = desc->createOperation (0);

    Tfr::TransformKernel* transformkernel = dynamic_cast<Tfr::TransformKernel*>( operation.get ());

    Signal::Interval expectedOutput;
    Signal::Interval requiredInterval = desc->requiredInterval (Signal::Interval (11,31), &expectedOutput);

    Tfr::pTransform t = transformkernel->transform();
    Signal::pMonoBuffer buffer( new Signal::MonoBuffer (requiredInterval, 1));
    Tfr::pChunk chunk = (*t)( buffer );

    Heightmap::Reference ref;

    pBlock block( new Block (ref, ctb.tfr_mapping));
    BlockData blockdata;
    blockdata.cpu_copy.reset( new DataStorage<float>(32,32) );

    ctb.mergeColumnMajorChunk(
            *block,
            chunk,
            blockdata );

    ctb.mergeRowMajorChunk(
            *block,
            chunk,
            blockdata );
}

} // namespace Heightmap
