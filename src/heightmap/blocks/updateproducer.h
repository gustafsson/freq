#ifndef HEIGHTMAP_BLOCKS_UPDATEPRODUCER_H
#define HEIGHTMAP_BLOCKS_UPDATEPRODUCER_H

#include "heightmap/tfrmapping.h"
#include "heightmap/mergechunk.h"
#include "updatequeue.h"
#include "tfr/chunkfilter.h"

namespace Heightmap {
namespace Blocks {

/**
 * @brief The UpdateProducer class should use a MergeChunk to update all
 * blocks in a tfrmap that matches a given Tfr::Chunk.
 */
class UpdateProducer: public Tfr::ChunkFilter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    UpdateProducer( UpdateQueue::ptr update_queue, Heightmap::TfrMapping::const_ptr tfrmap, MergeChunk::ptr merge_chunk );

    void operator()( Tfr::ChunkAndInverse& chunk );

    void set_number_of_channels( unsigned C );

private:
    UpdateQueue::ptr update_queue_;
    Heightmap::TfrMapping::const_ptr tfrmap_;
    MergeChunk::ptr merge_chunk_;

public:
    static void test();
};


/**
 * @brief The UpdateProducerDesc class should instantiate UpdateProducer
 * for different engines.
 *
 * OperationDesc::requiredInterval should take
 * ReferenceInfo::spannedElementsInterval into account
 * and correspondigly ChunkToBlock should only update those texels that have
 * full support.
 */
class UpdateProducerDesc: public Tfr::ChunkFilterDesc
{
public:
    UpdateProducerDesc( UpdateQueue::ptr update_queue, Heightmap::TfrMapping::const_ptr tfrmap );

    /**
     * @brief createChunkFilter creates a ChunkFilter.
     * If 'setMergeChunkDesc' has not been set createChunkFilter will return null.
     * @param engine to create a filter kernel for.
     * @return null or a ChunkFilter for the given engine.
     */
    Tfr::pChunkFilter createChunkFilter( Signal::ComputingEngine* engine=0 ) const;


    void setMergeChunkDesc( MergeChunkDesc::ptr mcdp ) { merge_chunk_desc_ = mcdp; }

private:
    UpdateQueue::ptr update_queue_;
    Heightmap::TfrMapping::const_ptr tfrmap_;
    MergeChunkDesc::ptr merge_chunk_desc_;

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_UPDATEPRODUCER_H
