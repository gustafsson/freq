#ifndef HEIGHTMAP_CHUNKBLOCKFILTER_H
#define HEIGHTMAP_CHUNKBLOCKFILTER_H

#include "heightmap/tfrmapping.h"
#include "mergechunk.h"
#include "blocks/ichunkmerger.h"

namespace Heightmap {

/**
 * @brief The ChunkBlockFilter class should use a MergeChunk to update all
 * blocks in a tfrmap that matches a given Tfr::Chunk.
 */
class ChunkBlockFilter: public Tfr::ChunkFilter, public Tfr::ChunkFilter::NoInverseTag
{
public:
    ChunkBlockFilter( Blocks::IChunkMerger::Ptr chunk_merger, Heightmap::TfrMapping::ConstPtr tfrmap, MergeChunk::Ptr merge_chunk );

    void operator()( Tfr::ChunkAndInverse& chunk );

    void set_number_of_channels( unsigned C );

private:
    Blocks::IChunkMerger::Ptr chunk_merger_;
    Heightmap::TfrMapping::ConstPtr tfrmap_;
    MergeChunk::Ptr merge_chunk_;

public:
    static void test();
};


/**
 * @brief The ChunkBlockFilterDesc class should instantiate ChunkBlockFilters
 * for different engines.
 *
 * OperationDesc::requiredInterval should take
 * ReferenceInfo::spannedElementsInterval into account
 * and correspondigly ChunkToBlock should only update those texels that have
 * full support.
 */
class ChunkBlockFilterDesc: public Tfr::ChunkFilterDesc
{
public:
    ChunkBlockFilterDesc( Blocks::IChunkMerger::Ptr chunk_merger, Heightmap::TfrMapping::ConstPtr tfrmap );

    /**
     * @brief createChunkFilter creates a ChunkFilter.
     * If 'setMergeChunkDesc' has not been set createChunkFilter will return null.
     * @param engine to create a filter kernel for.
     * @return null or a ChunkFilter for the given engine.
     */
    Tfr::pChunkFilter createChunkFilter( Signal::ComputingEngine* engine=0 ) const;


    void setMergeChunkDesc( MergeChunkDesc::Ptr mcdp ) { merge_chunk_desc_ = mcdp; }

private:
    Blocks::IChunkMerger::Ptr chunk_merger_;
    Heightmap::TfrMapping::ConstPtr tfrmap_;
    MergeChunkDesc::Ptr merge_chunk_desc_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKBLOCKFILTER_H
