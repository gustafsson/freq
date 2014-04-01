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
    ChunkBlockFilter( Blocks::IChunkMerger::ptr chunk_merger, Heightmap::TfrMapping::const_ptr tfrmap, MergeChunk::ptr merge_chunk );

    void operator()( Tfr::ChunkAndInverse& chunk );

    void set_number_of_channels( unsigned C );

private:
    Blocks::IChunkMerger::ptr chunk_merger_;
    Heightmap::TfrMapping::const_ptr tfrmap_;
    MergeChunk::ptr merge_chunk_;

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
    ChunkBlockFilterDesc( Blocks::IChunkMerger::ptr chunk_merger, Heightmap::TfrMapping::const_ptr tfrmap );

    /**
     * @brief createChunkFilter creates a ChunkFilter.
     * If 'setMergeChunkDesc' has not been set createChunkFilter will return null.
     * @param engine to create a filter kernel for.
     * @return null or a ChunkFilter for the given engine.
     */
    Tfr::pChunkFilter createChunkFilter( Signal::ComputingEngine* engine=0 ) const;


    void setMergeChunkDesc( MergeChunkDesc::ptr mcdp ) { merge_chunk_desc_ = mcdp; }

private:
    Blocks::IChunkMerger::ptr chunk_merger_;
    Heightmap::TfrMapping::const_ptr tfrmap_;
    MergeChunkDesc::ptr merge_chunk_desc_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKBLOCKFILTER_H
