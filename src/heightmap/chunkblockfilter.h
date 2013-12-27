#ifndef HEIGHTMAP_CHUNKBLOCKFILTER_H
#define HEIGHTMAP_CHUNKBLOCKFILTER_H

#include "tfr/filter.h"
#include "heightmap/tfrmapping.h"
#include "heightmap/block.h"

namespace Heightmap {

class MergeChunk : public VolatilePtr<MergeChunk> {
public:
    virtual ~MergeChunk() {}

    virtual void prepareChunk(Tfr::ChunkAndInverse&) {}

    virtual void mergeChunk(
            const Heightmap::Block& block,
            const Tfr::ChunkAndInverse& chunk,
            Heightmap::BlockData& outData ) = 0;
};


class MergeChunkDesc : public VolatilePtr<MergeChunkDesc>
{
public:
    virtual ~MergeChunkDesc() {}

    virtual MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine=0) const = 0;
};


/**
 * @brief The ChunkBlockFilter class should use a MergeChunk to update all
 * blocks in a tfrmap that matches a given Tfr::Chunk.
 */
class ChunkBlockFilter: public Tfr::ChunkFilter
{
public:
    ChunkBlockFilter( MergeChunk::Ptr merge_chunk, Heightmap::TfrMapping::Ptr tfrmap );

    void operator()( Tfr::ChunkAndInverse& chunk );

    void set_number_of_channels( unsigned C );

private:
    Heightmap::TfrMapping::Ptr tfrmap_;
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
class ChunkBlockFilterDesc: public Tfr::FilterKernelDesc
{
public:
    ChunkBlockFilterDesc( Heightmap::TfrMapping::Ptr tfrmap );

    /**
     * @brief createChunkFilter creates a ChunkFilter.
     * If 'setMergeChunkDesc' has not been set createChunkFilter will return null.
     * @param engine to create a filter kernel for.
     * @return null or a ChunkFilter for the given engine.
     */
    Tfr::pChunkFilter createChunkFilter( Signal::ComputingEngine* engine=0 ) const;


    void setMergeChunkDesc( MergeChunkDesc::Ptr mcdp ) { merge_chunk_desc_ = mcdp; }

private:
    MergeChunkDesc::Ptr merge_chunk_desc_;
    Heightmap::TfrMapping::Ptr tfrmap_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKBLOCKFILTER_H
