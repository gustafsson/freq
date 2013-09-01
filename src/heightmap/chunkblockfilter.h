#ifndef HEIGHTMAP_CHUNKBLOCKFILTER_H
#define HEIGHTMAP_CHUNKBLOCKFILTER_H

#include "tfr/filter.h"
#include "heightmap/tfrmapping.h"

namespace Heightmap {


class ChunkBlockFilterKernel : public Tfr::ChunkFilter
{
public:
    ChunkBlockFilterKernel(Heightmap::TfrMap::Ptr tfrmap);

    virtual bool applyFilter( Tfr::ChunkAndInverse& chunk );

    virtual bool operator()( Tfr::Chunk& );

private:
    Heightmap::TfrMap::Ptr tfrmap_;
};


class ChunkBlockFilterKernelDesc : public Tfr::FilterKernelDesc
{
public:
    typedef boost::shared_ptr<ChunkBlockFilterKernelDesc> Ptr;

    ChunkBlockFilterKernelDesc(Heightmap::TfrMap::Ptr tfrmap);
    virtual ~ChunkBlockFilterKernelDesc();

    virtual Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine=0) const;

private:
    Heightmap::TfrMap::Ptr tfrmap_;
};


class CreateChunkBlockFilter {
public:
    static Signal::OperationDesc::Ptr createOperationDesc (Heightmap::TfrMap::Ptr tfrmap);

    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKBLOCKFILTER_H
