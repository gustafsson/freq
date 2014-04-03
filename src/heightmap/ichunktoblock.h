#ifndef HEIGHTMAP_ICHUNKTOBLOCK_H
#define HEIGHTMAP_ICHUNKTOBLOCK_H

#include "block.h"
#include <memory>

namespace Heightmap {

class IChunkToBlock
{
public:
    typedef std::shared_ptr<IChunkToBlock> ptr;

    IChunkToBlock() {}
    IChunkToBlock(const IChunkToBlock&) = delete;
    IChunkToBlock& operator=(const IChunkToBlock&) = delete;
    virtual ~IChunkToBlock() {}

    float normalization_factor;

    virtual void init() = 0;
    virtual void prepareTransfer() = 0;
    virtual void prepareMerge(AmplitudeAxis amplitude_axis, Tfr::FreqAxis display_scale, BlockLayout bl) = 0;
    virtual void mergeChunk( pBlock block ) = 0;
};

} // namespace Heightmap

#endif // HEIGHTMAP_ICHUNKTOBLOCK_H
