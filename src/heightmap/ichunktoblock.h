#ifndef HEIGHTMAP_ICHUNKTOBLOCK_H
#define HEIGHTMAP_ICHUNKTOBLOCK_H

#include "block.h"
#include <memory>

namespace Heightmap {

class IChunkToBlock
{
public:
    typedef std::shared_ptr<IChunkToBlock> Ptr;

    IChunkToBlock() {}
    IChunkToBlock(const IChunkToBlock&) = delete;
    IChunkToBlock& operator=(const IChunkToBlock&) = delete;
    ~IChunkToBlock() {}

    virtual void mergeChunk( pBlock block ) = 0;
};

} // namespace Heightmap

#endif // HEIGHTMAP_ICHUNKTOBLOCK_H
