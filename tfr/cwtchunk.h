#ifndef CWTCHUNK_H
#define CWTCHUNK_H

#include "chunk.h"
#include <vector>

namespace Tfr {
    class CwtChunk:public Chunk
    {
    public:
        std::vector<pChunk> chunks;

        /**
          If chunks are clamped (cleaned from redundant data) the inverse will produce incorrect results.
          */
        boost::shared_ptr< Chunk > cleanChunk() const;
    };
} // namespace Tfr
#endif // CWTCHUNK_H
