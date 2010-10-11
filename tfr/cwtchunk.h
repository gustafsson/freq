#ifndef CWTCHUNK_H
#define CWTCHUNK_H

#include "chunk.h"
#include <vector>

namespace Tfr {
    class CwtChunkPart:public Chunk
    {
    public:
        /**
          If chunks are clamped (cleaned from redundant data) the inverse will produce incorrect results.
          */
        boost::shared_ptr< Chunk > cleanChunk() const;

        virtual Signal::Interval getInterval() const;
    };

    class CwtChunk:public Chunk
    {
    public:
        /**
          Collection of CwtChunkPart.
          */
        std::vector<pChunk> chunks;

        virtual Signal::Interval getInterval() const;
    };
} // namespace Tfr
#endif // CWTCHUNK_H
