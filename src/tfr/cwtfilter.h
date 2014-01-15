#ifndef TFRCWTFILTER_H
#define TFRCWTFILTER_H

#pragma once

#include "chunkfilter.h"

namespace Tfr {

class CwtChunkFilter: public Tfr::ChunkFilter
{
    void operator()( ChunkAndInverse& chunk );

    virtual void subchunk( ChunkAndInverse& chunk ) = 0;
};


class CwtChunkFilterDesc: public Tfr::ChunkFilterDesc
{
public:
    CwtChunkFilterDesc();

    void transformDesc( Tfr::pTransformDesc m );
};


} // namespace Tfr

#endif // TFRCWTFILTER_H
