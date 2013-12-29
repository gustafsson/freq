#ifndef FILTERS_MOVE_H
#define FILTERS_MOVE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class Move: public Tfr::ChunkFilter
{
public:
    Move(float df);

    void operator()( Tfr::ChunkAndInverse& chunk );

    float _df;
};


class MoveDesc: public Tfr::CwtFilterDesc {
public:
    MoveDesc(float df):Tfr::CwtFilterDesc(Tfr::pChunkFilter(new Move(df))){}
};


} // namespace Filters

#endif // FILTERS_MOVE_H
