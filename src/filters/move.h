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


class MoveDesc: public Tfr::CwtChunkFilterDesc {
public:
    MoveDesc(float df):df(df) {}

    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine=0) const {
        return Tfr::pChunkFilter(new Move(df));
    }

private:
    float df;
};


} // namespace Filters

#endif // FILTERS_MOVE_H
