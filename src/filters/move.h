#ifndef FILTERS_MOVE_H
#define FILTERS_MOVE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class Move: public Tfr::CwtFilter
{
public:
    Move(float df);

    virtual bool operator()( Tfr::Chunk& );

    float _df;
};


} // namespace Filters

#endif // FILTERS_MOVE_H
