#ifndef FILTERS_RECTANGLE_H
#define FILTERS_RECTANGLE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class Rectangle: public Tfr::CwtFilter
{
public:
    Rectangle(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    float _t1, _f1, _t2, _f2;
    bool _save_inside;
};

} // namespace Filters

#endif // FILTERS_RECTANGLE_H
