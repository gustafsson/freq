#ifndef PEAKFILTER_H
#define PEAKFILTER_H
#if 0
#include "tfr/cwtfilter.h"
#include "tools/support/brushfilter.h"

namespace Filters {

class PeakFilter : public Tfr::CwtFilter
{
public:
    PeakFilter();


    Tools::Support::MultiplyBrush brush;



    virtual Signal::Intervals affected_samples()
    {
        return brush.affected_samples();
    }


    virtual void operator()( Tfr::Chunk& chunk )
    {
        return brush( chunk );
    }
};

} // namespace Filters
#endif // 0
#endif // PEAKFILTER_H
