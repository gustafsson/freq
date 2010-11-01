#ifndef FILTERSELECTION_H
#define FILTERSELECTION_H

#include "tfr/cwtfilter.h"
#include "tools/support/selectionparams.h"


namespace Filters {

class Selection: public Tfr::CwtFilter
{
public:
    Selection( SelectionParams s );

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    SelectionParams s;

private:
    // TODO Why not copyable?
    Selection& operator=(const Selection& );
    Selection(const Selection& );
};

} // namespace Filters

#endif // FILTERSELECTION_H
