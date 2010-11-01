#ifndef FILTERS_RIDGE_H
#define FILTERS_RIDGE_H

#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"

namespace Filters
{
    class Ridge: public Tfr::CwtFilter
    {
        virtual void operator()( Tfr::Chunk& );
    };
}
#endif // FILTERS_RIDGE_H
