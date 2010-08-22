#ifndef SAWEMATLABFILTER_H
#define SAWEMATLABFILTER_H

#include "tfr/filter.h"
#include "sawe/matlaboperation.h"

namespace Sawe {

class MatlabFilter: public Tfr::Filter
{
public:
    MatlabFilter( std::string matlabFunction );

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::SamplesIntervalDescriptor getZeroSamples( unsigned FS ) const;
    virtual Signal::SamplesIntervalDescriptor getUntouchedSamples( unsigned FS ) const;

protected:
    MatlabFunction _matlab;
};

} // namespace Sawe

#endif // SAWEMATLABFILTER_H
