#ifndef SAWEMATLABFILTER_H
#define SAWEMATLABFILTER_H

#include "tfr/cwtfilter.h"
#include "sawe/matlaboperation.h"

namespace Sawe {

class MatlabFilter: public Tfr::CwtFilter
{
public:
    MatlabFilter( std::string matlabFunction );

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals ZeroedSamples() const;
    virtual Signal::Intervals affected_samples() const;

protected:
    MatlabFunction _matlab;
};

} // namespace Sawe

#endif // SAWEMATLABFILTER_H
