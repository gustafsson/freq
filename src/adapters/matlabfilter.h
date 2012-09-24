#ifndef ADAPTERS_MATLABFILTER_H
#define ADAPTERS_MATLABFILTER_H

#include "tfr/cwtfilter.h"
#include "matlaboperation.h"

namespace Adapters {

class MatlabFilter: public Tfr::CwtFilter
{
public:
    MatlabFilter( std::string matlabFunction );

    virtual bool operator()( Tfr::Chunk& );
    virtual Signal::Intervals ZeroedSamples() const;
    virtual Signal::Intervals affected_samples() const;

    void restart();
protected:
    boost::scoped_ptr<MatlabFunction> _matlab;
    Signal::Intervals _invalid_returns;
};

} // namespace Adapters

#endif // ADAPTERS_MATLABFILTER_H
