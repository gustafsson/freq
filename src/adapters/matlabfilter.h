#ifndef ADAPTERS_MATLABFILTER_H
#define ADAPTERS_MATLABFILTER_H

#include "tfr/cwtfilter.h"
#include "matlaboperation.h"

namespace Adapters {

class MatlabFilterKernelDesc: public Tfr::ChunkFilterDesc
{
public:
    MatlabFilterKernelDesc(std::string matlabFunction);

    Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* engine) const;

private:
    std::string matlabFunction;

public:
    static void test();
};


class MatlabFilter: public Tfr::ChunkFilter
{
public:
    MatlabFilter( std::string matlabFunction );

    void operator()( Tfr::ChunkAndInverse& chunk );

    void restart();

protected:
    boost::scoped_ptr<MatlabFunction> _matlab;
    Signal::Intervals _invalid_returns;
};


} // namespace Adapters

#endif // ADAPTERS_MATLABFILTER_H
