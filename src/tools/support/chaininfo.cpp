#include "chaininfo.h"

using namespace Signal::Processing;

namespace Tools {
namespace Support {

ChainInfo::
        ChainInfo(Signal::Processing::Chain::ConstPtr chain)
    :
      chain_(chain)
{
}


bool ChainInfo::
        hasWork()
{
    return 0 < out_of_date_sum();
}


int ChainInfo::
        n_workers()
{
    Workers::Ptr workers = read1(chain_)->workers();
    return read1(workers)->n_workers();
}


Signal::UnsignedIntervalType ChainInfo::
        out_of_date_sum()
{
    Signal::Intervals I;

    Targets::Ptr targets = read1(chain_)->targets();
    const std::vector<TargetNeeds::Ptr>& T = read1(targets)->getTargets();
    for (auto i=T.begin(); i!=T.end(); i++)
    {
        const TargetNeeds::Ptr& t = *i;
        I |= read1(t)->out_of_date();
    }

    return I.count ();
}



void ChainInfo::
        test()
{
    // It should provide info about the running state of a signal processing chain
    {
        Chain::Ptr cp = Chain::createDefaultChain ();
        ChainInfo c(cp);

        EXCEPTION_ASSERT( !c.hasWork () );
        EXCEPTION_ASSERT( QThread::idealThreadCount () + 1 == c.n_workers () );
    }
}


} // namespace Support
} // namespace Tools
