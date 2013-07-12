#ifndef SIGNAL_PROCESSING_WORKERS_H
#define SIGNAL_PROCESSING_WORKERS_H

#include "volatileptr.h"

namespace Signal {
namespace Processing {

class Workers: public VolatilePtr<Workers>
{
public:
    virtual ~Workers() {}

    typedef std::vector<Signal::ComputingEngine::Ptr> Engines;
    Engines workers() const { return workers_; }
    size_t n_workers() const { return workers_.size(); }

protected:
    Engines workers_;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKERS_H
