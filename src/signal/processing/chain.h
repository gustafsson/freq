#ifndef SIGNAL_PROCESSING_CHAIN_H
#define SIGNAL_PROCESSING_CHAIN_H

#include "volatileptr.h"
#include "targets.h"
#include "dag.h"
#include "workers.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Chain class should manage the creation of new signal processing chains.
 */
class Chain: public VolatilePtr<Chain>
{
public:
    static Chain::Ptr createDefaultChain();

    Dag::Ptr dag() const;
    Targets::Ptr targets() const;

private:
    Dag::Ptr dag_;
    Targets::Ptr targets_;
    Workers::Ptr workers_;

    Chain(Dag::Ptr, Targets::Ptr targets, Workers::Ptr workers);

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_CHAIN_H
