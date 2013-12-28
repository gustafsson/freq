#ifndef TOOLS_SUPPORT_CHAININFO_H
#define TOOLS_SUPPORT_CHAININFO_H

#include "signal/processing/chain.h"

namespace Tools {
namespace Support {

/**
 * @brief The ChainInfo class should provide info about the runnig state of a
 * signal processing chain.
 */
class ChainInfo
{
public:
    ChainInfo(Signal::Processing::Chain::ConstPtr chain);

    bool hasWork();
    int n_workers();
    int dead_workers();

    Signal::UnsignedIntervalType out_of_date_sum();

private:
    Signal::Processing::Chain::ConstPtr chain_;

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_CHAININFO_H
