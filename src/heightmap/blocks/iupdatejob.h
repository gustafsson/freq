#ifndef HEIGHTMAP_BLOCKS_IUPDATEJOB_H
#define HEIGHTMAP_BLOCKS_IUPDATEJOB_H

#include "signal/intervals.h"

#include <memory>

namespace Heightmap {
namespace Blocks {

class IUpdateJob
{
public:
    typedef std::shared_ptr<IUpdateJob> ptr;

    virtual ~IUpdateJob() {}

    virtual Signal::Interval getCoveredInterval() const = 0;
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_IUPDATEJOB_H
