#ifndef HEIGHTMAP_UPDATE_IUPDATEJOB_H
#define HEIGHTMAP_UPDATE_IUPDATEJOB_H

#include "signal/intervals.h"

#include <memory>

namespace Heightmap {
namespace Update {

class IUpdateJob
{
public:
    typedef std::shared_ptr<IUpdateJob> ptr;

    virtual ~IUpdateJob() {}

    virtual Signal::Interval getCoveredInterval() const = 0;
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_IUPDATEJOB_H
