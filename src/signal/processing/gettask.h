#ifndef SIGNAL_PROCESSING_GETTASK_H
#define SIGNAL_PROCESSING_GETTASK_H

#include "task.h"

namespace Signal {
namespace Processing {

class GetTask: public VolatilePtr<GetTask>
{
public:
    virtual ~GetTask() {}

    virtual Task::Ptr getTask() volatile=0;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GETTASK_H
