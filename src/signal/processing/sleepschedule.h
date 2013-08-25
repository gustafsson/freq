#ifndef SIGNAL_PROCESSING_SCHEDULEGETTASK_H
#define SIGNAL_PROCESSING_SCHEDULEGETTASK_H

#include "ischedule.h"
#include "bedroom.h"

namespace Signal {
namespace Processing {


/**
 * @brief The ScheduleGetTask class should stall callers while waiting for an
 * available task.
 */
class SleepSchedule: public ISchedule
{
public:
    SleepSchedule(Bedroom::Ptr bedroom, ISchedule::Ptr schedule);

    // Sleeps until a task can be returned
    virtual boost::shared_ptr<volatile Task> getTask() volatile;

private:
    Bedroom::Ptr bedroom_;
    ISchedule::Ptr schedule_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_SCHEDULEGETTASK_H
