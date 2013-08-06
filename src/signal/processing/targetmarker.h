#ifndef SIGNAL_PROCESSING_TARGETMARKER_H
#define SIGNAL_PROCESSING_TARGETMARKER_H

#include "targetneeds.h"

namespace Signal {
namespace Processing {

class Dag;

/**
 * @brief The TargetMarker class should mark the position of a target in the dag and remove it's vertices when the marker is deleted.
 */
class TargetMarker: public VolatilePtr<TargetMarker>
{
public:
    TargetMarker(TargetNeeds::Ptr target_needs, boost::shared_ptr<volatile Dag> dag);
    ~TargetMarker();

    /**
     * @see TargetNeeds::updateNeeds
     */
    void updateNeeds(
            Signal::Intervals needed_samples,
            int prio=0,
            Signal::IntervalType center=Signal::Interval::IntervalType_MIN,
            Signal::Intervals invalidate=Signal::Intervals());

    boost::weak_ptr<volatile Step> step() const;
    bool sleep(int sleep_ms) volatile;

private:
    TargetNeeds::Ptr target_needs_;
    boost::shared_ptr<volatile Dag> dag_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETMARKER_H
