#ifndef SIGNAL_PROCESSING_TARGETMARKER_H
#define SIGNAL_PROCESSING_TARGETMARKER_H

#include "targetneeds.h"

namespace Signal {
namespace Processing {

class Dag;

/**
 * @brief The TargetMarker class should mark the position of a target in the
 * dag. It also functions as an insertion point to add operations to a signal
 * processing chain. When it is removed all operations that can only be reached
 * from this target are also removed.
 *
 * Analogy: "marker" as in label in the sense of a version control branch
 */
class TargetMarker: public VolatilePtr<TargetMarker>
{
public:
    TargetMarker(TargetNeeds::Ptr target_needs, boost::shared_ptr<volatile Dag> dag);
    ~TargetMarker();

    TargetNeeds::Ptr target_needs() { return target_needs_; }
    TargetNeeds::ConstPtr target_needs() const { return target_needs_; }

    /**
     * @see TargetNeeds::updateNeeds
     */
    void updateNeeds(
            Signal::Intervals needed_samples,
            Signal::IntervalType center=Signal::Interval::IntervalType_MIN,
            Signal::IntervalType preferred=Signal::Interval::IntervalType_MAX,
            Signal::Intervals invalidate=Signal::Intervals(),
            int prio=0);

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
