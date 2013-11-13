#ifndef SIGNAL_PROCESSING_TARGETMARKER_H
#define SIGNAL_PROCESSING_TARGETMARKER_H

#include "targetneeds.h"

namespace Signal {
namespace Processing {

class Dag;
class TargetNeeds;

/**
 * @brief The TargetMarker class should mark the position of a target in the
 * dag. It also functions as an insertion point to add operations to a signal
 * processing chain. When it is removed all operations that can only be reached
 * from this target are also removed.
 *
 * Analogy: "marker" as in label in the sense of a version control branch
 */
class TargetMarker
{
public:
    typedef boost::shared_ptr<TargetMarker> Ptr;

    TargetMarker(boost::shared_ptr<volatile TargetNeeds> target_needs, boost::shared_ptr<volatile Dag> dag);
    ~TargetMarker();

    boost::shared_ptr<volatile TargetNeeds> target_needs() const;
    boost::weak_ptr<volatile Step> step() const;

private:
    boost::shared_ptr<volatile TargetNeeds> target_needs_;
    boost::shared_ptr<volatile Dag> dag_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETMARKER_H
