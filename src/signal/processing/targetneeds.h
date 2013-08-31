#ifndef SIGNAL_PROCESSING_TARGETNEEDS_H
#define SIGNAL_PROCESSING_TARGETNEEDS_H

#include "signal/intervals.h"
#include "volatileptr.h"

#include <boost/date_time/posix_time/ptime.hpp>

namespace Signal {
namespace Processing {

class Step;
class Bedroom;

/**
 * @brief The TargetNeeds class should describe what needs to be computed for a target.
 */
class TargetNeeds: public VolatilePtr<TargetNeeds>
{
public:
    TargetNeeds(boost::weak_ptr<volatile Step> step_, boost::weak_ptr<volatile Bedroom> bedroom);
    ~TargetNeeds();

    /**
     * Large portions of step can be out_of_date yet not needed by the target.
     *
     * Only the part of Step::not_started() that is overlapping with
     * 'needed_samples' provided here will actually be scheduled for
     * calculation (as returned by TargetNeeds::not_started()).
     *
     * @arg needed_samples Which portion that is actually needed by the target.
     * @arg prio A higher number makes sure this TargetNeed is computed before
     *           others.
     * @arg center From where to work of intervals from this->not_started()
     * @arg invalidate Samples to invalidate in the step cache while at it.
     *                 Note that Only the part overlapping with
     *                 arg 'needed_samples' will actually be scheduled for
     *                 calculation (as returned by this->not_started()).
     */
    void updateNeeds(
            Signal::Intervals needed_samples,
            Signal::IntervalType center=Signal::Interval::IntervalType_MIN,
            Signal::IntervalType preferred_update_size=Signal::Interval::IntervalType_MAX,
            Signal::Intervals invalidate=Signal::Intervals(),
            int prio=0 );

    boost::weak_ptr<volatile Step> step() const;
    boost::posix_time::ptime last_request() const;
    Signal::IntervalType work_center() const;
    Signal::IntervalType preferred_update_size() const;
    Signal::Intervals out_of_date() const;
    Signal::Intervals not_started() const;

    /**
     * @brief sleep sleeps the caller until all needed_samples have been provided.
     * @param sleep_ms number of milliseconds to wait, or -1 to wait indefinitely.
     * @return true if all needed_samples were provided before sleep_ms, false otherwise.
     */
    bool sleep(int sleep_ms) volatile;

private:
    const boost::weak_ptr<volatile Step> step_;
    boost::posix_time::ptime last_request_;
    Signal::IntervalType work_center_;
    Signal::IntervalType preferred_update_size_;
    Signal::Intervals needed_samples_;

    boost::weak_ptr<volatile Bedroom> bedroom_;

public:
    static void test();
};

/*

What is a Target?
All targets consists of two logical components

1) Operation. This component stores results in target-specific ways (such as an image for textures, in the audio driver for audio output, on the disk for disk output).
This is done by implementing OperationDesc and Operation::process for a computing engine.

2) Target needs. This component figures out what the target needs next. Each target may compute this in its own way. The result is used by the scheduler.
Each target implemented its own algorithm to determine when it needs to recompute what's needed.
When a target has recomputed what's needed it calls updateTarget().

How does a target wake up things?
How does an updated source wake up things?

DagOperation::deprecateCache iterates through the dag and all steps.
The scheduler can provide an Invalidator object.

class Invalidator {
    virtual void deprecateCache(Step::Ptr at, Signal::Intervals what)=0;
};

This object knows about ScheduleGetTask and calls wakeup.


class TargetUpdater {
    virtual void update(prio, center, intervals);
}
*/

// Adding/removing targets
// It should tell the scheduler that there's another target about

// It should compute what need to be done for a target
// Each target computes that for itself when the target see fit.

// It should provide information about what a target needs
// TargetNeeds { prio, center, intervals }


// It should tell the task finder/scheduler what a target needs
// Each
// How do you add a new Target?
// By telling the scheduler that there's another target about. The scheduler should thus have a weak pointer to the Target.
//
// How do you do that?
// OperationDesc::Ptr od = my_target_operation_desc();
// Step::Ptr place_to_add = how_do_you_figure_out_where_to_add_the_target?
// It should be possible to creating a target without connecting it to the graph.
// Connecting to the graph could be done later.
//
// take2
// OperationDesc::Ptr od = my_target_operation_desc();
// UnconnectedTarget::Ptr target = UnconnectedTarget(od);
// done
//
// adding the target
// Step::Ptr place_to_add = from_some_unknown_location;
// Step::Ptr step = scheduler.addTargetAt(place_to_add, target);
// done
//
// updating what's needed in the target
// step->setInvalid(stuff);
// target->last_request(now);
// target->work_center(
/*class UnconnectedTarget {
public:
    Signal::OperationDesc::Ptr od;

    boost::posix_time::ptime last_request() const;
    Signal::IntervalType work_center() const;

    void last_request(boost::posix_time::ptime);
    void work_center(Signal::IntervalType);
};*/
/*
class TargetInfo {
    TargetInfo(Step::Ptr step);

    Step::Ptr step_;

    Step::Ptr step() const;

    boost::posix_time::ptime last_request() const;
    Signal::IntervalType work_center() const;

    void update(Signal::Intervals missing, Signal::IntervalType center, int prio) {
        step->setInvalid(missing);
        last_request(now + prio*seconds);
        work_center(center);

        wakeup();
    }
};


Step::Ptr scheduler.addTargetAt(Step::Ptr position, UnconnectedTarget::Ptr ut) {
    Step::Ptr targetStep(new Step::Ptr(ut->od, position->sample_rate(), position->num_channels()));
    graph.addStep(targetStep, position);
}
*/
// TargetInfo
// Step::Target target_info = scheduler.createNewTarget();
//


/*    Target(Step::Ptr step);

    Step::Ptr step() const;
*/

    // Is Target::out_of_date any different from Step::out_of_date?
    // The target needs to fiddle with out_of_date. Different types of Step doesn't have any say.
    //
    // Does that make target a special type of step which is otherwise generalised?
    //
    // Is Target different from a Step?
    // Target has a work center, and a last request.
    // Or, Target is something that a scheduler uses to figure out which task to take on.
    //
    // Is Target::out_of_date() a subset of Target::step()->out_of_date()?
    // Yes
    // How do you guarantee that? By a test.
    //
    // When a cache is invalidated in a source. How is the target updated with that information?
    // Through the Step, by asking step()->out_of_date()
    //
    // How does for instance the onscreen/opengl texture target know that it should redraw when it's ready?
    // The OperationDesc associated to the step of a target has all information needed to
    // update the texture and redraw the screen.
    //
    // Does this mean that the OperationDesc also knows what should be drawn?
    // Not necessarily, a separate class is used for drawing the Heightmap.
    // The OperationDesc only updates the Heightmap.
    //
    // The task scheduler should take out_of_date from the Target rather than its associated
    // step to determine if a computation is needed.
    //
    // Should Target be connected to a step through a separate connection object?
    // What is the purpose of Target, what should it do?
    //
    // A target consists of a special OperationDesc and a TargetMissingDataProvider.
    // Should it be possible to instantiate a new target from a description? No.
    // But a Target should be described independently of the graph.
    //virtual Signal::Intervals out_of_date(Signal::Intervals skip = Signal::Intervals()) = 0;


/*class ClientGoalProvider: VolatilePtr<WorkRequests> {
public:
    struct Goal {
        Signal::IntervalType work_center;
        Signal::Intervals out_of_date;
    };

    // Don't consider things that we're already working on
    //
    //
    // To play well with the signal processing chain, a target needs to keep it's Step up-to-date. Providing its own out_of_date/skip puts
    // a lot of requirements on each Target to be thread-safe. Might not even be possible to ensure.
    virtual Goal getGoal() = 0;
};
*/
/*
class Target {
public:
    Signal::OperationDesc::Ptr operation_desc;
    WorkRequests::Ptr work_requests;
};

class TargetStep {
public:
    TargetStep(Target::Ptr t, Step::Ptr s);

    Target::Ptr target() const { return t; }
    Step::Ptr step() const { return s; }

private:
    Target::Ptr t;
    Step::Ptr s;
};
*/

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETNEEDS_H
