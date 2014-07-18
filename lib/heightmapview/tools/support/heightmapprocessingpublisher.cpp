#include "heightmapprocessingpublisher.h"
#include "heightmap/collection.h"
#include "signal/processing/step.h"

#include "tasktimer.h"

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

using namespace Signal;
using namespace Processing;

namespace Tools {
namespace Support {

HeightmapProcessingPublisher::HeightmapProcessingPublisher(
          TargetNeeds::ptr target_needs,
          Heightmap::TfrMapping::const_ptr tfrmapping,
          float* t_center,
          QObject* parent)
    :
      QObject(parent),
      target_needs_(target_needs),
      tfrmapping_(tfrmapping),
      t_center_(t_center),
      last_update_(Interval::Interval_ALL),
      failed_allocation_(false)
{
}


void HeightmapProcessingPublisher::
        setLastUpdatedInterval( Interval last_update )
{
    last_update_ = last_update ? last_update : Interval::Interval_ALL;
}


void HeightmapProcessingPublisher::
        update()
{
    TIME_PAINTGL_DETAILS TaskTimer tt("Find things to work on");

    //Intervals invalid_samples;
    Intervals things_to_add;
    Intervals needed_samples;
    float fs, L;
    Heightmap::TfrMapping::Collections C;
    {
        auto tm = tfrmapping_.read ();
        fs = tm->targetSampleRate();
        L = tm->length();
        C = tm->collections();
    }

    IntervalType center = std::round(*t_center_ * fs);

    for ( auto cp : C )
    {
        auto c = cp.read();
        things_to_add |= c->recently_created();
        needed_samples |= c->needed_samples();
    }

    target_needs_->deprecateCache (things_to_add);

    Interval target_interval(0, std::round(L*fs));
    needed_samples &= target_interval;

    Intervals to_compute;
    if (auto step = target_needs_->step ().lock ())
        to_compute = step.read ()->not_started();
    to_compute |= things_to_add;
    to_compute &= needed_samples;


    // It should update the view (a) in sections equal in size to (the smallest
    // visible block with deprecated content), unless (b) the view isn't
    // currently being invalidated.
    //
    // (case a): default, this makes the size of update sections correlate to
    // the size of visible blocks, it doesn't mean some optimization of
    // updating whole. Interactive feedback is a tradeoff between computing
    // something immediately after a request (latency) to computing everything
    // needed after a request (throughput).
    //
    // (case b): during a recording it is, generally, more important to render
    // newly recorded data than to redraw everything else.
    UnsignedIntervalType update_size = Interval::IntervalType_MAX;
    for ( auto c : C ) for ( auto a : c.raw()->cache()->clone () )
    {
        const Heightmap::Block& b = *a.second;
        Interval i = b.getInterval();

        // If this block overlaps data to be computed
        if (i & to_compute)
            update_size = std::min(update_size, i.count ());
    }

    // If the last invalidated interval is smaller than the suggested
    // update_size use the size of the invalidated interval instead.
    // But only if the last invalidated interval is visible and still invalid.
    if (to_compute.fetchInterval (update_size, center) & last_update_)
        update_size = std::min(update_size, last_update_.count ());

    TIME_PAINTGL_DETAILS TaskInfo(boost::format(
            "RenderView needed_samples = %s, "
            "things_to_add = %s, center = %d, size = %d")
            % needed_samples
            % things_to_add
            % center
            % update_size);

    target_needs_->updateNeeds(
                needed_samples,
                center,
                update_size,
                0
            );

    failed_allocation_ = false;
    for ( auto c : C )
        failed_allocation_ |= c.write ()->failed_allocation ();

    TIME_PAINTGL_DETAILS {
        if (Step::ptr step = target_needs_->step ().lock())
        {
            Intervals not_started = target_needs_->not_started();
            auto stepp = step.read ();
            TaskInfo(boost::format("RenderView step->out_of_date = %s, step->not_started = %s, target_needs->not_started = %s")
                             % stepp->out_of_date()
                             % stepp->not_started()
                             % not_started);
        }
    }
}


bool HeightmapProcessingPublisher::
        isHeightmapDone() const
{
    return !target_needs_->out_of_date();
}


bool HeightmapProcessingPublisher::
        failedAllocation() const
{
    return failed_allocation_;
}

} // namespace Support
} // namespace Tools

#include "signal/processing/bedroom.h"
#include "signal/processing/task.h"
#include "signal/processing/bedroomnotifier.h"
#include <QApplication>
#include <QGLWidget>

namespace Tools {
namespace Support {

void HeightmapProcessingPublisher::
        test()
{
    std::string name = "HeightmapProcessingPublisher";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should update a processing target depending on which things that are
    // missing in a heightmap block cache
    {
        OperationDesc::ptr operation_desc;
        Step::ptr step(new Step(operation_desc));
        Bedroom::ptr bedroom(new Bedroom);
        BedroomNotifier::ptr notifier(new BedroomNotifier(bedroom));
        TargetNeeds::ptr target_needs(new TargetNeeds(step, notifier));

        Heightmap::BlockLayout block_layout(10,10,1);
        Heightmap::TfrMapping::ptr tfrmapping(new Heightmap::TfrMapping(block_layout,1));
        float t_center = 10;
        HeightmapProcessingPublisher hpp(target_needs, tfrmapping, &t_center);

        Heightmap::Collection::ptr collection = tfrmapping->collections()[0];

//        OperationDesc::Extent x;
//        x.interval = Interval(-10,20);
        tfrmapping->length(30);
        hpp.setLastUpdatedInterval (Signal::Interval(0,5));

        EXCEPTION_ASSERT(hpp.isHeightmapDone ());

        hpp.update();

        EXCEPTION_ASSERT(hpp.isHeightmapDone ());

        Heightmap::Reference entireHeightmap = collection.read ()->entireHeightmap();
        unsigned frame_number = collection.read ()->frame_number();

        collection->getBlock(entireHeightmap)->frame_number_last_used = frame_number - 2;
        hpp.update();

        EXCEPTION_ASSERT(hpp.isHeightmapDone ());

        collection->getBlock(entireHeightmap)->frame_number_last_used = frame_number;
        hpp.update();

        EXCEPTION_ASSERT(!hpp.isHeightmapDone ());

        Task task(step.write (),
                  Step::ptr (),
                  std::vector<Step::const_ptr>(),
                  Operation::ptr(),
                  Interval(0,2), Interval());

        EXCEPTION_ASSERT(!hpp.isHeightmapDone ());
    }
}

} // namespace Support
} // namespace Tools
