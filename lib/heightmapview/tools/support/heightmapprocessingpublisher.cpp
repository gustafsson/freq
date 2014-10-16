#include "heightmapprocessingpublisher.h"
#include "heightmap/collection.h"
#include "signal/processing/step.h"
#include "signal/processing/purge.h"

#include "tasktimer.h"
#include "log.h"

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

//#define LOG_PURGED_CACHES
#define LOG_PURGED_CACHES if(0)

using namespace Signal;
using namespace Processing;

namespace Tools {
namespace Support {

HeightmapProcessingPublisher::HeightmapProcessingPublisher(
          TargetMarker::ptr target_marker,
          Heightmap::TfrMapping::const_ptr tfrmapping,
          shared_state<Tools::Support::RenderCamera> camera,
          QObject* parent)
    :
      QObject(parent),
      target_needs_(target_marker->target_needs ()),
      dag_(target_marker->dag ()),
      tfrmapping_(tfrmapping),
      camera_(camera),
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

    float t_center = camera_.read ()->q[0];
    IntervalType center = std::round(t_center * fs);

    for ( auto cp : C )
    {
        auto c = cp.write ();
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
    UnsignedIntervalType update_size = Signal::Cache::chunkSize/2; // Interval::IntervalType_MAX;
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
                             % ~Step::cache (step)->samplesDesc()
                             % stepp->not_started()
                             % not_started);
        }
    }

    size_t purged = Purge(dag_).purge (target_needs_);
    if (0 < purged)
    {
        purged *= sizeof(Signal::TimeSeriesData::element_type);
        size_t sz = Purge(dag_).cache_size ();

        LOG_PURGED_CACHES Log("Purged %s from the %s cache")
                % DataStorageVoid::getMemorySizeText (purged)
                % DataStorageVoid::getMemorySizeText (purged + sz);
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
#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

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
        Dag::ptr dag(new Dag); dag->appendStep(step);
        TargetMarker::ptr target_marker(new TargetMarker(target_needs, dag));

        Heightmap::BlockLayout block_layout(10,10,1);
        Heightmap::TfrMapping::ptr tfrmapping(new Heightmap::TfrMapping(block_layout,1));
        shared_state<Tools::Support::RenderCamera> camera(new Tools::Support::RenderCamera);
        camera->q[0] = 10;
        HeightmapProcessingPublisher hpp(target_marker, tfrmapping, camera);

        Heightmap::Collection::ptr collection = tfrmapping->collections()[0];

//        OperationDesc::Extent x;
//        x.interval = Interval(-10,20);
        tfrmapping->lengthSamples(30 * block_layout.sample_rate ());
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

        Task task(step,
                  std::vector<Step::const_ptr>(),
                  Operation::ptr(),
                  Interval(0,2), Interval());

        EXCEPTION_ASSERT(!hpp.isHeightmapDone ());
    }
}

} // namespace Support
} // namespace Tools
