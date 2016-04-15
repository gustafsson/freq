#include "heightmapprocessingpublisher.h"
#include "heightmap/collection.h"
#include "signal/processing/step.h"
#include "signal/processing/purge.h"

#include "tasktimer.h"
#include "log.h"
#include "largememorypool.h"
#include "GlException.h"

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
          double prio,
          QObject* parent)
    :
      QObject(parent),
      target_needs_(target_marker->target_needs ()),
      dag_(target_marker->dag ()),
      tfrmapping_(tfrmapping),
      camera_(camera),
      prio_(prio),
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

    Intervals missing_data;
    Intervals needed_samples;
    float fs;
    IntervalType Ls;
    Heightmap::TfrMapping::Collections C;
    {
        auto tm = tfrmapping_.read ();
        fs = tm->targetSampleRate();
        Ls = tm->lengthSamples();
        C = tm->collections();
    }

    float t_center = camera_.read ()->q[0];
    IntervalType center = std::round(t_center * fs);

    Signal::Intervals recently_created;
    for ( auto cp : C )
    {
        auto c = cp.write ();
        missing_data |= c->missing_data();
        recently_created |= c->recently_created();
        needed_samples |= c->needed_samples();
    }

    // new blocks based on invalid data contain invalid data
    auto missing_data_org = missing_data;
    missing_data |= ~last_valid_ & recently_created;

    TIME_PAINTGL_DETAILS if (missing_data || recently_created)
        Log("target_needs_->deprecateCache: %s\nmissing_data_org: %s, recently_created: %s\nlast_valid_: %s, needed_samples: %s")
                % missing_data % missing_data_org % recently_created % last_valid_ % needed_samples;

    target_needs_->deprecateCache (missing_data);

    Interval target_interval(0, Ls);
    needed_samples &= target_interval;

    last_valid_ = needed_samples;
    if (auto step = target_needs_->step ().lock ())
    {
        auto I = Step::cache (step).read ()->samplesDesc();
        last_valid_ &= I;
    }


    Intervals to_compute;
    if (auto step = target_needs_->step ().lock ())
        to_compute = step.read ()->not_started();
    to_compute |= missing_data;
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
    for ( auto c : C ) for ( auto a : Heightmap::Collection::cache (c)->clone () )
    {
        const Heightmap::Block& b = *a.second;
        Interval i = b.getInterval();

        // If this block overlaps data to be computed and is a currently visible block
        if (i & to_compute && b.frame_number_last_used == c.read()->frame_number())
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
            % missing_data
            % center
            % update_size);

    target_needs_->updateNeeds(
                needed_samples,
                center,
                update_size,
                prio_
            );

    failed_allocation_ = false;
    for ( auto c : C )
        failed_allocation_ |= c.write ()->failed_allocation ();

    TIME_PAINTGL_DETAILS {
        if (Step::ptr step = target_needs_->step ().lock())
        {
            Intervals not_started = target_needs_->not_started();
            auto out_of_date = ~Step::cache (step).read ()->samplesDesc();
            auto stepp = step.read ();
            TaskInfo(boost::format("RenderView step->out_of_date = %s, step->not_started = %s, target_needs->not_started = %s")
                             % out_of_date
                             % stepp->not_started()
                             % not_started);
        }
    }

    size_t purged = Purge(dag_).purge (target_needs_,false);
    if (0 < purged)
    {
        purged *= sizeof(Signal::TimeSeriesData::element_type);
        size_t sz = Purge(dag_).cache_size ();

        LOG_PURGED_CACHES Log("Purged %s from the %s cache")
                % DataStorageVoid::getMemorySizeText (purged)
                % DataStorageVoid::getMemorySizeText (purged + sz);
    }

    if (target_needs_->out_of_date().empty ())
    {
        // release all unused memory when left idle for a whole minute
        if (!aggressive_purge_timer_) {
            aggressive_purge_timer_ = startTimer(60000); // 60 seconds
        }
    }
    else
    {
        if (aggressive_purge_timer_) {
            killTimer (aggressive_purge_timer_);
            aggressive_purge_timer_ = 0;
        }
    }
}


void HeightmapProcessingPublisher::
        timerEvent (QTimerEvent *e)
{
    EXCEPTION_ASSERT_EQUALS(e->timerId (), aggressive_purge_timer_);

    killTimer (aggressive_purge_timer_); // only run once
    aggressive_purge_timer_ = 0;

    aggressivePurge();
}


void HeightmapProcessingPublisher::
        aggressivePurge()
{
    Log ("HeightmapProcessingPublisher: Aggressive purge");

    size_t purged = Purge(dag_).purge (target_needs_, true);
    if (0 < purged)
    {
        purged *= sizeof(Signal::TimeSeriesData::element_type);
        size_t sz = Purge(dag_).cache_size ();

        Log("HeightmapProcessingPublisher: Aggressively purged %s from the %s cache")
                % DataStorageVoid::getMemorySizeText (purged)
                % DataStorageVoid::getMemorySizeText (purged + sz);
    }

    auto C = tfrmapping_->collections();
    for ( auto cp : C )
        cp->runGarbageCollection(true);

    lmp_gc (true);
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
#include "glstate.h"
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
#ifndef LEGACY_OPENGL
    QGLFormat f = QGLFormat::defaultFormat ();
    f.setProfile( QGLFormat::CoreProfile );
    f.setVersion( 3, 2 );
    QGLFormat::setDefaultFormat (f);
#endif
    QGLWidget w;
    w.makeCurrent ();
#if !defined(LEGACY_OPENGL) && !defined(_WIN32)
    GLuint VertexArrayID;
    GlException_SAFE_CALL( glGenVertexArrays(1, &VertexArrayID) );
    GlException_SAFE_CALL( glBindVertexArray(VertexArrayID) );
#endif
    GlState::assume_default_gl_states ();

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
        HeightmapProcessingPublisher hpp(target_marker, tfrmapping, camera, 0);

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
