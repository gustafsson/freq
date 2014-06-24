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
          QObject* parent)
    :
      QObject(parent),
      target_needs_(target_needs),
      tfrmapping_(tfrmapping),
      preferred_update_size_(std::numeric_limits<Signal::UnsignedIntervalType>::max()),
      failed_allocation_(false)
{
}


void HeightmapProcessingPublisher::
        setLastUpdateSize( Signal::UnsignedIntervalType last_update_size )
{
    // _last_update_size must be non-zero to be divisable
    preferred_update_size_ = std::max(1llu, last_update_size);
}


void HeightmapProcessingPublisher::
        update(float t_center)
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

    IntervalType center = t_center * fs;

    // It should update the view in sections equal in size to the smallest
    // visible block if the view isn't currently being invalidated.
    UnsignedIntervalType update_size = preferred_update_size_;

    for ( const Heightmap::Collection::ptr &c : C ) {
        auto wc = c.write ();
        //invalid_samples |= wc->invalid_samples();
        things_to_add |= wc->recently_created();
        needed_samples |= wc->needed_samples(update_size);
    }

    Signal::Interval target_interval(0, L*fs);

    if (needed_samples & target_interval)
        needed_samples &= target_interval;
    else
        needed_samples = needed_samples.fetchInterval (1, center);

    update_size = std::min(update_size, (UnsignedIntervalType)Interval::IntervalType_MAX);

    TIME_PAINTGL_DETAILS TaskInfo(boost::format(
            "RenderView needed_samples = %s, "
            "things_to_add = %s, center = %d, size = %d")
            % needed_samples
            % things_to_add
            % center
            % update_size);

    target_needs_->deprecateCache (things_to_add);
    target_needs_->updateNeeds(
                needed_samples,
                center,
                update_size,
                0
            );

    failed_allocation_ = false;
    foreach( const Heightmap::Collection::ptr &c, tfrmapping_->collections() )
    {
        failed_allocation_ |= c.write ()->failed_allocation ();
    }

    TIME_PAINTGL_DETAILS {
        if (Step::ptr step = target_needs_->step ().lock())
        {
            Signal::Intervals not_started = target_needs_->not_started();
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
        HeightmapProcessingPublisher hpp(target_needs, tfrmapping);

        Heightmap::Collection::ptr collection = tfrmapping->collections()[0];

        float t_center = 10;
//        OperationDesc::Extent x;
//        x.interval = Interval(-10,20);
        tfrmapping->length(30);
        UnsignedIntervalType preferred_update_size = 5;
        hpp.setLastUpdateSize (preferred_update_size);

        EXCEPTION_ASSERT(hpp.isHeightmapDone ());

        hpp.update(t_center);

        EXCEPTION_ASSERT(hpp.isHeightmapDone ());

        Heightmap::Reference entireHeightmap = collection.read ()->entireHeightmap();
        unsigned frame_number = collection.read ()->frame_number();

        collection->getBlock(entireHeightmap)->frame_number_last_used = frame_number - 2;
        hpp.update(t_center);

        EXCEPTION_ASSERT(hpp.isHeightmapDone ());

        collection->getBlock(entireHeightmap)->frame_number_last_used = frame_number;
        hpp.update(t_center);

        EXCEPTION_ASSERT(!hpp.isHeightmapDone ());

        Task task(step.write (),
                  Step::ptr (),
                  std::vector<Step::const_ptr>(),
                  Operation::ptr(),
                  Signal::Interval(0,2), Signal::Interval());

        EXCEPTION_ASSERT(!hpp.isHeightmapDone ());
    }
}

} // namespace Support
} // namespace Tools
