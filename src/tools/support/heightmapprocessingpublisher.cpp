#include "heightmapprocessingpublisher.h"
#include "heightmap/collection.h"
#include "signal/processing/step.h"

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

using namespace Signal;
using namespace Processing;

namespace Tools {
namespace Support {

HeightmapProcessingPublisher::
        HeightmapProcessingPublisher(TargetNeeds::Ptr target_needs,
                      Heightmap::TfrMapping::Collections collections)
    :
      target_needs_(target_needs),
      collections_(collections),
      failed_allocation_(false)
{
}


void HeightmapProcessingPublisher::
        update(float t_center, OperationDesc::Extent x, UnsignedIntervalType preferred_update_size)
{
    TIME_PAINTGL_DETAILS TaskTimer tt("Find things to work on");

    //Intervals invalid_samples;
    Intervals things_to_add;
    Intervals needed_samples;
    IntervalType center = t_center * x.sample_rate.get ();
    UnsignedIntervalType update_size = Interval::IntervalType_MAX;

    foreach( const Heightmap::Collection::Ptr &c, collections_ ) {
        Heightmap::Collection::WritePtr wc(c);
        //invalid_samples |= wc->invalid_samples();
        things_to_add |= wc->recently_created();
        needed_samples |= wc->needed_samples(update_size);
    }

    if (needed_samples & x.interval.get ())
        needed_samples &= x.interval.get ();
    else
        needed_samples = needed_samples.fetchInterval (1, center);

    if (preferred_update_size < update_size)
        update_size = preferred_update_size;

    TIME_PAINTGL_DETAILS TaskInfo(boost::format(
            "RenderView needed_samples = %s, "
            "things_to_add = %s, center = %d, size = %d")
            % needed_samples
            % things_to_add
            % center
            % update_size);

    write1(target_needs_)->updateNeeds(
                needed_samples,
                center,
                update_size,
                things_to_add,
                0
            );

    // It should update the view in sections equal in size to the smallest
    // visible block if the view isn't currently being invalidated.
    if (preferred_update_size < std::numeric_limits<UnsignedIntervalType>::max() / 5 * 4)
    {
        if (preferred_update_size == preferred_update_size * 5 / 4)
            preferred_update_size++;
        preferred_update_size = preferred_update_size * 5 / 4;
    } else {
        preferred_update_size = std::numeric_limits<UnsignedIntervalType>::max();
    }

    failed_allocation_ = false;
    foreach( const Heightmap::Collection::Ptr &c, collections_ )
    {
        failed_allocation_ |= write1(c)->failed_allocation ();
    }

    TIME_PAINTGL_DETAILS {
        Step::Ptr step = read1(target_needs_)->step().lock();

        if (step)
        {
            Step::ReadPtr stepp(step);
            TaskInfo(boost::format("RenderView step out_of_date%s\n"
                               "not_started = %s")
                             % stepp->out_of_date()
                             % stepp->not_started());
        }
    }
}


bool HeightmapProcessingPublisher::
        hasWork() const
{
    TargetNeeds::ReadPtr target_needs(target_needs_);
    return target_needs->out_of_date();
}


bool HeightmapProcessingPublisher::
        isWorking() const
{
    TargetNeeds::ReadPtr target_needs(target_needs_);
    return target_needs->not_started() != target_needs->out_of_date();
}


bool HeightmapProcessingPublisher::
        workerCrashed() const
{
    // TODO this is not a reliable way to detect if the worker has crashed.
    // Right after a call to update this will return true before the worker
    // has noticed the change.
    TargetNeeds::ReadPtr target_needs(target_needs_);
    bool is_working = target_needs->not_started() != target_needs->out_of_date();
    bool has_work = target_needs->out_of_date();
    return !is_working && has_work;
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

namespace Tools {
namespace Support {

void HeightmapProcessingPublisher::
        test()
{
    // It should update a processing target depending on which things that are
    // missing in a heightmap block cache
    {
        OperationDesc::Ptr operation_desc;
        Step::Ptr step(new Step(operation_desc));
        Bedroom::Ptr bedroom(new Bedroom);
        TargetNeeds::Ptr target_needs(new TargetNeeds(step, bedroom));

        Heightmap::BlockLayout block_layout(10,10,1);
        Heightmap::VisualizationParams::Ptr visualization_params(new Heightmap::VisualizationParams);
        Heightmap::Collection::Ptr collection(new Heightmap::Collection(
                                                   block_layout, visualization_params));

        Heightmap::TfrMapping::Collections collections;
        collections.push_back (collection);

        HeightmapProcessingPublisher hpp(target_needs, collections);

        float t_center = 10;
        OperationDesc::Extent x;
        x.interval = Interval(-10,20);
        x.number_of_channels = 1;
        x.sample_rate = 1;
        UnsignedIntervalType preferred_update_size = 5;

        EXCEPTION_ASSERT(!hpp.isWorking ());
        EXCEPTION_ASSERT(!hpp.hasWork ());

        hpp.update(t_center, x, preferred_update_size);

        EXCEPTION_ASSERT(!hpp.isWorking ());

        Heightmap::Reference entireHeightmap = read1(collection)->entireHeightmap();
        write1(collection)->getBlock(entireHeightmap);

        EXCEPTION_ASSERT(!hpp.isWorking ());

        hpp.update(t_center, x, preferred_update_size);

        EXCEPTION_ASSERT(!hpp.isWorking ());

        unsigned frame_number = read1(collection)->frame_number();
        write1(collection)->getBlock(entireHeightmap)->frame_number_last_used = frame_number;

        EXCEPTION_ASSERT(!hpp.isWorking ());
        EXCEPTION_ASSERT(!hpp.hasWork ());

        hpp.update(t_center, x, preferred_update_size);

        EXCEPTION_ASSERT(!hpp.isWorking ());
        EXCEPTION_ASSERT(hpp.hasWork ());

        Task task(&*write1(step), step,
                  std::vector<Signal::Processing::Step::Ptr>(),
                  Signal::Interval(0,2));

        EXCEPTION_ASSERT(hpp.isWorking ());
    }
}

} // namespace Support
} // namespace Tools
