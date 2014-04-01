#include "fantrackermodel.h"

namespace Tools {

FanTrackerModel::FanTrackerModel(RenderModel* render_model)
{
    render_model_ = render_model;
}

const Support::FanTrackerFilter* FanTrackerModel::
        selected_filter(Signal::OperationDesc::Ptr::read_ptr& w)
{
    return dynamic_cast<const Support::FanTrackerFilter*>( &*w );
}

} // namespace Tools
