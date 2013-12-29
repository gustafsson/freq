#include "fantrackermodel.h"

namespace Tools {

FanTrackerModel::FanTrackerModel(RenderModel* render_model)
{
    render_model_ = render_model;
}

volatile Support::FanTrackerFilter* FanTrackerModel::selected_filter()
{
    return dynamic_cast<volatile Support::FanTrackerFilter*>( filter.get() );
}

} // namespace Tools
