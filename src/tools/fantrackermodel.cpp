#include "fantrackermodel.h"

namespace Tools {

FanTrackerModel::FanTrackerModel(RenderModel* render_model)
{
    render_model_ = render_model;
}

Support::FanTrackerFilter* FanTrackerModel::selected_filter()
{
    return dynamic_cast<Support::FanTrackerFilter*>( filter.get() );
}

} // namespace Tools
