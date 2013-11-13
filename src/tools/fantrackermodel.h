#ifndef FANTRACKERMODEL_H
#define FANTRACKERMODEL_H

#include "rendermodel.h"
#include "support/fantrackerfilter.h"

namespace Tools {

class FanTrackerModel
{
public:
    FanTrackerModel(RenderModel*);
    Support::FanTrackerFilter* selected_filter();
    Signal::pOperation filter;

private:
    RenderModel* render_model_;
};

} // namespace Tools

#endif // FANTRACKERMODEL_H
