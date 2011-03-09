#include "fantrackerview.h"
#include "ui/mainwindow.h"
//#include "renderview.h"
#include "support/paintline.h"

namespace Tools {

    FanTrackerView::FanTrackerView(FanTrackerModel* model, RenderView* render_view)
{
    render_view_ = render_view;
    model_ = model;
}

void FanTrackerView::
        draw()
{

    if (( model_->selected_filter() ))
    {

    Tfr::FreqAxis const& fa = render_view_->model->display_scale();
    float FS = model_->selected_filter()->sample_rate();

    Support::FanTrackerFilter::PointsT map_ = (model_->selected_filter()->track);

    std::vector<Heightmap::Position> pts;

    pts.resize(map_.size());

    TaskTimer tt("Fantracker - number of points %f", (float)pts.size());

    unsigned i = 0;
    foreach(  const Support::FanTrackerFilter::PointsT::value_type& a, map_ )
        {
            float time = a.first/FS;
            float hz = a.second.Hz;
            pts[i++] = Heightmap::Position( time, fa.getFrequencyScalar( hz ));
        }
    Support::PaintLine::drawSlice( pts.size(), &pts[0], 0, 0, 0);
    }
}

} // namespace Tools
