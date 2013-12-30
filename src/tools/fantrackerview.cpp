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

    if ( model_->selected_filter() )
    {

    Signal::OperationDesc::ReadPtr fp(model_->filter);
    const Support::FanTrackerFilter* f = dynamic_cast<const Support::FanTrackerFilter*>(&*fp);
    Tfr::FreqAxis const& fa = render_view_->model->display_scale();
    //float FS = model_->selected_filter()->sample_rate();
    float FS = f->last_fs;

    const std::vector<tvector<4> >& colors = render_view_->channelColors();

    for (unsigned C = 0; C < f->track.size (); ++C )
    {
        Support::FanTrackerFilter::PointsT map_ = (f->track[C]);

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

        Support::PaintLine::drawSlice( pts.size(), &pts[0], colors[C][0], colors[C][1], colors[C][2] );
    }
    }
}

} // namespace Tools
