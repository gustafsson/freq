#include "fantrackerview.h"
#include "ui/mainwindow.h"
//#include "renderview.h"
#include "support/paintline.h"
#include "support/channelcolors.h"

namespace Tools {

    FanTrackerView::FanTrackerView(FanTrackerModel* model, RenderView* render_view)
{
    render_view_ = render_view;
    model_ = model;
}

void FanTrackerView::
        draw()
{
    Signal::OperationDesc::ptr::read_ptr fp = model_->filter.read ();
    const Support::FanTrackerFilter* f = model_->selected_filter(fp);

    if (f)
    {
    Heightmap::FreqAxis const& fa = render_view_->model->display_scale();
    //float FS = model_->selected_filter()->sample_rate();
    float FS = f->last_fs;

    int N = render_view_->model->tfr_mapping ()->channels();
    const std::vector<tvector<4> > colors = Support::ChannelColors::compute(N);

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
