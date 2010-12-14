#include "tooltipcontroller.h"

#include "renderview.h"
#include "rendermodel.h"
#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "tfr/cwt.h"
#include "tfr/cwtfilter.h"
#include "ui_mainwindow.h"

#include "commentcontroller.h"
#include "heightmap/renderer.h"

// Qt
#include <QMouseEvent>
#include <QToolTip>

// Std
#include <sstream>
using namespace std;

namespace Tools
{

TooltipController::
        TooltipController(TooltipView* view,
                          RenderView *render_view,
                          CommentController* comments)
            :
            view_(view),
            render_view_(render_view),
            _comments(comments)
{
    setupGui();
}


TooltipController::
        ~TooltipController()
{

}


void TooltipController::
        receiveToggleInfoTool(bool active)
{
    render_view_->toolSelector()->setCurrentTool( this, active );
}


void TooltipController::
        mouseReleaseEvent ( QMouseEvent * e )
{
    if( e->button() == Qt::LeftButton)
    {
        infoToolButton.release();
    }
}


void TooltipController::
        mousePressEvent ( QMouseEvent * e )
{
    if(isEnabled()) {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton) {
            infoToolButton.press( e->x(), this->height() - 1 - e->y() );

            model()->max_so_far = -1;
            if (model()->comment && model()->comment->isHidden())
                model()->comment = 0;
            model()->automarkers = true;

            mouseMoveEvent( e );
        }
    }
}


void TooltipController::
        mouseMoveEvent ( QMouseEvent * e )
{
    int x = e->x(), y = this->height() - 1 - e->y();

    if (infoToolButton.isDown())
    {
        bool success=false;
        Heightmap::Position p = render_view_->getPlanePos(QPointF(x, y), &success);
        if (success)
        {
            showToolTip( p );
        }

//        double current[2];
//        render_view_->makeCurrent();
//        if( infoToolButton.worldPos(x, y, current[0], current[1], render_view_->model->xscale) )
//        {
//            float t = current[0];
//            float s = current[1];
            //float FS = _model->project()->worker.source()->sample_rate();
            //t = ((unsigned)(t*FS+.5f))/(float)FS;
            //s = ((unsigned)(s*c.nScales(FS)+.5f))/(float)c.nScales(FS);
            //s = ((s*c.nScales(FS)+.5f))/(float)c.nScales(FS);

            //showToolTip(Heightmap::Position(t, s) );
//        }
    }

    infoToolButton.update(x, y);
}


void TooltipController::
        wheelEvent(QWheelEvent *event)
{
    if (0 < event->delta())
        model()->markers++;
    else if(1 < model()->markers)
        model()->markers--;

    if (model()->comment)
    {
        model()->automarkers = false;
        showToolTip(model()->pos );
    }
}


void TooltipController::
        showToolTip(Heightmap::Position p )
{
    render_view_->userinput_update();

    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    float val = render_view_->getHeightmapValue( p ) / render_view_->model->renderer->y_scale;
    bool found_better = val > model()->max_so_far;
    if (found_better)
        model()->pos = p;
    else
        p = model()->pos;

    float FS = render_view_->model->project()->worker.source()->sample_rate();
    float f = c.compute_frequency2( FS, p.scale );
    float std_t = c.morlet_sigma_samples( FS, f ) / FS;
    float std_f = c.morlet_sigma_f( f );

    if (found_better && model()->automarkers)
        model()->markers = guessHarmonicNumber( p );

    if (found_better)
        model()->max_so_far = val;

    stringstream ss;
    ss << setiosflags(ios::fixed)
       << "Time: " << setprecision(3) << p.time << " s<br/>"
       << "Frequency: " << setprecision(1) << f << " Hz<br/>";

    if (dynamic_cast<Tfr::CwtFilter*>(
            dynamic_cast<Signal::PostSink*>(render_view_->model->postsink().get())
            ->sinks()[0]->source().get()))
       ss << "Morlet standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz<br/>";

    ss << "Value here: " << model()->max_so_far;

    if ( 0 < model()->markers )
    {
        ss << endl << "<br/><br/>Harmonic number: " << model()->markers;
        ss << endl << "<br/>Fundamental frequency: " << setprecision(2) << (f/model()->markers) << " Hz";
    }

    //model()->frequency = f;

    bool first = 0 == model()->comment;
    _comments->setComment( model()->pos, ss.str(), &model()->comment );
    if (first)
        model()->comment->thumbnail( true );

    model()->comment->model->pos = Heightmap::Position(
            p.time + 0.05/render_view_->model->xscale,
            p.scale);

    //QToolTip::showText( screen_pos.toPoint(), QString(), this ); // Force tooltip to change position even if the text is the same as in the previous tooltip
    //QToolTip::showText( screen_pos.toPoint(), QString::fromLocal8Bit(ss.str().c_str()), this );

    if ( first )
    {
        model()->comment->resize( 380, 190 );
    }
}


unsigned TooltipController::
        guessHarmonicNumber( const Heightmap::Position& pos )
{
    double max_s = 0;
    unsigned max_i = 0;
    const Tfr::FreqAxis&  display_scale = render_view_->model->display_scale();

    double F = display_scale.getFrequency( pos.scale );
    double F_top = display_scale.getFrequency(1.f);
    double F_min = 8;
    Heightmap::Position p = pos;
    Heightmap::Reference ref(render_view_->model->collections[0].get());
    ref.block_index[0] = (unsigned)-1;

    unsigned fetched_heightmap_values = 0;
    unsigned n_tests = F/F_min;
    for (unsigned i=1; i<n_tests; ++i)
    {
        double k = pow((double)i, -0.99);

        double s = 0;
        double fundamental_f = F/i;
        unsigned count = std::min(F_top, 3*F)/fundamental_f;
        for (unsigned j=1; j<=count; ++j)
        {
            double f = fundamental_f * j;
            p.scale = display_scale.getFrequencyScalar( f );
            float value = render_view_->getHeightmapValue( p, &ref );
            ++fetched_heightmap_values;
            value*=value;

            s += value * k;
        }

        if (max_s < s)
        {
            max_s = s;
            max_i = i;
        }
    }

    //if (max_i<=1)
        //return 0;

    return max_i;
}


void TooltipController::
        changeEvent(QEvent *event)
{
    if (event->type() & QEvent::EnabledChange)
        if (!isEnabled())
            emit enabledChanged(isEnabled());
}


void TooltipController::
        setupGui()
{
    Ui::MainWindow* ui = render_view_->model->project()->mainWindow()->getItems();
    connect(ui->actionActivateInfoTool, SIGNAL(toggled(bool)), SLOT(receiveToggleInfoTool(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateInfoTool, SLOT(setChecked(bool)));

    // Paint the ellipse when render view paints
    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));
    // Close this widget before the OpenGL context is destroyed to perform
    // proper cleanup of resources
    connect(render_view_, SIGNAL(destroying()), SLOT(close()));
}

} // namespace Tools
