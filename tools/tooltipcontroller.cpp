#include "tooltipcontroller.h"

#include "renderview.h"
#include "rendermodel.h"
#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "tfr/cwt.h"
#include "ui_mainwindow.h"

#include "commentcontroller.h"

// Qt
#include <QMouseEvent>
#include <QToolTip>

// Std
#include <sstream>
using namespace std;

namespace Tools
{

TooltipController::
        TooltipController(RenderView *view, RenderModel *model, CommentController* comments)
            :
            _view(view),
            _model(model),
            _comments(comments),
            _prev_f(-1),
            _current_f(-1),
            _max_so_far(-1),
            comment(0)
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
    _view->toolSelector()->setCurrentTool( this, active );
}


void TooltipController::
        mouseReleaseEvent ( QMouseEvent * e )
{
    if( e->button() == Qt::LeftButton)
    {
        infoToolButton.release();
        _prev_f = _current_f;
        _max_so_far = -1;
        comment = 0;
    }
}


void TooltipController::
        mousePressEvent ( QMouseEvent * e )
{
    if(isEnabled()) {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton) {
            infoToolButton.press( e->x(), this->height() - e->y() );
            mouseMoveEvent( e );
        }
    }
}


void TooltipController::
        mouseMoveEvent ( QMouseEvent * e )
{
    Tools::RenderView &r = *_view;
    r.makeCurrent();

    int x = e->x(), y = this->height() - e->y();

    if (infoToolButton.isDown())
    {
        double current[2];
        if( infoToolButton.worldPos(x, y, current[0], current[1], r.model->xscale) )
        {
            float t = current[0];
            float s = current[1];
            //float FS = _model->project()->worker.source()->sample_rate();
            //t = ((unsigned)(t*FS+.5f))/(float)FS;
            //s = ((unsigned)(s*c.nScales(FS)+.5f))/(float)c.nScales(FS);
            //s = ((s*c.nScales(FS)+.5f))/(float)c.nScales(FS);

            showToolTip(Heightmap::Position(t, s) );
        }
    }

    infoToolButton.update(x, y);
}


void TooltipController::
        showToolTip(Heightmap::Position p )
{
    Tools::RenderView &r = *_view;
    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    float FS = _model->project()->worker.source()->sample_rate();
    float f = c.compute_frequency2( FS, p.scale );
    float std_t = c.morlet_sigma_t( FS, f );
    float std_f = c.morlet_sigma_f( f );
    float val = r.getHeightmapValue( p );

    if (val <= _max_so_far)
        return;

    QPointF screen_pos = r.getScreenPos(p,0);

    _max_so_far = val;
    stringstream ss;
    ss << setiosflags(ios::fixed)
       << "Time: " << setprecision(3) << p.time << " s<br/>" << endl
       << "Frequency: " << setprecision(1) << f << " Hz<br/>" << endl
       << "Standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz<br/>" << endl
       << "Value: " << val;

    if (_prev_f>=0)
       ss << endl << "<br/>Dist to previous: " << setprecision(1) << (f-_prev_f) << " Hz";

    _current_f = f;
    _comments->setComment( p, ss.str(), &comment );
    //QToolTip::showText( screen_pos.toPoint(), QString(), this ); // Force tooltip to change position even if the text is the same as in the previous tooltip
    //QToolTip::showText( screen_pos.toPoint(), QString::fromLocal8Bit(ss.str().c_str()), this );
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
    Ui::MainWindow* ui = _model->project()->mainWindow()->getItems();
    connect(ui->actionActivateInfoTool, SIGNAL(toggled(bool)), SLOT(receiveToggleInfoTool(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateInfoTool, SLOT(setChecked(bool)));
}

} // namespace Tools
