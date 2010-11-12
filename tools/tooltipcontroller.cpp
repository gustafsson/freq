#include "tooltipcontroller.h"

#include "renderview.h"
#include "rendermodel.h"
#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "tfr/cwt.h"
#include "ui_mainwindow.h"

// Qt
#include <QMouseEvent>
#include <QToolTip>

// Std
#include <sstream>
using namespace std;

namespace Tools
{

TooltipController::
        TooltipController(RenderView *view, RenderModel *model)
            :
            _view(view),
            _model(model)
{
    setupGui();
}


void TooltipController::
        receiveToggleInfoTool(bool active)
{
    _view->toolSelector()->setCurrentTool( this, active );

    setEnabled(active);
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
            infoToolButton.press( e->x(), this->height() - e->y() );
            mouseMoveEvent( e );
        }
    }
}


void TooltipController::
        mouseMoveEvent ( QMouseEvent * e )
{
    Tools::RenderView &r = *_view;

    int x = e->x(), y = this->height() - e->y();

    if (infoToolButton.isDown())
    {
        GLvector current;
        if( infoToolButton.worldPos(x, y, current[0], current[1], r.xscale) )
        {
            Tfr::Cwt& c = Tfr::Cwt::Singleton();
            float FS = _model->project()->worker.source()->sample_rate();
            float t = ((unsigned)(current[0]*FS+.5f))/(float)FS;
            //current[1] = ((unsigned)(current[1]*c.nScales(FS)+.5f))/(float)c.nScales(FS);
            current[1] = ((current[1]*c.nScales(FS)+.5f))/(float)c.nScales(FS);
            float f = c.compute_frequency2( FS, current[1] );
            float std_t = c.morlet_sigma_t( FS, f );
            float std_f = c.morlet_sigma_f( f );

            stringstream ss;
            ss << setiosflags(ios::fixed)
               << "Time: " << setprecision(3) << t << " s" << endl
               << "Frequency: " << setprecision(1) << f << " Hz" << endl
               << "Standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz";
            QToolTip::showText( e->globalPos(), QString::fromLocal8Bit("..."), this ); // Force tooltip to change position even if the text is the same as in previous tooltip
            QToolTip::showText( e->globalPos(), QString::fromLocal8Bit(ss.str().c_str()), this );
        }
    }

    infoToolButton.update(x, y);
}


void TooltipController::
        setupGui()
{
    Ui::MainWindow* ui = _model->project()->mainWindow()->getItems();
    connect(ui->actionActivateInfoTool, SIGNAL(toggled(bool)), SLOT(receiveToggleInfoTool(bool)));
}

} // namespace Tools
