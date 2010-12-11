#include "navigationcontroller.h"

// Sonic AWE
#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"

// Qt
#include <QMouseEvent>

// Todo remove
#include "sawe/project.h"
#include "toolfactory.h"

#include <demangle.h>

namespace Tools
{

NavigationController::
        NavigationController(RenderView* view)
            :
            _view(view)
{
    connectGui();

    setAttribute( Qt::WA_DontShowOnScreen, true );
    setEnabled( false );
}


NavigationController::
        ~NavigationController()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void NavigationController::
        receiveToggleNavigation(bool active)
{
    _view->toolSelector()->setCurrentTool( this, active );
}


void NavigationController::
        mousePressEvent ( QMouseEvent * e )
{
    //TaskTimer("NavigationController mousePressEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();

    /*switch ( e->button() )
    {
        case Qt::LeftButton:
            if(' '==lastKey)
                selectionButton.press( e->x(), this->height() - e->y() );
            else
                leftButton.press( e->x(), this->height() - e->y() );
            //printf("LeftButton: Press\n");
            break;

        case Qt::MidButton:
            middleButton.press( e->x(), this->height() - e->y() );
            //printf("MidButton: Press\n");
            break;

        case Qt::RightButton:
        {
            rightButton.press( e->x(), this->height() - e->y() );
            //printf("RightButton: Press\n");
        }
            break;

        default:
            break;
    }*/

    if(isEnabled()) {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
            moveButton.press( e->x(), this->height() - e->y() );

        if( (e->button() & Qt::RightButton) == Qt::RightButton)
            rotateButton.press( e->x(), this->height() - e->y() );

    }

//    if(leftButton.isDown() && rightButton.isDown())
//        selectionButton.press( e->x(), this->height() - e->y() );

    _view->userinput_update();
}

void NavigationController::
        mouseReleaseEvent ( QMouseEvent * e )
{
    //TaskTimer("NavigationController mouseReleaseEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
    switch ( e->button() )
    {
        case Qt::LeftButton:
            moveButton.release();
            break;

        case Qt::MidButton:
            break;

        case Qt::RightButton:
            scaleButton.release();
            rotateButton.release();
            break;

        default:
            break;
    }
    _view->userinput_update();
}


void NavigationController::
        wheelEvent ( QWheelEvent *e )
{
    //TaskTimer("NavigationController wheelEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
    Tools::RenderView &r = *_view;
    float ps = 0.0005;
    float rs = 0.08;
    if(e->modifiers().testFlag(Qt::ShiftModifier))
    {
        float d = ps * e->delta();
        if (d>0.8)
            d=0.8;
        if (d<-0.8)
            d=-0.8;
        r.model->xscale *= (1-d);

        float max_scale = 0.01*r.model->project()->head_source()->sample_rate();
        float min_scale = 1.f/r.model->project()->head_source()->length();
        if (r.model->xscale>max_scale)
            r.model->xscale=max_scale;
        if (r.model->xscale<min_scale)
            r.model->xscale=min_scale;
    }
    else
    {
        if( e->orientation() == Qt::Horizontal )
        {
            r.model->_ry -= rs * e->delta();
        }
        else
        {
            r.model->_pz *= (1+ps * e->delta());

            if (r.model->_pz<-40) r.model->_pz = -40;
            if (r.model->_pz>-.1) r.model->_pz = -.1;
        }
    }

    _view->userinput_update();
}


void NavigationController::
        mouseMoveEvent ( QMouseEvent * e )
{
    //TaskTimer("NavigationController mouseMoveEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
    Tools::RenderView &r = *_view;
    r.makeCurrent();

    float rs = 0.2;

    int x = e->x(), y = this->height() - e->y();
//    TaskTimer tt("moving");

    if (scaleButton.isDown()) {
        // TODO scale selection
    }
    if( rotateButton.isDown() ){
        //Controlling the rotation with the left button.
        r.model->_ry += (1-_view->orthoview)*rs * rotateButton.deltaX( x );
        r.model->_rx -= rs * rotateButton.deltaY( y );
        if (r.model->_rx<10) r.model->_rx=10;
        if (r.model->_rx>90) { r.model->_rx=90; _view->orthoview=1; }
        if (0<_view->orthoview && r.model->_rx<90) { r.model->_rx=90; _view->orthoview=0; }

    }

    if( moveButton.isDown() )
    {
        //Controlling the position with the right button.
        double last[2], current[2];
        if( moveButton.worldPos(last[0], last[1], r.model->xscale) &&
            moveButton.worldPos(x, y, current[0], current[1], r.model->xscale) )
        {
            float l = _view->model->project()->worker.source()->length();

            Tools::RenderView& r = *_view;
            r.model->_qx -= current[0] - last[0];
            r.model->_qz -= current[1] - last[1];

            if (r.model->_qx<0) r.model->_qx=0;
            if (r.model->_qz<0) r.model->_qz=0;
            if (r.model->_qz>1) r.model->_qz=1;
            if (r.model->_qx>l) r.model->_qx=l;
        }
    }



    //Updating the buttons
    moveButton.update(x, y);
    rotateButton.update(x, y);
    scaleButton.update(x, y);

    _view->userinput_update();
}


void NavigationController::
        changeEvent(QEvent * event)
{
    if (event->type() & QEvent::EnabledChange)
        emit enabledChanged(isEnabled());
}


void NavigationController::
        connectGui()
{
    Ui::MainWindow* ui = _view->model->project()->mainWindow()->getItems();
    connect(ui->actionActivateNavigation, SIGNAL(toggled(bool)), this, SLOT(receiveToggleNavigation(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateNavigation, SLOT(setChecked(bool)));
}


} // namespace Tools
