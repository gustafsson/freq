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

namespace Tools
{

NavigationController::
        NavigationController(RenderView* view)
            :
            _view(view)
{
    setupGui();
}


void NavigationController::
        receiveToggleNavigation(bool active)
{
    if (active)
    {
        _view->toolSelector()->setCurrentTool( this );
    }

    setEnabled( active );
}


void NavigationController::
        mousePressEvent ( QMouseEvent * e )
{
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

    _view->update();
}

void NavigationController::
        mouseReleaseEvent ( QMouseEvent * e )
{
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
    _view->update();
}


void NavigationController::
        wheelEvent ( QWheelEvent *e )
{
    Tools::RenderView &r = *_view;
    float ps = 0.0005;
    float rs = 0.08;
    if( e->orientation() == Qt::Horizontal )
    {
        if(e->modifiers().testFlag(Qt::ShiftModifier))
            r.xscale *= (1-ps * e->delta());
        else
            r._ry -= rs * e->delta();
    }
    else
    {
        if(e->modifiers().testFlag(Qt::ShiftModifier))
            r.xscale *= (1-ps * e->delta());
        else
            r._pz *= (1+ps * e->delta());

        if (r._pz<-40) r._pz = -40;
        if (r._pz>-.1) r._pz = -.1;
    }

    _view->update();
}


void NavigationController::
        mouseMoveEvent ( QMouseEvent * e )
{
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
        r._ry += (1-_view->orthoview)*rs * rotateButton.deltaX( x );
        r._rx -= rs * rotateButton.deltaY( y );
        if (r._rx<10) r._rx=10;
        if (r._rx>90) { r._rx=90; _view->orthoview=1; }
        if (0<_view->orthoview && r._rx<90) { r._rx=90; _view->orthoview=0; }

    }

    if( moveButton.isDown() )
    {
        //Controlling the position with the right button.
        GLvector last, current;
        if( moveButton.worldPos(last[0], last[1], r.xscale) &&
            moveButton.worldPos(x, y, current[0], current[1], r.xscale) )
        {
            float l = _view->model->project()->worker.source()->length();

            Tools::RenderView& r = *_view;
            r._qx -= current[0] - last[0];
            r._qz -= current[1] - last[1];

            if (r._qx<0) r._qx=0;
            if (r._qz<0) r._qz=0;
            if (r._qz>1) r._qz=1;
            if (r._qx>l) r._qx=l;
        }
    }



    //Updating the buttons
    moveButton.update(x, y);
    rotateButton.update(x, y);
    scaleButton.update(x, y);

    _view->model->project()->worker.requested_fps(30);
    _view->update();
}


void NavigationController::
        setupGui()
{
    Ui::MainWindow* ui = _view->model->project()->mainWindow()->getItems();
    connect(ui->actionActivateNavigation, SIGNAL(toggled(bool)), SLOT(receiveToggleNavigation(bool)));

    ui->actionActivateNavigation->setChecked(true);
}


} // namespace Tools
