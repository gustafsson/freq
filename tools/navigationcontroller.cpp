#include "navigationcontroller.h"

// Sonic AWE
#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "heightmap/renderer.h"

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
            _view(view),
            zoom_only_(false)
{
    connectGui();

    setAttribute( Qt::WA_DontShowOnScreen, true );
}


NavigationController::
        ~NavigationController()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void NavigationController::
        receiveToggleNavigation(bool active)
{
    if (active || zoom_only_ == false)
        _view->toolSelector()->setCurrentTool( this, active );
    if (active)
        zoom_only_ = false;
}


void NavigationController::
        receiveToggleZoom(bool active)
{
    if (active || zoom_only_ == true)
        _view->toolSelector()->setCurrentTool( this, active );
    if (active)
        zoom_only_ = true;
}


void NavigationController::
        moveUp()
{
    moveCamera(0, 0.1f/_view->model->zscale);
    _view->userinput_update();
}


void NavigationController::
        moveDown()
{
    moveCamera(0, -0.1f/_view->model->zscale);
    _view->userinput_update();
}


void NavigationController::
        moveLeft()
{
    moveCamera(-0.1f/_view->model->xscale, 0);
    _view->userinput_update();
}


void NavigationController::
        moveRight()
{
    moveCamera(0.1f/_view->model->xscale, 0);
    _view->userinput_update();
}


void NavigationController::
        scaleUp()
{
    zoom( -40, ScaleZ );
    _view->userinput_update();
}


void NavigationController::
        scaleDown()
{
    zoom( 40, ScaleZ );
    _view->userinput_update();
}


void NavigationController::
        scaleLeft()
{
    zoom( 40, ScaleX );
    _view->userinput_update();
}


void NavigationController::
        scaleRight()
{
    zoom( -40, ScaleX );
    _view->userinput_update();
}

void NavigationController::
        mousePressEvent ( QMouseEvent * e )
{
    //TaskTimer("NavigationController mousePressEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();

    /*switch ( e->button() )
    {
        case Qt::LeftButton:
            if(' '==lastKey)
                selectionButton.press( e->x(), e->y() );
            else
                leftButton.press( e->x(), e->y() );
            //printf("LeftButton: Press\n");
            break;

        case Qt::MidButton:
            middleButton.press( e->x(), e->y() );
            //printf("MidButton: Press\n");
            break;

        case Qt::RightButton:
        {
            rightButton.press( e->x(), e->y() );
            //printf("RightButton: Press\n");
        }
            break;

        default:
            break;
    }*/

    if(isEnabled()) {
        if( (e->button() & Qt::LeftButton) == Qt::LeftButton)
            moveButton.press( e->x(), e->y() );

        if( (e->button() & Qt::RightButton) == Qt::RightButton)
            rotateButton.press( e->x(), e->y() );

    }

//    if(leftButton.isDown() && rightButton.isDown())
//        selectionButton.press( e->x(), e->y() );

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
    float rs = 0.08;
    if( e->orientation() == Qt::Horizontal )
        _view->model->_ry -= rs * e->delta();
    else if (e->modifiers().testFlag(Qt::ControlModifier))
        zoom( e->delta(), ScaleX );
    else if (e->modifiers().testFlag(Qt::AltModifier))
        zoom( e->delta(), ScaleZ );
    else
        zoom( e->delta(), Zoom );

    _view->userinput_update();
}


void NavigationController::
        zoom(int delta, ZoomMode mode)
{
    Tools::RenderView &r = *_view;
    float L = r.last_length();
    float min_xscale = 0.01f/L;
    float max_xscale = 0.5*r.model->project()->head->head_source()->sample_rate();

    float min_yscale = FLT_MIN;
    float max_yscale = FLT_MAX;

    switch(mode)
    {
    case Zoom: doZoom(delta); break;
    case ScaleX: doZoom( delta, &r.model->xscale, &min_xscale, &max_xscale); break;
    case ScaleZ: doZoom( delta, &r.model->zscale, &min_yscale, &max_yscale ); break;
    }
}


void NavigationController::
        doZoom(int delta, float* scale, float* min_scale, float* max_scale)
{
    //TaskTimer("NavigationController wheelEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
    Tools::RenderView &r = *_view;
    float ps = 0.0005;
    if(scale)
    {
        float d = ps * delta;
        if (d>0.1)
            d=0.1;
        if (d<-0.1)
            d=-0.1;

        *scale *= (1-d);

        if (d > 0 )
        {
            if (min_scale && *scale<*min_scale)
                *scale = *min_scale;
        }
        else
        {
            if (max_scale && *scale>*max_scale)
                *scale = *max_scale;
        }
    }
    else
    {
        r.model->_pz *= (1+ps * delta);

        if (r.model->_pz<-40) r.model->_pz = -40;
        if (r.model->_pz>-.1) r.model->_pz = -.1;
    }

    _view->userinput_update();
}


void NavigationController::
        mouseMoveEvent ( QMouseEvent * e )
{
    //TaskTimer("NavigationController mouseMoveEvent %s %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
    Tools::RenderView &r = *_view;

    float rs = 0.2;

    int x = e->x(), y = e->y();
//    TaskTimer tt("moving");

    if (scaleButton.isDown()) {
        // TODO scale selection
    }
    if( rotateButton.isDown() ) {
        if (e->modifiers().testFlag(Qt::AltModifier))
        {
            zoom( 10* (rotateButton.deltaX( x ) + rotateButton.deltaY( y )), Zoom );
        }
        else if (zoom_only_ || e->modifiers().testFlag(Qt::ControlModifier))
        {
            if (r.model->renderer->left_handed_axes)
            {
                zoom( -10*rotateButton.deltaX( x ), ScaleX );
                zoom( 30*rotateButton.deltaY( y ), ScaleZ );
            }
            else
            {
                zoom( 30*rotateButton.deltaY( y ), ScaleX );
                zoom( -10*rotateButton.deltaX( x ), ScaleZ );
            }
        }
        else
        {
            //Controlling the rotation with the right button.
            r.model->_ry += (1-_view->orthoview)*rs * rotateButton.deltaX( x );
            r.model->_rx += rs * rotateButton.deltaY( y );
            if (r.model->_rx<10) r.model->_rx=10;
            if (r.model->_rx>90) { r.model->_rx=90; _view->orthoview=1; }
            if (0<_view->orthoview && r.model->_rx<90) { r.model->_rx=90; _view->orthoview=0; }
        }

    }

    if( moveButton.isDown() )
    {
        if (zoom_only_)
        {
            zoom( 10* (moveButton.deltaX( x ) + moveButton.deltaY( y )), Zoom );
        }
        else
        {
            //Controlling the position with the left button.
            bool success1, success2;
            Heightmap::Position last = r.getPlanePos( QPointF(moveButton.getLastx(), moveButton.getLasty()), &success1);
            Heightmap::Position current = r.getPlanePos( e->posF(), &success2);
            if (success1 && success2)
            {
                moveCamera( last.time - current.time, last.scale - current.scale);
            }
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
    {
        if (!isEnabled())
            emit enabledChanged(isEnabled());
        else
        {
            one_action_at_a_time_->defaultAction()->setChecked(0!=parent());
        }
    }
}


void NavigationController::
        connectGui()
{
    Ui::SaweMainWindow* main = _view->model->project()->mainWindow();
    Ui::MainWindow* ui = main->getItems();

    connect(ui->actionToggleNavigationToolBox, SIGNAL(toggled(bool)), ui->toolBarOperation, SLOT(setVisible(bool)));
    connect(ui->toolBarOperation, SIGNAL(visibleChanged(bool)), ui->actionToggleNavigationToolBox, SLOT(setChecked(bool)));


    connect(ui->actionActivateNavigation, SIGNAL(toggled(bool)), this, SLOT(receiveToggleNavigation(bool)));
    connect(ui->actionZoom, SIGNAL(toggled(bool)), this, SLOT(receiveToggleZoom(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateNavigation, SLOT(setChecked(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionZoom, SLOT(setChecked(bool)));

    one_action_at_a_time_.reset( new Ui::ComboBoxAction() );
    one_action_at_a_time_->decheckable( false );
    one_action_at_a_time_->addActionItem( ui->actionActivateNavigation );
    one_action_at_a_time_->addActionItem( ui->actionZoom );

    _view->tool_selector->default_tool = this;

    QList<QKeySequence> shortcuts = ui->actionActivateNavigation->shortcuts();
    shortcuts.push_back( Qt::Key_Escape );
    ui->actionActivateNavigation->setShortcuts( shortcuts );

    ui->actionActivateNavigation->setChecked(true);

    bindKeyToSlot( main, "Up", this, SLOT(moveUp()) );
    bindKeyToSlot( main, "Down", this, SLOT(moveDown()) );
    bindKeyToSlot( main, "Left", this, SLOT(moveLeft()) );
    bindKeyToSlot( main, "Right", this, SLOT(moveRight()) );
    bindKeyToSlot( main, "Shift+Up", this, SLOT(scaleUp()) );
    bindKeyToSlot( main, "Shift+Down", this, SLOT(scaleDown()) );
    bindKeyToSlot( main, "Shift+Left", this, SLOT(scaleLeft()) );
    bindKeyToSlot( main, "Shift+Right", this, SLOT(scaleRight()) );
}


void NavigationController::bindKeyToSlot( QWidget* owner, const char* keySequence, const QObject* receiver, const char* slot )
{
    QAction* a = new QAction(owner);
    a->setShortcut(QString(keySequence));
    QObject::connect(a, SIGNAL(triggered()), receiver, slot);
    owner->addAction( a );
}


void NavigationController::
        moveCamera( float dt, float ds )
{
    float l = _view->model->project()->worker.source()->length();

    Tools::RenderView& r = *_view;
    r.model->_qx += dt;
    r.model->_qz += ds;

    if (r.model->_qx<0) r.model->_qx=0;
    if (r.model->_qz<0) r.model->_qz=0;
    if (r.model->_qz>1) r.model->_qz=1;
    if (r.model->_qx>l) r.model->_qx=l;
}

} // namespace Tools
