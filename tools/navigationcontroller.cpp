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
        if (e->buttons().testFlag(Qt::LeftButton))
            moveButton.press( e->x(), e->y() );

        if (e->buttons().testFlag(Qt::RightButton))
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
    if ( !e->buttons().testFlag(Qt::LeftButton))
        moveButton.release();

    if ( !e->buttons().testFlag(Qt::RightButton))
    {
        scaleButton.release();
        rotateButton.release();
    }

    _view->userinput_update();
}


void NavigationController::
        wheelEvent ( QWheelEvent *e )
{
    static bool canScrollHorizontal = false;
    if( e->orientation() == Qt::Horizontal )
        canScrollHorizontal = true;

    if (!canScrollHorizontal)
    {
        if (e->modifiers().testFlag(Qt::ControlModifier))
            zoom( e->delta(), ScaleX );
        else if (e->modifiers().testFlag(Qt::AltModifier))
            zoom( e->delta(), ScaleZ );
        else
            zoom( e->delta(), Zoom );
    }
    else if (e->modifiers().testFlag(Qt::ControlModifier))
    {
        if( e->orientation() == Qt::Horizontal )
            zoom( e->delta(), ScaleZ );
        else
            zoom( e->delta(), ScaleX );
    }
    else if (e->modifiers().testFlag(Qt::AltModifier))
    {
        zoom( e->delta(), Zoom );
    }
    else
    {
        bool success1, success2;

        float s = -0.125f;
        QPointF prev = e->pos();
        if( e->orientation() == Qt::Horizontal )
            prev.setX( prev.x() + s*e->delta() );
        else
            prev.setY( prev.y() + s*e->delta() );

        Heightmap::Position last = _view->getPlanePos( prev, &success1);
        Heightmap::Position current = _view->getPlanePos( e->pos(), &success2);
        if (success1 && success2)
        {
            moveCamera( last.time - current.time, last.scale - current.scale);
        }
    }

    _view->userinput_update();
}


bool NavigationController::
        zoom(int delta, ZoomMode mode)
{
    Tools::RenderView &r = *_view;
    float L = r.last_length();
    float fs = r.model->project()->head->head_source()->sample_rate();
    float min_xscale = 4.f/std::max(L,10/fs);
    float max_xscale = 0.05f*fs;


    const Tfr::FreqAxis& tfa = r.model->collections[0]->transform()->freqAxis(fs);
    unsigned maxi = tfa.getFrequencyScalar(fs/2);

    float hza = tfa.getFrequency(0u);
    float hza2 = tfa.getFrequency(1u);
    float hzb = tfa.getFrequency(maxi - 1);
    float hzb2 = tfa.getFrequency(maxi - 2);

    const Tfr::FreqAxis& ds = r.model->display_scale();
    float scalara = ds.getFrequencyScalar( hza );
    float scalara2 = ds.getFrequencyScalar( hza2 );
    float scalarb = ds.getFrequencyScalar( hzb );
    float scalarb2 = ds.getFrequencyScalar( hzb2 );

    float minydist = std::min(fabsf(scalara2 - scalara), fabsf(scalarb2 - scalarb));

    float min_yscale = 4.f;
    float max_yscale = 1.f/minydist;

    if (delta > 0)
    {
        switch(mode)
        {
        case ScaleX: if (r.model->xscale == min_xscale)
                return false;
            break;
        case ScaleZ: if (r.model->zscale == min_yscale)
                return false;
            break;
        default:
            break;
        }
    }

    switch(mode)
    {
    case Zoom: doZoom(delta); break;
    case ScaleX: doZoom( delta, &r.model->xscale, &min_xscale, &max_xscale); break;
    case ScaleZ: doZoom( delta, &r.model->zscale, &min_yscale, &max_yscale ); break;
    }

    return true;
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
    // update currently not pressed mouse buttons
    mouseReleaseEvent(e);

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
            bool success1, success2;
            Heightmap::Position last = r.getPlanePos( QPointF(rotateButton.getLastx(), rotateButton.getLasty()), &success1);
            Heightmap::Position current = r.getPlanePos( e->posF(), &success2);
            if (success1 && success2)
            {
                if (!zoom( 1500*(last.time - current.time)*r.model->xscale, ScaleX ))
                {
                    float L = _view->model->project()->worker.source()->length();
                    float d = std::min( 0.5f * fabsf(last.time - current.time), fabsf(r.model->_qx - L/2));
                    r.model->_qx += r.model->_qx>L*.5f ? -d : d;
                }

                if (!zoom( 1500*(last.scale - current.scale)*r.model->zscale, ScaleZ ))
                {
                    float d = std::min( 0.5f * fabsf(last.scale - current.scale), fabsf(r.model->_qz - .5f));
                    r.model->_qz += r.model->_qz>.5f ? -d : d;
                }
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
