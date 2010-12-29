#include "brushcontroller.h"

#include "ui/comboboxaction.h"
#include "ui/mainwindow.h"
#include "renderview.h"
#include "ui_mainwindow.h"

#include "heightmap/renderer.h"

#include <QMouseEvent>

namespace Tools {

BrushController::
        BrushController( BrushView* view, RenderView* render_view )
            :
            view_(view),
            render_view_(render_view),
            paint_button_(Qt::LeftButton)
{
    setupGui();

    setAttribute(Qt::WA_DontShowOnScreen, true);
    setEnabled( false );
}


BrushController::
        ~BrushController()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void BrushController::
        setupGui()
{
    Ui::SaweMainWindow* main = render_view_->model->project()->mainWindow();
    Ui::MainWindow* ui = main->getItems();

    connect(ui->actionAmplitudeBrush, SIGNAL(toggled(bool)), SLOT(receiveToggleBrush(bool)));
    connect(ui->actionAirbrush, SIGNAL(toggled(bool)), SLOT(receiveToggleBrush(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionAmplitudeBrush, SLOT(setChecked(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionAirbrush, SLOT(setChecked(bool)));

    connect(render_view_, SIGNAL(painting()), view_, SLOT(draw()));
    connect(render_view_, SIGNAL(destroying()), SLOT(close()));
    connect(render_view_, SIGNAL(destroying()), SLOT(destroying()));


    QToolBar* toolBarTool = new QToolBar(main);
    toolBarTool->setObjectName(QString::fromUtf8("toolBarTool"));
    toolBarTool->setEnabled(true);
    toolBarTool->setContextMenuPolicy(Qt::NoContextMenu);
    toolBarTool->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::TopToolBarArea, toolBarTool);

    {   Ui::ComboBoxAction * qb = new Ui::ComboBoxAction();
        qb->addActionItem( ui->actionAmplitudeBrush );
        qb->addActionItem( ui->actionAirbrush );
        qb->addActionItem( ui->actionSmoothBrush );

        ui->actionAmplitudeBrush->setEnabled( true );
        ui->actionAirbrush->setEnabled( true );
        ui->actionSmoothBrush->setEnabled( false );

        toolBarTool->addWidget( qb );
    }
}


void BrushController::
        receiveToggleBrush(bool /*active*/)
{
    Ui::SaweMainWindow* main = render_view_->model->project()->mainWindow();
    Ui::MainWindow* ui = main->getItems();

    model()->brush_factor = 0;
#ifdef _MSC_VER
    float A = .5f;
#else
    float A = .5f;
#endif
    if (ui->actionAirbrush->isChecked())
        model()->brush_factor = -A;
    if (ui->actionAmplitudeBrush->isChecked())
        model()->brush_factor = A;

    render_view_->toolSelector()->setCurrentTool( this, model()->brush_factor != 0 );
}


void BrushController::
        destroying()
{
    model()->filter()->images.reset();
}


void BrushController::
        mousePressEvent ( QMouseEvent * e )
{
    mouseMoveEvent( e );
}


void BrushController::
        mouseReleaseEvent ( QMouseEvent * )
{
    render_view_->model->project()->worker.invalidate_post_sink( drawn_interval_ );
    drawn_interval_.clear();

    render_view_->userinput_update();
}


void BrushController::
        mouseMoveEvent ( QMouseEvent * e )
{
    if (e->buttons().testFlag( paint_button_ ) || e->buttons().testFlag( Qt::RightButton ))
    {
        Tools::RenderView &r = *render_view_;
        r.makeCurrent();

        float org_factor = model()->brush_factor;

        if (e->buttons().testFlag( Qt::RightButton ))
            model()->brush_factor *= -1;

        Heightmap::Position p = r.getHeightmapPos( QPointF( e->x(), height() - 1 - e->y() ) );
        Heightmap::Reference ref = r.findRefAtCurrentZoomLevel( p );
        if (ref.containsPoint(p))
        {
            // TODO Paint with a lower resolution
            if (0)
            {
                ref = ref.parent().parent().parent();

                Heightmap::Position a,b;
                ref.getArea(a,b);
                while(b.scale>1)
                {
                    ref = ref.bottom();
                    ref.getArea(a,b);
                }
                while(b.time > 2*r.last_length())
                {
                    ref = ref.left();
                    ref.getArea(a,b);
                }
            }

            drawn_interval_ |= model()->paint( ref, p );
        }

        model()->brush_factor = org_factor;
    }

    render_view_->userinput_update();
}


void BrushController::
        changeEvent ( QEvent * event )
{
    if (event->type() & QEvent::EnabledChange)
    {
        view_->enabled = isEnabled();

        if (false == view_->enabled)
            emit enabledChanged(view_->enabled);
    }
}


} // namespace Tools
