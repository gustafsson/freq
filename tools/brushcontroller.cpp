#include "brushcontroller.h"

#include "ui/comboboxaction.h"

namespace Tools {

BrushController::
        BrushController( BrushView* view, RenderView* render_view )
            :
            view_(view),
            render_view_(render_view)
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
        receiveToggleBrush(bool active)
{

}


void BrushController::
        mousePressEvent ( QMouseEvent * e )
{
    e
}


void BrushController::
        mouseReleaseEvent ( QMouseEvent * e )
{

}


void BrushController::
        mouseMoveEvent ( QMouseEvent * e )
{

}


void BrushController::
        changeEvent ( QEvent * event )
{

}


void BrushController::
        setupGui()
{
    {   ComboBoxAction * qb = new ComboBoxAction();
        qb->addActionItem( ui->actionAmplitudeBrush );
        qb->addActionItem( ui->actionAirbrush );
        qb->addActionItem( ui->actionSmoothBrush );
        qb->setEnabled( false );
        ui->toolBarTool->addWidget( qb );
    }

//    Ui::MainWindow* ui = _render_view->model->project()->mainWindow()->getItems();
//    connect(ui->actionActivateSelection, SIGNAL(toggled(bool)), SLOT(receiveToggleSelection(bool)));
//    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateSelection, SLOT(setChecked(bool)));

//    connect(_render_view, SIGNAL(painting()), _view, SLOT(draw()));
//    connect(_render_view, SIGNAL(destroying()), SLOT(close()));

//    connect(ui->actionActionAdd_selection, SIGNAL(triggered(bool)), SLOT(receiveAddSelection(bool)));
//    connect(ui->actionActionRemove_selection, SIGNAL(triggered(bool)), SLOT(receiveAddClearSelection(bool)));
//    connect(ui->actionCropSelection, SIGNAL(triggered()), SLOT(receiveCropSelection()));
//    connect(ui->actionMoveSelection, SIGNAL(triggered(bool)), SLOT(receiveMoveSelection(bool)));
//    connect(ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), SLOT(receiveMoveSelectionInTime(bool)));

//    /*ui->actionToolSelect->setEnabled( true );
//    ui->actionActivateSelection->setEnabled( true );
//    ui->actionSquareSelection->setEnabled( true );
//    ui->actionSplineSelection->setEnabled( true );
//    ui->actionPolygonSelection->setEnabled( true );
//    ui->actionPeakSelection->setEnabled( true );*/
//    // ui->actionPeakSelection->setChecked( false );

//    Ui::SaweMainWindow* main = model()->project->mainWindow();
//    QToolBar* toolBarTool = new QToolBar(main);
//    toolBarTool->setObjectName(QString::fromUtf8("toolBarTool"));
//    toolBarTool->setEnabled(true);
//    toolBarTool->setContextMenuPolicy(Qt::NoContextMenu);
//    toolBarTool->setToolButtonStyle(Qt::ToolButtonIconOnly);
//    main->addToolBar(Qt::TopToolBarArea, toolBarTool);

//    {   Ui::ComboBoxAction * qb = new Ui::ComboBoxAction();
//        Ui::MainWindow* ui = main->getItems();
//        qb->addActionItem( ui->actionActivateSelection );
//        qb->addActionItem( ui->actionSquareSelection );
//        qb->addActionItem( ui->actionSplineSelection );
//        qb->addActionItem( ui->actionPolygonSelection );
//        qb->addActionItem( ui->actionPeakSelection );

//        toolBarTool->addWidget( qb );
//    }
}

} // namespace Tools
