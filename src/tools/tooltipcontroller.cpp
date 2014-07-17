#include "tooltipcontroller.h"
#include "tooltipview.h"

#include "tools/support/renderviewinfo.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "tools/support/toolbar.h"

#include "graphicsview.h"

// boost
#include <boost/foreach.hpp>

// Qt
#include <QMouseEvent>
#include <QGLWidget>

namespace Tools
{

TooltipController::Actions::
        Actions(QObject* parent)
{
    actionActivateInfoTool = new QAction(parent);
    actionActivateInfoTool->setObjectName(QStringLiteral("actionActivateInfoTool"));
    actionActivateInfoTool->setCheckable(true);
    actionActivateInfoTool->setChecked(false);
    QIcon icon27;
    icon27.addFile(QStringLiteral(":/icons/icons/icon-show-info.png"), QSize(), QIcon::Normal, QIcon::Off);
    actionActivateInfoTool->setIcon(icon27);
    actionActivateInfoTool->setIconVisibleInMenu(true);

    actionAddComment = new QAction(parent);
    actionAddComment->setObjectName(QStringLiteral("actionAddComment"));
    actionAddComment->setCheckable(true);
    actionAddComment->setChecked(false);
    QIcon icon28;
    icon28.addFile(QStringLiteral(":/icons/icons/icon-place-comment.png"), QSize(), QIcon::Normal, QIcon::Off);
    actionAddComment->setIcon(icon28);
    actionAddComment->setIconVisibleInMenu(true);

    actionShowComments = new QAction(parent);
    actionShowComments->setObjectName(QStringLiteral("actionShowComments"));
    actionShowComments->setCheckable(true);
    QIcon icon29;
    icon29.addFile(QStringLiteral(":/icons/icons/icon-show-comment.png"), QSize(), QIcon::Normal, QIcon::Off);
    actionShowComments->setIcon(icon29);
    actionShowComments->setIconVisibleInMenu(true);

    actionActivateInfoTool->setText(QApplication::translate("MainWindow", "Get info", 0));
    actionActivateInfoTool->setToolTip(QApplication::translate("MainWindow", "Drag slowly over a ridge to see information of the peak. Show continius information [J]", 0));
    actionAddComment->setText(QApplication::translate("MainWindow", "Add comment", 0));
    actionAddComment->setToolTip(QApplication::translate("MainWindow", "Add comment [A]", 0));
    actionShowComments->setText(QApplication::translate("MainWindow", "Hide/show comments", 0));
    actionShowComments->setToolTip(QApplication::translate("MainWindow", "Hide/show comments", 0));
    actionShowComments->setShortcut(QApplication::translate("MainWindow", "Ctrl+H", 0));
}

TooltipController::
        TooltipController(RenderView *render_view,
                          CommentController* comments,
                          Sawe::Project* project,
                          Tools::Support::ToolSelector* tool_selector)
            :
            ui(new Actions(this)),
            render_view_(render_view),
            comments_(comments),
            project_(project),
            tool_selector_(tool_selector),
            current_view_(0)
{
    setEnabled( false );

    setupGui();
}


TooltipController::
        ~TooltipController()
{
    BOOST_FOREACH( QPointer<TooltipView>& tv, views_)
    {
        if (tv.data())
            delete tv.data();
    }
    views_.clear();
}


void TooltipController::
        createView( ToolModelP modelp, ToolRepo* repo, Sawe::Project* p )
{
    TooltipModel* model = dynamic_cast<TooltipModel*>(modelp.get());
    if (!model)
        return;

    model->setPtrs( repo->render_view(), this->comments_, p );

    setCurrentView( new TooltipView( model, this, repo->render_view() ) );
    setupView(current_view_);
}


void TooltipController::
        setCurrentView(TooltipView* value )
{
    if (views_.end() == std::find(views_.begin(), views_.end(), value))
    {
        current_view_ = 0;
        views_.push_back(value);
    }

    if (value != current_view_)
    {
        current_view_ = value;
        emit tooltipChanged();
    }

    if (value == 0)
    {
        tool_selector_->setCurrentTool( this, value != 0 );
        setEnabled( value != 0 );
    }
}


TooltipView* TooltipController::
        current_view()
{
    return current_view_;
}


void TooltipController::
        receiveToggleInfoTool(bool active)
{
    tool_selector_->setCurrentTool( this, active );

    emitTooltipChanged();
}


void TooltipController::
        emitTooltipChanged()
{
    emit tooltipChanged();
    render_view_->redraw();
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
            infoToolButton.press( e->x(), e->y() );

            TooltipModel* model = new TooltipModel();

            model->automarking = TooltipModel::AutoMarkerWorking;

            // calls createView
            project_->tools().addModel( model );

            mouseMoveEvent( e );
        }
    }
}


void TooltipController::
        mouseMoveEvent ( QMouseEvent * e )
{
    if (hover_info_model_)
    {
        bool success = true;
        Heightmap::Position p = Tools::Support::RenderViewInfo(render_view_).getHeightmapPos( e->localPos ());
        //Heightmap::Position p = Tools::Support::RenderViewInfo(render_view_).getPlanePos( e->posF(), &success);
        TaskTimer tt("TooltipController::mouseMoveEvent hover_info_model(%g, %g)", p.time, p.scale);
        if (success)
        {
            hover_info_model_->showToolTip( p, false );
            if (hover_info_model_->comment)
            {
                hover_info_model_->comment->setEnabled( false );
            }
        }
    }

    if (!current_model())
        return;

    if (infoToolButton.isDown())
    {
        bool success = true;
        Tools::Support::RenderViewInfo r(render_view_);
        Heightmap::Position p = r.getHeightmapPos( e->localPos ());
        //Heightmap::Position p = render_view_->getPlanePos( e->posF(), &success);
        TaskTimer tt("TooltipController::mouseMoveEvent (%g, %g)", p.time, p.scale);
        if (success)
        {
            Heightmap::Position o = current_model()->pos();
            bool t = TooltipModel::AutoMarkerWorking == current_model()->automarking;

            current_model()->showToolTip( p );

            t |= current_model()->pos() != o;

            if (t)
                emitTooltipChanged();
        }
    }

    infoToolButton.update(e->x(), e->y());
}


void TooltipController::
        wheelEvent(QWheelEvent *event)
{
    if (!current_model())
        return;

    if (0 < event->delta())
        current_model()->markers++;
    else if(1 < current_model()->markers)
        current_model()->markers--;

    EXCEPTION_ASSERT(current_model()->comment);

    current_model()->automarking = TooltipModel::ManualMarkers;
    current_model()->showToolTip(current_model()->pos() );
    emitTooltipChanged();
    render_view_->redraw();
}


void TooltipController::
        changeEvent(QEvent *event)
{
    if (event->type() == QEvent::EnabledChange)
    {
        if (!isEnabled())
        {
            if (hover_info_action_)
                hover_info_action_->setChecked( false );
            setCurrentView(0);
        }

        emit enabledChanged(isEnabled());
    }
}


void TooltipController::
        hoverInfoToggled(bool enabled)
{
    if (enabled)
    {
        ui->actionActivateInfoTool->setChecked( true );

        hover_info_model_.reset( new TooltipModel() );
        hover_info_model_->automarking = TooltipModel::NoMarkers;
        hover_info_model_->setPtrs( render_view_, this->comments_, project_ );
        setMouseTracking( true );

        QMouseEvent event(QEvent::MouseMove, render_view_->glwidget->mapFromGlobal( QCursor::pos() ),  Qt::NoButton, Qt::NoButton, Qt::NoModifier );
        mouseMoveEvent(&event);

        this->render_view_->graphicsview->setToolFocus( true );
        this->render_view_->redraw();
    }
    else if (hover_info_model_)
    {
        delete hover_info_model_->comment.data();
        hover_info_model_.reset();
        setMouseTracking( false );

        this->render_view_->graphicsview->setToolFocus( false );
    }
}


void TooltipController::
        setupGui()
{
    Ui::SaweMainWindow* mainWindow = project_->mainWindow();

    Tools::Support::ToolBar *toolBarOperation;
    toolBarOperation = new Tools::Support::ToolBar(mainWindow);

    toolBarOperation->setObjectName(QStringLiteral("toolBarOperation"));
    toolBarOperation->setEnabled(true);
    toolBarOperation->setWindowTitle(QApplication::translate("MainWindow", "toolBar", 0));
    mainWindow->addToolBar( Qt::TopToolBarArea, toolBarOperation );

    toolBarOperation->addAction(ui->actionActivateInfoTool);
    toolBarOperation->addAction(ui->actionAddComment);
    toolBarOperation->addAction(ui->actionShowComments);

    connect(ui->actionActivateInfoTool, SIGNAL(toggled(bool)), SLOT(receiveToggleInfoTool(bool)));
    connect(this, SIGNAL(enabledChanged(bool)), ui->actionActivateInfoTool, SLOT(setChecked(bool)));

    hover_info_action_ = new QAction(mainWindow);
    hover_info_action_->setCheckable( true );
    hover_info_action_->setShortcutContext(Qt::WindowShortcut);
    hover_info_action_->setShortcut(QString("j"));
    mainWindow->addAction( hover_info_action_ );
    connect(hover_info_action_.data(), SIGNAL(toggled(bool)), SLOT(hoverInfoToggled(bool)));

    // Close this widget before the OpenGL context is destroyed to perform
    // proper cleanup of resources
    connect(render_view_, SIGNAL(destroying()), SLOT(close()));
}


void TooltipController::
        setupView(TooltipView* view_)
{
    connect(view_, SIGNAL(tooltipChanged()), SLOT(emitTooltipChanged()));
    connect(view_, SIGNAL(destroyed()), SLOT(emitTooltipChanged()));
}


TooltipModel* TooltipController::
        current_model()
{
    if (0==current_view_)
        return 0;
    return current_view_->model();
}


} // namespace Tools
