#include "graphcontroller.h"

// connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateLayerList(Signal::pOperation)));
// connect(d, SIGNAL(operationsUpdated(Signal::pOperation)), this, SLOT(updateOperationsTree(Signal::pOperation)));
//connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(receiveCurrentSelection(int, bool)));
//connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(receiveFilterRemoval(int)));

// updateOperationsTree( project->worker.source() );

#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/support/operation-composite.h"

namespace Tools
{
    class TreeItem: public QTreeWidgetItem
    {
    public:
        TreeItem(QWidget* parent, Signal::pOperation operation)
            :
            QTreeWidgetItem( parent ),
            operation(operation)
        {

        }

        Signal::pOperation operation;
    };

    GraphController::
            GraphController( RenderView* render_view )
                :
                render_view_(render_view),
                project_(&render_view->model->project())
    {
        setupGui();
    }


    GraphController::
            ~GraphController()
    {
        TaskTimer(__FUNCTION__).suppressTiming();
    }


    void GraphController::
            redraw_operation_tree()
    {
        Ui::SaweMainWindow* main = render_view_->model->project()->mainWindow();
        Ui::MainWindow* ui = main->getItems();

        Signal::pOperation o = worker_->source();
        ui->operationsTree->clear();
        while(o)
        {
            QString name = QString::fromStdString( subop->name() );

            TreeItem* itm = new TreeItem(ui->operationsTree, o);
            itm->setText(0, name);

            o = o->source();
        }
    }


    void GraphController::
            currentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous)
    {
        TreeItem* currentItem = dynamic_cast<TreeItem*>(current);
        BOOST_ASSERT( currentItem );
        worker_->source( currentItem->operation );
    }


    void GraphController::
            setupGui()
    {
        Ui::SaweMainWindow* MainWindow = render_view_->model->project()->mainWindow();
        actionToggleOperationsWindow = new QAction(MainWindow);
        actionToggleOperationsWindow->setObjectName(QString::fromUtf8("actionToggleOperationsWindow"));
        actionToggleOperationsWindow->setCheckable(true);
        actionToggleOperationsWindow->setChecked(true);
        actionToggleOperationsWindow->setText(QApplication::translate("MainWindow", "Operations", 0, QApplication::UnicodeUTF8));
        actionToggleOperationsWindow->setToolTip(QApplication::translate("MainWindow", "Toggle the operations window", 0, QApplication::UnicodeUTF8));
        actionToggleOperationsWindow->setShortcut(QApplication::translate("MainWindow", "O", 0, QApplication::UnicodeUTF8));


        operationsWindow = new QDockWidget(MainWindow);
        operationsWindow->setObjectName(QString::fromUtf8("operationsWindow"));
        operationsWindow->setMinimumSize(QSize(113, 113));
        operationsWindow->setFeatures(QDockWidget::AllDockWidgetFeatures);
        operationsWindow->setAllowedAreas(Qt::AllDockWidgetAreas);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout = new QVBoxLayout(dockWidgetContents);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        operationsTree = new QTreeWidget(dockWidgetContents);
        operationsTree->setObjectName(QString::fromUtf8("operationsTree"));
        operationsTree->setContextMenuPolicy(Qt::NoContextMenu);
        operationsTree->setAutoFillBackground(true);
        operationsTree->setColumnCount(1);
        operationsTree->header()->setVisible(false);
        operationsTree->header()->setDefaultSectionSize(10);
        operationsTree->header()->setMinimumSectionSize(10);

        verticalLayout->addWidget(operationsTree);

        operationsWindow->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), operationsWindow);

        QTreeWidgetItem *headeritem = operationsTree->headerItem();
        headeritem->setText(0, QApplication::translate("MainWindow", "1", 0, QApplication::UnicodeUTF8));
        operationsWindow->setWindowTitle(QApplication::translate("MainWindow", "History", 0, QApplication::UnicodeUTF8));


        MainWindow->getItems()->menu_Windows->addAction(actionToggleOperationsWindow);
        connect(actionToggleOperationsWindow, SIGNAL(toggled(bool)), operationsWindow, SLOT(setVisible(bool)));
        connect(actionToggleOperationsWindow, SIGNAL(triggered()), operationsWindow, SLOT(raise()));
        connect(operationsWindow, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibilityOperations(bool)));
        actionToggleOperationsWindow->setChecked( false );

        connect(operationsTree, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)),
                SLOT(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)));

        connect( worker_, SIGNAL(source_changed()), SLOT(redraw_operation_tree()));
        redraw_operation_tree();
    }

}
