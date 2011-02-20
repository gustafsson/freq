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
#include "signal/operationcache.h"

#include <boost/foreach.hpp>

namespace Tools
{
    class TreeItem: public QTreeWidgetItem
    {
    public:
        TreeItem(QTreeWidgetItem*itm, Signal::pOperation operation, Signal::pChain chain)
            :
            QTreeWidgetItem(itm),
            operation(operation),
            chain(chain)
        {

        }

        Signal::pOperation operation;
        Signal::pChain chain;
    };


    GraphController::
            GraphController( RenderView* render_view )
                :
                render_view_(render_view),
                project_(render_view->model->project())
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
        operationsTree->clear();

        QFlags<Qt::ItemFlag> flg = Qt::ItemIsUserCheckable |
                                   Qt::ItemIsSelectable |
                                   Qt::ItemIsEnabled;

        BOOST_FOREACH( Signal::pChain c, project_->layers.layers() )
        {
            QTreeWidgetItem* chainItm = new QTreeWidgetItem(operationsTree);
            chainItm->setText(0, c->name);
            chainItm->setExpanded( true );

            Signal::pOperation o = c->tip_source();
            while(o)
            {
                TreeItem* itm = new TreeItem(chainItm, o, c);
                if (o == project_->head->head_source())
                    itm->setSelected( true );

                if (dynamic_cast<Signal::OperationCacheLayer*>(o.get()))
                {
                    if (o->source())
                        o = o->source();
                }
                QString name = QString::fromStdString( o->name() );
                itm->setText(0, name);
                //itm->setFlags( flg );
                //itm->setCheckState(0, Qt::Unchecked);
                //itm->setCheckState(0, Qt::Checked);

                o = o->source();
            }
        }
    }


    void GraphController::
            currentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous)
    {
        TreeItem* currentItem = dynamic_cast<TreeItem*>(current);
        TreeItem* previousItem = dynamic_cast<TreeItem*>(previous);
        if ( !currentItem )
        {
            if (previousItem)
            {
                foreach (QTreeWidgetItem* itm, operationsTree->selectedItems() )
                    itm->setSelected( false );

                operationsTree->setCurrentItem( previous );
            }
        }
        else
        {
            // head_source( pOperation ) invalidates models where approperiate
            Signal::pChainHead head = project_->tools().render_model.renderSignalTarget->findHead( currentItem->chain );
            head->head_source( currentItem->operation );

            head = project_->tools().playback_model.playbackTarget->findHead( currentItem->chain );
            head->head_source( currentItem->operation );

            project_->head->head_source( currentItem->operation );
        }
    }


    void GraphController::
            setupGui()
    {
        Ui::SaweMainWindow* MainWindow = project_->mainWindow();
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
        operationsTree->setSelectionMode( QAbstractItemView::SingleSelection );
        //operationsTree->header()->setDefaultSectionSize(60);
        //operationsTree->header()->setMinimumSectionSize(20);
        //operationsTree->setSelectionMode( QAbstractItemView::MultiSelection );

        verticalLayout->addWidget(operationsTree);

        operationsWindow->setWidget(dockWidgetContents);
        MainWindow->addDockWidget( Qt::RightDockWidgetArea, operationsWindow );
        operationsWindow->hide();

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
        //connect(operationsTree, SIGNAL(itemSelectionChanged()), SLOT(selectionChanged()));


        BOOST_FOREACH( Signal::pChain c, project_->layers.layers() )
        {
            connect( c.get(), SIGNAL(chainChanged()), SLOT(redraw_operation_tree()));
        }

        connect( project_->head.get(), SIGNAL(headChanged()), SLOT(redraw_operation_tree()));


        redraw_operation_tree();
    }


    void GraphController::
            checkVisibilityOperations(bool visible)
    {
        Ui::SaweMainWindow* MainWindow = project_->mainWindow();
        visible |= !MainWindow->tabifiedDockWidgets( operationsWindow ).empty();
        visible |= operationsWindow->isVisibleTo( operationsWindow->parentWidget() );
        actionToggleOperationsWindow->setChecked(visible);
    }
}
