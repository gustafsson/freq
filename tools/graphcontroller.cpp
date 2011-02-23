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

#include <QPushButton>

namespace Tools
{
    class TreeItem: public QTreeWidgetItem
    {
    public:
        TreeItem(QTreeWidgetItem*itm, Signal::pOperation operation, Signal::pChain chain)
            :
            QTreeWidgetItem(itm),
            operation(operation),
            tail(operation),
            chain(chain)
        {

        }


        Signal::pOperation operation;
        Signal::pOperation tail;
        Signal::pChain chain;
    };


    class TreeWidget: public QTreeWidget
    {
    public:
        TreeWidget(QWidget*parent)
            :
            QTreeWidget(parent)
        {}

        virtual void dropEvent ( QDropEvent * event ) {
            QTreeWidget::dropEvent ( event );

            for (int l = 0; l<invisibleRootItem()->childCount(); ++l)
            {
                QTreeWidgetItem* layer = invisibleRootItem()->child(l);

                // Deselect all
                for (int i=0; i<layer->childCount(); ++i)
                    layer->child(i)->setSelected( false );

                // Make sure the root element is at the bottom
                Signal::pOperation firstmoved;
                for (int i=0; i<layer->childCount(); ++i)
                {
                    TreeItem* itm = dynamic_cast<TreeItem*>(layer->child(i));
                    if (0 == itm->tail->source() && i != layer->childCount()-1)
                    {
                        firstmoved = itm->tail;
                        layer->takeChild( i );
                        layer->addChild( itm );
                        setCurrentItem( itm );
                        break;
                    }
                }

                // rebuild the chain bottom up
                Signal::pOperation src;
                Signal::pChain chain;
                for (unsigned i=layer->childCount(); i>0; i--)
                {
                    TreeItem* itm = dynamic_cast<TreeItem*>(layer->child(i-1));
                    if (src)
                    {
                        if (itm->tail->source() != src)
                        {
                            if (!firstmoved)
                                firstmoved = itm->tail;
                            itm->tail->source( src );
                            setCurrentItem( itm );
                        }
                    }
                    else
                        chain = itm->chain;

                    src = itm->operation;
                }

                // Set the tip
                chain->tip_source( src );
                firstmoved->invalidate_samples(Signal::Interval(0, firstmoved->number_of_samples()));
            }
        }
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

        TaskInfo("project head source: %s", project_->head->head_source()->toString().c_str());
        TaskInfo("project head output: %s", project_->head->head_source()->parentsToString().c_str());

        operationsTree->invisibleRootItem()->setFlags(
                operationsTree->invisibleRootItem()->flags() & ~Qt::ItemIsDropEnabled );

        BOOST_FOREACH( Signal::pChain c, project_->layers.layers() )
        {
            QTreeWidgetItem* chainItm = new QTreeWidgetItem(operationsTree);
            chainItm->setText(0, QString::fromStdString( c->name ) );
            chainItm->setExpanded( true );
            chainItm->setFlags( chainItm->flags() & ~Qt::ItemIsSelectable );

            Signal::pOperation o = c->tip_source();
            while(o)
            {
                TreeItem* itm = new TreeItem(chainItm, o, c);
                if (o == project_->head->head_source())
                    operationsTree->setCurrentItem( itm );

                if (dynamic_cast<Signal::OperationCacheLayer*>(o.get()))
                {
                    if (o->source())
                        o = o->source();
                }
                itm->tail = o;
                itm->setFlags( itm->flags() & ~Qt::ItemIsDropEnabled );
                QString name = QString::fromStdString( o->name() );
                itm->setText(0, name);
                //itm->setCheckState(0, Qt::Unchecked);
                //itm->setCheckState(0, Qt::Checked);

                o = o->source();
            }
        }
    }


    void GraphController::
            currentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous)
    {
        if (!current)
            operationsTree->setContextMenuPolicy(Qt::NoContextMenu);

        if (!previous || !current)
        {
            return;
        }

        TreeItem* currentItem = dynamic_cast<TreeItem*>(current);
        TreeItem* previousItem = dynamic_cast<TreeItem*>(previous);

        if ( !currentItem )
        {
            if (current && current->childCount())
                operationsTree->setCurrentItem( current->child(0) );
            else if (previousItem)
                operationsTree->setCurrentItem( previous );
        }
        else
        {
            operationsTree->setContextMenuPolicy(Qt::ActionsContextMenu);
            // head_source( pOperation ) invalidates models where approperiate
            Signal::pChain chain = currentItem->chain;
            Signal::pOperation operation = currentItem->operation;
            Signal::pChainHead head1 = project_->tools().render_model.renderSignalTarget->findHead( chain );
            Signal::pChainHead head2 = project_->tools().playback_model.playbackTarget->findHead( chain );
            Signal::pChainHead head3 = project_->head;
            head1->head_source( operation );
            head2->head_source( operation );
            head3->head_source( operation );
        }
    }


    void GraphController::
            removeSelected()
    {
        QList<QTreeWidgetItem*> itms = operationsTree->selectedItems();
        if (itms.empty())
            return;

        TreeItem* currentItem = dynamic_cast<TreeItem*>(itms.front());
        if ( !currentItem )
            return;

        Signal::pOperation currentSource;
        // If the current operation is a cache, don't just remove the cache but
        // remove what was cached as well. So jump an extra steps down in source()
        if (dynamic_cast<Signal::OperationCacheLayer*>(currentItem->operation.get()) )
            currentSource = currentItem->operation->source()->source();
        else
            currentSource = currentItem->operation->source();

        if (!currentSource)
            return;

        Signal::pOperation o = Signal::Operation::findParentOfSource( currentItem->chain->tip_source(), currentItem->operation );
        if (o)
        {
            o->invalidate_samples( Signal::Operation::affecetedDiff(o->source(), currentSource ));

            o->source( currentSource );

            // If there is a cache right above this, set the cache as head_source instead
            Signal::pOperation o2 = Signal::Operation::findParentOfSource( currentItem->chain->tip_source(), o );
            if (dynamic_cast<Signal::OperationCacheLayer*>(o2.get()) )
                o = o2;
        }
        else
        {
            o = currentSource;
        }

        Signal::pChainHead head = project_->tools().render_model.renderSignalTarget->findHead( currentItem->chain );
        head->head_source( o );

        head = project_->tools().playback_model.playbackTarget->findHead( currentItem->chain );
        head->head_source( o );

        project_->head->head_source( o );

        redraw_operation_tree();

        if (o==currentSource)
            currentItem->chain->tip_source( o );
    }


    void GraphController::
            removeHidden()
    {
        QList<QTreeWidgetItem*> itms = operationsTree->selectedItems();
        if (itms.empty())
            return;

        TreeItem* currentItem = dynamic_cast<TreeItem*>(itms.front());
        if ( !currentItem )
            return;

        currentItem->chain->tip_source( currentItem->operation );

        redraw_operation_tree();
    }


    void GraphController::
            removeCaches()
    {
        QList<QTreeWidgetItem*> itms = operationsTree->selectedItems();
        if (itms.empty())
            return;

        TreeItem* currentItem = dynamic_cast<TreeItem*>(itms.front());
        if ( !currentItem )
            return;

        currentItem->operation->invalidate_samples(Signal::Interval(0, currentItem->operation->number_of_samples()));

        redraw_operation_tree();
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
        operationsTree = new TreeWidget(dockWidgetContents);
        operationsTree->setDragDropMode( QAbstractItemView::InternalMove );
        operationsTree->setAcceptDrops( true );
        // setDragEnabled( false );
        operationsTree->setAlternatingRowColors( true );
        operationsTree->setObjectName(QString::fromUtf8("operationsTree"));
        operationsTree->setContextMenuPolicy(Qt::NoContextMenu);
        operationsTree->setAutoFillBackground(true);
        operationsTree->setColumnCount(1);
        operationsTree->header()->setVisible(false);
        operationsTree->setSelectionMode( QAbstractItemView::SingleSelection );
        QAction* removeCurrentItem = new QAction("Remove", MainWindow);
        operationsTree->addAction(removeCurrentItem);
        connect(removeCurrentItem, SIGNAL(triggered()), SLOT(removeSelected()));
        //operationsTree->header()->setDefaultSectionSize(60);
        //operationsTree->header()->setMinimumSectionSize(20);
        //operationsTree->setSelectionMode( QAbstractItemView::MultiSelection );

        QWidget* buttons = new QWidget;
        //buttons->setLayout( new QHBoxLayout );
        buttons->setLayout( new QVBoxLayout );
        QPushButton* removeSelectedButton = new QPushButton("Remove selected");
        QPushButton* removeHiddenButton = new QPushButton("Remove hidden");
        QPushButton* removeCachesdButton = new QPushButton("Discard caches");
        removeCachesdButton->setToolTip( "Discard caches for operations above the selected operation" );
        connect(removeSelectedButton, SIGNAL(clicked()), SLOT(removeSelected()));
        connect(removeHiddenButton, SIGNAL(clicked()), SLOT(removeHidden()));
        connect(removeCachesdButton, SIGNAL(clicked()), SLOT(removeCaches()));
        buttons->layout()->addWidget( removeSelectedButton );
        buttons->layout()->addWidget( removeHiddenButton );
        //buttons->layout()->addWidget( removeCachesdButton );

        verticalLayout->addWidget(operationsTree);
        verticalLayout->addWidget(buttons);

        operationsWindow->setWidget(dockWidgetContents);
        MainWindow->addDockWidget( Qt::RightDockWidgetArea, operationsWindow );
        operationsWindow->hide();

        QTreeWidgetItem *headeritem = operationsTree->headerItem();
        headeritem->setText(0, QApplication::translate("MainWindow", "1", 0, QApplication::UnicodeUTF8));
        operationsWindow->setWindowTitle(QApplication::translate("MainWindow", "Operations", 0, QApplication::UnicodeUTF8));


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
            connect( c.get(), SIGNAL(chainChanged()), SLOT(redraw_operation_tree()), Qt::QueuedConnection );
        }

        connect( project_->head.get(), SIGNAL(headChanged()), SLOT(redraw_operation_tree()), Qt::QueuedConnection );


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
