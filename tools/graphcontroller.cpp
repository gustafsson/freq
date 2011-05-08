#include "graphcontroller.h"

#include "renderview.h"
#include "ui_mainwindow.h"
#include "ui/mainwindow.h"
#include "tools/support/operation-composite.h"
#include "signal/operationcache.h"

#include <boost/foreach.hpp>

#include <QPushButton>


#define DEBUG_GRAPH
//#define DEBUG_GRAPH if(0)

#define INFO_GRAPH
//#define INFO_GRAPH if(0)

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
        Sawe::Project* project_;
    public:
        TreeWidget(QWidget*parent, Sawe::Project* project)
            :
            QTreeWidget(parent),
            project_(project)
        {}

        virtual void dropEvent ( QDropEvent * event ) {
            QTreeWidget::dropEvent ( event );

            DEBUG_GRAPH TaskInfo ti("operation_tree dropEvent");
            DEBUG_GRAPH TaskInfo("project head source: %s", project_->head->head_source()->toString().c_str());
            DEBUG_GRAPH TaskInfo("project head output: %s", project_->head->head_source()->parentsToString().c_str());

            for (int l = 0; l<invisibleRootItem()->childCount(); ++l)
            {
                QTreeWidgetItem* layer = invisibleRootItem()->child(l);

                // Deselect all
                for (int i=0; i<layer->childCount(); ++i)
                    layer->child(i)->setSelected( false );

                Signal::pOperation
                        // The operation closest to root that was changed by the move operation
                        firstmoved,

                        // The operation furthest away from root that was changed by the move operation
                        lastmoved;

                // Make sure the root element is at the bottom
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
                            lastmoved = itm->tail;
                            itm->tail->source( src );
                            setCurrentItem( itm );
                        }
                    }
                    else
                        chain = itm->chain;

                    src = itm->operation;
                }

                // Set the tip
                if (chain && src)
                    chain->tip_source( src );
                if (firstmoved)
                    firstmoved->source()->invalidate_samples(Signal::Operation::affectedDiff( firstmoved, lastmoved ));
            }

            DEBUG_GRAPH TaskInfo("results");
            DEBUG_GRAPH TaskInfo("project head source: %s", project_->head->head_source()->toString().c_str());
            DEBUG_GRAPH TaskInfo("project head output: %s", project_->head->head_source()->parentsToString().c_str());
        }
    };

    GraphController::
            GraphController( RenderView* render_view )
                :
                render_view_(render_view),
                project_(render_view->model->project()),
                dontredraw_(false),
                removing_(false)
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
        if (dontredraw_ && !removing_)
            return;

        operationsTree->clear();

        DEBUG_GRAPH TaskInfo ti("redraw_operation_tree");
        DEBUG_GRAPH TaskInfo("project head source: %s", project_->head->head_source()->toString().c_str());
        DEBUG_GRAPH TaskInfo("project head output: %s", project_->head->head_source()->parentsToString().c_str());

        operationsTree->invisibleRootItem()->setFlags(
                operationsTree->invisibleRootItem()->flags() & ~Qt::ItemIsDropEnabled );

        BOOST_FOREACH( Signal::pChain c, project_->layers.layers() )
        {
            QTreeWidgetItem* chainItm = new QTreeWidgetItem(operationsTree);
            chainItm->setText(0, QString::fromLocal8Bit( c->name.c_str() ) );
            chainItm->setExpanded( true );
            chainItm->setFlags( chainItm->flags() & ~Qt::ItemIsSelectable );

            Signal::pOperation o = c->tip_source();
            INFO_GRAPH TaskInfo ti("Operation tree: %s", o->toString().c_str());

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
                QString name = QString::fromLocal8Bit( o->name().c_str() );
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
        TreeItem* currentItem = dynamic_cast<TreeItem*>(current);
        TreeItem* previousItem = dynamic_cast<TreeItem*>(previous);

        if (currentItem && (bool)currentItem->operation->source())
            operationsTree->setContextMenuPolicy(Qt::ActionsContextMenu);
        else
            operationsTree->setContextMenuPolicy(Qt::NoContextMenu);

        dontredraw_ = true;
        timerUpdateContextMenu.start();

        if (!previous || !current)
        {
            return;
        }


        if ( !currentItem )
        {
            QTreeWidgetItem* selectAnother = 0;
            if (current && current->childCount())
                selectAnother = current->child(0);
            else if (previousItem)
                selectAnother = previous;

            if (selectAnother)
            {
                operationsTree->clearSelection();
                operationsTree->clearFocus();
                operationsTree->setCurrentItem( selectAnother );

                operationsTree->setContextMenuPolicy(Qt::NoContextMenu);
            }
        }
        else
        {
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
        DEBUG_GRAPH TaskInfo ti("removeSelected");
        DEBUG_GRAPH TaskInfo("project head source: %s", project_->head->head_source()->toString().c_str());
        DEBUG_GRAPH TaskInfo("project head output: %s", project_->head->head_source()->parentsToString().c_str());

        QList<QTreeWidgetItem*> itms = operationsTree->selectedItems();
        if (itms.empty())
            return;

        Signal::pOperation currentOperation, newCurrentOperation;
        Signal::pChain currentChain;

        {
            TreeItem* currentItem = dynamic_cast<TreeItem*>(itms.front());
            if ( !currentItem )
                return;

            currentOperation = currentItem->operation;
            currentChain = currentItem->chain;
        }

        // If the current operation is a cache, don't just remove the cache but
        // remove what was cached as well. So jump an extra steps down in source()
        if (dynamic_cast<Signal::OperationCacheLayer*>(currentOperation.get()) )
            newCurrentOperation = currentOperation->source()->source();
        else
            newCurrentOperation = currentOperation->source();

        if (!newCurrentOperation)
            return;

        removing_ = true;

        Signal::pOperation newHead = Signal::Operation::findParentOfSource( currentChain->tip_source(), currentOperation );
        if (newHead)
        {
            newHead->invalidate_samples( Signal::Operation::affectedDiff(newHead->source(), newCurrentOperation ));

            newHead->source( newCurrentOperation );

            // If there is a cache right above this, set the cache as head_source instead
            Signal::pOperation o2 = Signal::Operation::findParentOfSource( currentChain->tip_source(), newHead );
            if (dynamic_cast<Signal::OperationCacheLayer*>(o2.get()) )
                newHead = o2;
        }
        else
        {
            newHead = newCurrentOperation;
        }

        Signal::pChainHead head = project_->tools().render_model.renderSignalTarget->findHead( currentChain );
        head->head_source( newHead );

        head = project_->tools().playback_model.playbackTarget->findHead( currentChain );
        head->head_source( newHead );

        project_->head->head_source( newHead );
        project_->setModified();

        currentOperation->source(Signal::pOperation());

        redraw_operation_tree();

        if (newHead==newCurrentOperation)
            currentChain->tip_source( newHead );
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
            updateContextMenu()
    {
        QList<QTreeWidgetItem*> itms = operationsTree->selectedItems();
        bool currentHasSource = false;
        this->dontredraw_ = false;
        this->removing_ = false;

        if (!itms.empty())
        {
            TreeItem* currentItem = dynamic_cast<TreeItem*>(itms.front());

            if (currentItem)
                currentHasSource = (bool)currentItem->operation->source();
        }

        if (currentHasSource)
            operationsTree->setContextMenuPolicy(Qt::ActionsContextMenu);
        else
            operationsTree->setContextMenuPolicy(Qt::NoContextMenu);

        removeSelectedButton->setEnabled( currentHasSource );
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
        operationsTree = new TreeWidget(dockWidgetContents, project_);
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
        removeCurrentItem->setShortcut(Qt::Key_Backspace);
        operationsTree->addAction(removeCurrentItem);
        connect(removeCurrentItem, SIGNAL(triggered()), SLOT(removeSelected()));
        //operationsTree->header()->setDefaultSectionSize(60);
        //operationsTree->header()->setMinimumSectionSize(20);
        //operationsTree->setSelectionMode( QAbstractItemView::MultiSelection );

        QWidget* buttons = new QWidget;
        //buttons->setLayout( new QHBoxLayout );
        buttons->setLayout( new QVBoxLayout );
        removeSelectedButton = new QPushButton("Remove selected");
        QPushButton* removeHiddenButton = new QPushButton("Remove hidden");
        QPushButton* removeCachesdButton = new QPushButton("Discard caches");
        removeCachesdButton->setToolTip( "Discard caches for operations above the selected operation" );
        connect(removeSelectedButton, SIGNAL(clicked()), SLOT(removeSelected()));
        connect(removeHiddenButton, SIGNAL(clicked()), SLOT(removeHidden()));
        connect(removeCachesdButton, SIGNAL(clicked()), SLOT(removeCaches()));
        buttons->layout()->addWidget( removeSelectedButton );
        //buttons->layout()->addWidget( removeHiddenButton );
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

        timerUpdateContextMenu.setSingleShot( true );
        timerUpdateContextMenu.setInterval( 300 );
        connect(&timerUpdateContextMenu, SIGNAL(timeout()), SLOT(updateContextMenu()));


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
