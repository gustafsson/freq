#include "mainwindow.h"

// Ui
#include "ui_mainwindow.h"
#include "comboboxaction.h"
#include "propertiesselection.h"

// Sonic AWE
#include "adapters/microphonerecorder.h"
#include "adapters/audiofile.h"
#include "tfr/filter.h"
#include "tools/timelineview.h"
#include "sawe/application.h"
#include "signal/operation-basic.h"

// Qt
#include <QMessageBox>
#include <QKeyEvent>
#include <QSlider>

// boost
#include <boost/graph/adjacency_list.hpp>
#include <boost/unordered_set.hpp>
#include <boost/graph/adjacency_iterator.hpp>

// std
#include <sstream>
#include <iomanip>
#include <demangle.h>

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

using namespace std;
using namespace boost;

namespace Ui {

SaweMainWindow::
        SaweMainWindow(const char* title, Sawe::Project* project, QWidget *parent)
:   QMainWindow(parent),
    project( project ),
    ui(new MainWindow)
{
#ifdef Q_WS_MAC
//    qt_mac_set_menubar_icons(false);
#endif
    ui->setupUi(this);
    QString qtitle = QString::fromLocal8Bit(title);
    this->setWindowTitle( qtitle );

    add_widgets();

    hide();
    show();
}


void SaweMainWindow::
        add_widgets()
{
    // setup docking areas
    setCorner( Qt::BottomLeftCorner, Qt::LeftDockWidgetArea );
    setCorner( Qt::BottomRightCorner, Qt::RightDockWidgetArea );
    setCorner( Qt::TopLeftCorner, Qt::LeftDockWidgetArea );
    setCorner( Qt::TopRightCorner, Qt::RightDockWidgetArea );

    // Connect actions in the File menu
    connect(ui->actionNew_recording, SIGNAL(triggered()), Sawe::Application::global_ptr(), SLOT(slotNew_recording()));
    connect(ui->actionOpen, SIGNAL(triggered()), Sawe::Application::global_ptr(), SLOT(slotOpen_file()));
    connect(ui->actionSave_project, SIGNAL(triggered()), SLOT(saveProject()));
    connect(ui->actionSave_project_as, SIGNAL(triggered()), SLOT(saveProjectAs()));
    connect(ui->actionExit, SIGNAL(triggered()), SLOT(close()));


    // TODO remove layerWidget and deleteFilterButton
    //connect(ui->layerWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(slotDbclkFilterItem(QListWidgetItem*)));
    //connect(ui->layerWidget, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(slotNewSelection(QListWidgetItem*)));
    //connect(ui->deleteFilterButton, SIGNAL(clicked(void)), this, SLOT(slotDeleteSelection(void)));

    // connect 'show window X' menu items to their respective windows

    // TODO move into each tool
    // TODO remove actionToggleTimelineWindow, and dockWidgetTimeline
//    connectActionToWindow(ui->actionToggleTopFilterWindow, ui->topFilterWindow);
//    connectActionToWindow(ui->actionToggleOperationsWindow, ui->operationsWindow);
//    connectActionToWindow(ui->actionToggleHistoryWindow, ui->historyWindow);
//    connectActionToWindow(ui->actionToggleTimelineWindow, ui->dockWidgetTimeline);
//    connect(ui->actionToggleToolToolBox, SIGNAL(toggled(bool)), ui->toolBarTool, SLOT(setVisible(bool)));

    // TODO move into each tool
    //this->addDockWidget( Qt::RightDockWidgetArea, ui->toolPropertiesWindow );
    this->addDockWidget( Qt::RightDockWidgetArea, ui->operationsWindow );
    //this->addDockWidget( Qt::RightDockWidgetArea, ui->topFilterWindow );
    //this->addDockWidget( Qt::RightDockWidgetArea, ui->historyWindow );

    this->removeDockWidget( ui->toolPropertiesWindow );
    //this->removeDockWidget( ui->operationsWindow );
    this->removeDockWidget( ui->topFilterWindow );
    this->removeDockWidget( ui->historyWindow );

    // todo move into toolfactory
    this->tabifyDockWidget(ui->operationsWindow, ui->topFilterWindow);
    this->tabifyDockWidget(ui->operationsWindow, ui->historyWindow);
    ui->topFilterWindow->raise();

    // todo move into toolfactory
    this->addToolBar( Qt::TopToolBarArea, ui->toolBarTool );
    this->addToolBar( Qt::TopToolBarArea, ui->toolBarOperation );
    this->addToolBar( Qt::BottomToolBarArea, ui->toolBarPlay );

    //new Saweui::PropertiesSelection( ui->toolPropertiesWindow );
    //ui->toolPropertiesWindow-
    //new Saweui::PropertiesSelection( ui->frameProperties ); // TODO fix, tidy, what?

    /*QComboBoxAction * qb = new QComboBoxAction();
    qb->addActionItem( ui->actionActivateSelection );
    qb->addActionItem( ui->actionActivateNavigation );
    ui->toolBarTool->addWidget( qb );*/


    // TODO what does actionToolSelect do?
    /*{   QToolButton * tb = new QToolButton();

        tb->setDefaultAction( ui->actionToolSelect );

        ui->toolBarTool->addWidget( tb );
        connect( tb, SIGNAL(triggered(QAction *)), tb, SLOT(setDefaultAction(QAction *)));
    }*/

    connect(this, SIGNAL(onMainWindowCloseEvent(QWidget*)),
        Sawe::Application::global_ptr(), SLOT(slotClosed_window( QWidget*)),
        Qt::QueuedConnection);
}

/*
 todo move into each separate tool
void SaweMainWindow::slotCheckWindowStates(bool)
{
    unsigned int size = controlledWindows.size();
    for(unsigned int i = 0; i < size; i++)
    {
        controlledWindows[i].a->setChecked(!(controlledWindows[i].w->isHidden()));
    }
}
void SaweMainWindow::slotCheckActionStates(bool)
{
    unsigned int size = controlledWindows.size();
    for(unsigned int i = 0; i < size; i++)
    {
        controlledWindows[i].w->setVisible(controlledWindows[i].a->isChecked());
    }
}
*/

/*
 todo create some generic solution for showing/hiding tool windows
void SaweMainWindow::connectActionToWindow(QAction *a, QWidget *b)
{
    connect(a, SIGNAL(toggled(bool)), this, SLOT(slotCheckActionStates(bool)));
    connect(b, SIGNAL(visibilityChanged(bool)), this, SLOT(slotCheckWindowStates(bool)));
    controlledWindows.push_back(ActionWindowPair(b, a));
}
*/

SaweMainWindow::~SaweMainWindow()
{
    TaskTimer tt("~SaweMainWindow");
    delete ui;
}


/* todo remove
void SaweMainWindow::slotDbclkFilterItem(QListWidgetItem * item)
{
    //emit sendCurrentSelection(ui->layerWidget->row(item), );
}


void SaweMainWindow::slotNewSelection(QListWidgetItem *item)
{
    int index = ui->layerWidget->row(item);
    if(index < 0){
        ui->deleteFilterButton->setEnabled(false);
        return;
    }else{
        ui->deleteFilterButton->setEnabled(true);
    }
    bool checked = false;
    if(ui->layerWidget->item(index)->checkState() == Qt::Checked){
        checked = true;
    }
    printf("Selecting new item: index:%d checked %d\n", index, checked);
    emit sendCurrentSelection(index, checked);
}

void SaweMainWindow::slotDeleteSelection(void)
{
    emit sendRemoveItem(ui->layerWidget->currentRow());
}
*/


void SaweMainWindow::
        closeEvent(QCloseEvent * e)
{
    if (project->isModified())
    {
        switch( QMessageBox::question(this, "Save Changes",
                              "Save current state of the project?",
                              QMessageBox::Discard, QMessageBox::Cancel, QMessageBox::Save) )
        {
        case QMessageBox::Discard:
            break;

        case QMessageBox::Cancel:
            e->ignore();
            return;

        case QMessageBox::Save:
            if (!project->save())
            {
                e->ignore();
                return;
            }

            break;

        default:
            e->ignore();
            return;
        }
    }

    e->accept();

    // TODO hack
#ifndef _MSC_VER
#ifndef __APPLE__
    emit onMainWindowCloseEvent( this );
#endif
#endif
}


void SaweMainWindow::
        saveProject()
{
    project->save();
}


void SaweMainWindow::
        saveProjectAs()
{
    project->saveAs();
}

} // namespace Ui
