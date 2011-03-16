#include "mainwindow.h"

// Ui
#include "ui_mainwindow.h"

// Sonic AWE
#include "sawe/application.h"

// Qt
#include <QCloseEvent>
#include <QSettings>
#include <QDir>

using namespace std;
using namespace boost;

namespace Ui {

SaweMainWindow::
        SaweMainWindow(const char* title, Sawe::Project* project, QWidget *parent)
:   QMainWindow(parent),
    project( project ),
    ui(new MainWindow),
    escape_action(0)
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
    connect(ui->actionToggleFullscreen, SIGNAL(toggled(bool)), SLOT(toggleFullscreen(bool)));
    connect(ui->actionToggleFullscreenNoMenus, SIGNAL(toggled(bool)), SLOT(toggleFullscreenNoMenus(bool)));
    connect(ui->actionRestore_layout, SIGNAL(triggered()), SLOT(restoreLayout()));
    connect(ui->actionOperation_details, SIGNAL(toggled(bool)), ui->toolPropertiesWindow, SLOT(setVisible(bool)));
    connect(ui->actionOperation_details, SIGNAL(triggered()), ui->toolPropertiesWindow, SLOT(raise()));
    connect(ui->toolPropertiesWindow, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibilityToolProperties(bool)));
    ui->actionOperation_details->setChecked( false );

    // Make the two fullscreen modes exclusive
    fullscreen_combo.decheckable( true );
    fullscreen_combo.addAction( ui->actionToggleFullscreen );
    fullscreen_combo.addAction( ui->actionToggleFullscreenNoMenus );

    ui->actionToggleFullscreenNoMenus->setShortcutContext( Qt::ApplicationShortcut );

    // TODO remove layerWidget and deleteFilterButton
    //connect(ui->layerWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(slotDbclkFilterItem(QListWidgetItem*)));
    //connect(ui->layerWidget, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(slotNewSelection(QListWidgetItem*)));
    //connect(ui->deleteFilterButton, SIGNAL(clicked(void)), this, SLOT(slotDeleteSelection(void)));

    // connect 'show window X' menu items to their respective windows

    // TODO move into each tool
    // TODO remove actionToggleTimelineWindow, and dockWidgetTimeline
//    connectActionToWindow(ui->actionToggleTopFilterWindow, ui->topFilterWindow);

    //    connectActionToWindow(ui->actionToggleTimelineWindow, ui->dockWidgetTimeline);

    // TODO move into each tool
    this->addDockWidget( Qt::RightDockWidgetArea, ui->toolPropertiesWindow );
    //this->addDockWidget( Qt::RightDockWidgetArea, ui->topFilterWindow );
    //this->addDockWidget( Qt::RightDockWidgetArea, ui->historyWindow );

    //ui->toolPropertiesWindow->hide();
    ui->topFilterWindow->hide();
    ui->historyWindow->hide();
    //this->removeDockWidget( ui->toolPropertiesWindow );
    //this->removeDockWidget( ui->operationsWindow );
    //this->removeDockWidget( ui->topFilterWindow );
    //this->removeDockWidget( ui->historyWindow );

    // todo move into toolfactory
//    this->tabifyDockWidget(ui->operationsWindow, ui->topFilterWindow);
//    this->tabifyDockWidget(ui->operationsWindow, ui->historyWindow);
//    ui->topFilterWindow->raise();

    // todo move into toolfactory
    this->addToolBar( Qt::TopToolBarArea, ui->toolBarOperation );
    //this->addToolBar( Qt::TopToolBarArea, ui->toolBarMatlab );
    this->addToolBar( Qt::LeftToolBarArea, ui->toolBarPlay );

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


    {
        QSettings settings;
        QStringList recent_files = settings.value("recent files").toStringList();
        ui->menu_Recent_files->setEnabled( !recent_files.empty() );
        int i = 0;
        foreach(QString recent, recent_files)
        {
            QString home = QDir::homePath();
            QString display = recent;
            if (display.left(home.size())==home)
            {
#ifdef __GNUC__
                display = "~" + display.mid( home.size() );
#else
                display = display.mid( home.size()+1 );
#endif
            }

            i++;
            display = QString("%1%2. %3").arg(i<10?"&":"").arg(i).arg(display);

            QAction * a = new QAction(display, this);
            a->setData( recent );
            connect(a, SIGNAL(triggered()), SLOT(openRecentFile()));

            ui->menu_Recent_files->addAction( a );
        }
    }

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
        if (!askSaveChanges())
        {
            e->ignore();
            return;
        }
    }

    e->accept();

    {
        TaskInfo ti("onMainWindowCloseEvent");
        emit onMainWindowCloseEvent( this );
    }

    {
        TaskInfo ti("Saving settings");
        QSettings settings;
        settings.setValue("geometry", saveGeometry());
        settings.setValue("windowState", saveState());
    }

    {
        TaskTimer ti("QMainWindow::closeEvent");
        QMainWindow::closeEvent(e);
    }
}


bool SaweMainWindow::
        askSaveChanges()
{
    TaskInfo("Save current state of the project?");
    QMessageBox save_changes_msgbox("Save Changes", "Save current state of the project?",
                                          QMessageBox::Question, QMessageBox::Discard, QMessageBox::Cancel, QMessageBox::Save, this );
    save_changes_msgbox.setDetailedText( QString::fromStdString( "Current state:\n" + project->layers.toString()) );
    save_changes_msgbox.exec();
    QAbstractButton * button = save_changes_msgbox.clickedButton();
    TaskInfo("Save changes answer: %s, %d",
             button->text().toStdString().c_str(),
             (int)save_changes_msgbox.buttonRole(button));

    switch ( save_changes_msgbox.buttonRole(button) )
    {
    case QMessageBox::DestructiveRole:
        return true; // close

    case QMessageBox::AcceptRole:
        if (!project->save())
        {
            return false; // abort
        }

        return true; // close

    case QMessageBox::RejectRole:
    default:
        return false; // abort
    }
}


void SaweMainWindow::
        openRecentFile()
{
    QAction* a = dynamic_cast<QAction*>(sender());
    BOOST_ASSERT( a );
    QString s = a->data().toString();
    BOOST_ASSERT( !s.isEmpty() );
    if (0 == Sawe::Application::global_ptr()->slotOpen_file( s.toStdString() ))
    {
        QSettings settings;
        QStringList recent_files = settings.value("recent files").toStringList();
        recent_files.removeAll( s );
        settings.setValue("recent files", recent_files);
    }
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


void SaweMainWindow::
        toggleFullscreen( bool fullscreen )
{
    if (fullscreen)
        ui->actionToggleFullscreenNoMenus->setChecked( false );

    this->setWindowState( fullscreen ? Qt::WindowFullScreen : Qt::WindowActive);
}


void SaweMainWindow::
        toggleFullscreenNoMenus( bool fullscreen )
{
    if (fullscreen)
        ui->actionToggleFullscreen->setChecked( false );

    TaskInfo ti("%s %d", __FUNCTION__, fullscreen);

    if (fullscreen)
    {
        fullscreen_widget = centralWidget();
        fullscreen_widget->setParent(0);
        fullscreen_widget->setWindowState( Qt::WindowFullScreen );
        fullscreen_widget->show();
        hide();

        QList<QKeySequence> shortcuts;
        //shortcuts.push_back( Qt::ALT | Qt::Key_Return ); using ui->actionToggleFullscreenNoMenus instead
        shortcuts.push_back( Qt::ALT | Qt::Key_Enter );
        shortcuts.push_back( Qt::Key_Escape );
        if (0==escape_action)
        {
            escape_action = new QAction( this );
            escape_action->setShortcuts( shortcuts );
            escape_action->setCheckable( true );

            connect(escape_action, SIGNAL(triggered(bool)), ui->actionToggleFullscreenNoMenus, SLOT(setChecked(bool)));
        }

        escape_action->setChecked( true );

        fullscreen_widget->addAction( escape_action );
        fullscreen_widget->addAction( ui->actionToggleFullscreenNoMenus );
    } else {
        setCentralWidget( fullscreen_widget );
        fullscreen_widget->setWindowState( Qt::WindowActive );
        show();

        fullscreen_widget->removeAction( escape_action );
        fullscreen_widget->removeAction( ui->actionToggleFullscreenNoMenus );
    }
}


void SaweMainWindow::
        restoreLayout()
{
    project->restoreDefaultLayout();
}


void SaweMainWindow::
        checkVisibilityToolProperties(bool visible)
{
    visible |= !tabifiedDockWidgets( ui->toolPropertiesWindow ).empty();
    visible |= ui->toolPropertiesWindow->isVisibleTo( ui->toolPropertiesWindow->parentWidget() );
    ui->actionOperation_details->setChecked(visible);
}




} // namespace Ui
