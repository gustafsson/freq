#include "mainwindow.h"

// Ui
#include "ui_mainwindow.h"

// Sonic AWE
#include "sawe/application.h"

// Qt
#include <QCloseEvent>
#include <QSettings>
#include <QDir>
#include <QDesktopServices>
#include <QUrl>

using namespace std;
using namespace boost;

namespace Ui {

SaweMainWindow::
        SaweMainWindow(const char* title, Sawe::Project* project, QWidget *parent)
:   QMainWindow( parent ),
    project( project ),
    ui( new MainWindow ),
    escape_action( 0 ),
    fullscreen_widget( 0 )
{
#ifdef Q_WS_MAC
//    qt_mac_set_menubar_icons(false);
#endif
    ui->setupUi(this);
    QString qtitle = QString::fromLocal8Bit(title);
    this->setWindowTitle( qtitle );

    add_widgets();

    hide();
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
#if !defined(TARGET_reader)
    connect(ui->actionSave_project, SIGNAL(triggered()), SLOT(saveProject()));
    connect(ui->actionSave_project_as, SIGNAL(triggered()), SLOT(saveProjectAs()));
#else
    ui->actionSave_project->setEnabled( false );
    ui->actionSave_project_as->setEnabled( false );
#endif
    connect(ui->actionExit, SIGNAL(triggered()), SLOT(close()));
    connect(ui->actionToggleFullscreen, SIGNAL(toggled(bool)), SLOT(toggleFullscreen(bool)));
    connect(ui->actionToggleFullscreenNoMenus, SIGNAL(toggled(bool)), SLOT(toggleFullscreenNoMenus(bool)));
    connect(ui->actionReset_layout, SIGNAL(triggered()), SLOT(resetLayout()));
    connect(ui->actionReset_view, SIGNAL(triggered()), SLOT(resetView()));
    connect(ui->actionClear_settings, SIGNAL(triggered()), SLOT(clearSettings()));
    connect(ui->actionOperation_details, SIGNAL(toggled(bool)), ui->toolPropertiesWindow, SLOT(setVisible(bool)));
    connect(ui->actionOperation_details, SIGNAL(triggered()), ui->toolPropertiesWindow, SLOT(raise()));
    connect(ui->toolPropertiesWindow, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibilityToolProperties(bool)));
#if !defined(TARGET_reader)
    connect(ui->action_Enter_product_key, SIGNAL(triggered()), SLOT(reenterProductKey()));
#else
    ui->action_Enter_product_key->setVisible( false );
#endif
    connect(ui->actionMuchdifferent_com, SIGNAL(triggered()), SLOT(gotomuchdifferent()));
    connect(ui->actionReport_a_bug, SIGNAL(triggered()), SLOT(gotobugsmuchdifferent()));
    connect(ui->actionAsk_for_help, SIGNAL(triggered()), SLOT(gotosonicaweforum()));
    connect(ui->actionFind_plugins, SIGNAL(triggered()), SLOT(findplugins()));
    connect(ui->actionFind_updates, SIGNAL(triggered()), SLOT(findupdates()));

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
#ifdef _WIN32
            display.replace("/", "\\");
#endif
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
        disableFullscreen()
{
    ui->actionToggleFullscreen->setChecked( false );
    ui->actionToggleFullscreenNoMenus->setChecked( false );
    toggleFullscreen( false );
    toggleFullscreenNoMenus( false );
}

void SaweMainWindow::
        closeEvent(QCloseEvent * e)
{
#if !defined(TARGET_reader)
    if (project->isModified())
    {
        if (!askSaveChanges())
        {
            e->ignore();
            return;
        }
    }
#endif

    e->accept();

    {
        TaskInfo ti("onMainWindowCloseEvent");
        emit onMainWindowCloseEvent( this );
    }

    {
        TaskInfo ti("Saving settings");
        QSettings().setValue("GuiState", saveSettings());
    }

    {
        TaskTimer ti("QMainWindow::closeEvent");
        QMainWindow::closeEvent(e);
    }
}


#if !defined(TARGET_reader)
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
#endif

void SaweMainWindow::
        openRecentFile()
{
    QAction* a = dynamic_cast<QAction*>(sender());
    BOOST_ASSERT( a );
    QString s = a->data().toString();
    BOOST_ASSERT( !s.isEmpty() );
    if (0 == Sawe::Application::global_ptr()->slotOpen_file( s.toLocal8Bit().constData() ))
    {
        QSettings settings;
        QStringList recent_files = settings.value("recent files").toStringList();
        recent_files.removeAll( s );
        settings.setValue("recent files", recent_files);
    }
}


#if !defined(TARGET_reader)
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
#endif


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

    if (0 == fullscreen_widget)
        fullscreen_widget = centralWidget();

    if (fullscreen)
    {
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
        resetLayout()
{
    project->resetLayout();
}


void SaweMainWindow::
        resetView()
{
    project->resetView();
}


void SaweMainWindow::
        clearSettings()
{
    if (QMessageBox::Yes == QMessageBox::question(this, "Sonic AWE", "Clear all user defined settings?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No))
    {
        QSettings settings;
        QString value = settings.value("value").toString();
        settings.clear();
        settings.setValue("value", value);

        resetLayout();
        resetView();
    }
}


void SaweMainWindow::
        reenterProductKey()
{
    if (QMessageBox::Yes == QMessageBox::question(this, "Sonic AWE", "Clear the currently stored license key?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No))
    {
        QMessageBox::information(this, "Sonic AWE", "Restart Sonic AWE to enter a new license key");
        QSettings settings;
        settings.remove("value");
    }
}


void SaweMainWindow::
        gotomuchdifferent()
{
    QDesktopServices::openUrl(QUrl("http://muchdifferent.com"));
}


void SaweMainWindow::
        gotobugsmuchdifferent()
{
    QMessageBox message(
            QMessageBox::Information,
            "bugs.muchdifferent.com",
            "You are very welcome to report any bugs to us at bugs.muchdifferent.com. To help us help you, please include the log files. See logfile location in details below:");

    QString localAppDir = Sawe::Application::log_directory();
    message.setDetailedText( localAppDir );

    message.exec();

    QDesktopServices::openUrl(QUrl("http://bugs.muchdifferent.com"));
}


void SaweMainWindow::
        gotosonicaweforum()
{
    QMessageBox message(
            QMessageBox::Information,
            "sonicawe.muchdifferent.com",
            "You are very welcome to ask questions about Sonic AWE in our forum at sonicawe.muchdifferent.com!");

    message.exec();

    QDesktopServices::openUrl(QUrl("http://sonicawe.muchdifferent.com"));
}


void SaweMainWindow::
        findplugins()
{
    QMessageBox message(
            QMessageBox::Information,
            "sonicawe.muchdifferent.com",
            "If you want to browse plugins developed by others (or have a plugin to share yourself), please see our forum and search for scripts at sonicawe.muchdifferent.com.");

    message.exec();

    QDesktopServices::openUrl(QUrl("http://sonicawe.muchdifferent.com"));
}


void SaweMainWindow::
        findupdates()
{
    QMessageBox message(
            QMessageBox::Information,
            "www.muchdifferent.com",
            QString("Your version of Sonic AWE is '%1'. The latest version of Sonic AWE can be found at muchdifferent.com.").arg(Sawe::Application::version_string().c_str()));

    message.exec();

    QDesktopServices::openUrl(QUrl("http://muchdifferent.com/?page=signals-download"));
}


void SaweMainWindow::
        checkVisibilityToolProperties(bool visible)
{
    visible |= !tabifiedDockWidgets( ui->toolPropertiesWindow ).empty();
    visible |= ui->toolPropertiesWindow->isVisibleTo( ui->toolPropertiesWindow->parentWidget() );
    ui->actionOperation_details->setChecked(visible);
}


void SaweMainWindow::
        restoreSettings(QByteArray array)
{
    QMap<QString, QVariant> state;
    QDataStream ds(&array, QIODevice::ReadOnly );
    ds >> state;

    {
        TaskInfo("SaweMainWindow::readSettings - {%u actions and sliders}", state.size());
        QMapIterator<QString, QVariant> i(state);
        while (i.hasNext())
        {
            i.next();
            if (i.value().type() == QVariant::ByteArray)
                TaskInfo( "[%s] = {%u bytes}", i.key().toLatin1().data(), i.value().toByteArray().size() );
            else
                TaskInfo( "[%s] = %s", i.key().toLatin1().data(), i.value().toString().toLatin1().data() );
        }
    }

    restoreState(state["MainWindow/windowState"].toByteArray());
    restoreGuiState( this, state );

    // Always start with the navigation tool activated
    ui->actionActivateNavigation->trigger();

    // Always start stopped
    ui->actionStopPlayBack->trigger();
}


QByteArray SaweMainWindow::
        saveSettings() const
{
    QMap<QString, QVariant> state;
    state["MainWindow/windowState"] = saveState();

    getGuiState( this, state);

    {
        TaskInfo("SaweMainWindow::writeSettings - {%u actions and sliders}", state.size());
        QMapIterator<QString, QVariant> i(state);
        while (i.hasNext())
        {
            i.next();
            if (i.value().type() == QVariant::ByteArray)
                TaskInfo( "[%s] = {%u bytes}", i.key().toLatin1().data(), i.value().toByteArray().size() );
            else
                TaskInfo( "[%s] = %s", i.key().toLatin1().data(), i.value().toString().toLatin1().data() );
        }
    }


    QByteArray array;
    QDataStream ds(&array, QIODevice::WriteOnly);
    ds << state;
    return array;
}


QByteArray SaweMainWindow::
        saveGeometry() const
{
    return QMainWindow::saveGeometry();
}


QByteArray SaweMainWindow::
        saveState(int version) const
{
    return QMainWindow::saveState(version);
}


void SaweMainWindow::
        restoreGeometry(const QByteArray &state)
{
    QMainWindow::restoreGeometry(state);
}


void SaweMainWindow::
        restoreState(const QByteArray &state, int version)
{
    QMainWindow::restoreState(state, version);
}


void SaweMainWindow::
        getGuiState( const QObject* object, QMap<QString, QVariant>& state ) const
{
    if (!object->objectName().isEmpty())
        if (const QWidget* w = dynamic_cast<const QWidget*>(object))
        {
            state.insert(w->objectName()+"/geometry", w->saveGeometry());
            state.insert(w->objectName()+"/visible", w->isVisible());
        }

    foreach( const QObject* o, object->children())
    {
        if (o->objectName().isEmpty())
            continue;

        if (const QSlider* s = dynamic_cast<const QSlider*>(o))
            state.insert(s->objectName(), s->value());

        if (const QAction* a = dynamic_cast<const QAction*>(o))
            state.insert(a->objectName(), a->isChecked());
    }

    foreach( const QObject* o, object->children() )
        getGuiState( o, state );
}


void SaweMainWindow::
        restoreGuiState( QObject* o, const QMap<QString, QVariant>& state )
{
    QMapIterator<QString, QVariant> i(state);
    while (i.hasNext())
    {
        i.next();

        if (QWidget* w = dynamic_cast<QWidget*>(o)) if(o->objectName() == i.key())
        {
            w->restoreGeometry( state[ w->objectName()+"/geometry" ].toByteArray() );
            w->setVisible( state[ w->objectName()+"/visible" ].toBool() );
        }

        if (QSlider* s = o->findChild<QSlider*>( i.key() ))
            s->setValue( i.value().toInt() );

        if (QAction* a = o->findChild<QAction*>( i.key() ))
        {
            if (i.value().toBool() && !a->isChecked())
                a->trigger();
            else if (!i.value().toBool() && a->isChecked())
                a->setChecked( false );
        }
    }

    foreach( QObject* c , o->children())
        restoreGuiState( c, state );
}


} // namespace Ui
