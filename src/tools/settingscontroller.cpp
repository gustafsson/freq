#include "settingscontroller.h"
#include "settingsdialog.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "sawe/project.h"
#include <QAction>

namespace Tools {

SettingsController::
        SettingsController(Sawe::Project*project)
            :
            project_(project),
            initialized_(false)
{
    showSettingsAction = new QAction( "&Settings", project->mainWindowWidget() );
    showSettingsAction->setObjectName( "showSettingsAction" );
    showSettingsAction->setShortcut( QString("Alt+S") );
    showSettingsAction->setMenuRole( QAction::PreferencesRole );

    project->mainWindow()->getItems()->menuTools->addAction( showSettingsAction );
    project->mainWindowWidget()->addAction( showSettingsAction );

    connect(showSettingsAction, SIGNAL(triggered()), SLOT(showSettings()), Qt::QueuedConnection);

    showSettingsAction->trigger();
}


SettingsController::
        ~SettingsController()
{
    // mainWindow owns showSettingsAction
}


void SettingsController::
        showSettings()
{
    // the pointer is owned by project->mainWindowWidget(), the dialog has WA_DeleteOnClose
    SettingsDialog* settingsDialog = new SettingsDialog(project_, project_->mainWindowWidget());

    if (!initialized_)
    {
        settingsDialog->close(); // destroys settingsDialog
        initialized_ = true;
        return;
    }

    connect(settingsDialog, SIGNAL(finished(int)), SLOT(dialogFinished(int)));

    settingsDialog->show();
}


void SettingsController::
        dialogFinished(int /*result*/)
{
    showSettingsAction->setChecked( false );
}

} // namespace Tools
