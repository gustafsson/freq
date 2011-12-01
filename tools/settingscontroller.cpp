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
            project(project)
{
    showSettingsAction = new QAction( "&Settings", project->mainWindowWidget() );
    showSettingsAction->setObjectName( "showSettingsAction" );
    showSettingsAction->setShortcut( QString("Alt+S") );

    project->mainWindow()->getItems()->menuTools->addAction( showSettingsAction );
    project->mainWindowWidget()->addAction( showSettingsAction );

    connect(showSettingsAction, SIGNAL(triggered()), SLOT(showSettings()));
}


SettingsController::
        ~SettingsController()
{
    // mainWindow owns showSettingsAction
}


void SettingsController::
        showSettings()
{
    // the pointer is owned by project->mainWindowWidget()
    SettingsDialog* settingsDialog = new SettingsDialog(project, project->mainWindowWidget());

    connect(settingsDialog, SIGNAL(finished(int)), SLOT(dialogFinished(int)));

    settingsDialog->show();
}


void SettingsController::
        dialogFinished(int /*result*/)
{
    showSettingsAction->setChecked( false );
}

} // namespace Tools
