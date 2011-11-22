#include "commandhistory.h"
#include "ui_commandhistory.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "projectstate.h"
#include "sawe/project.h"

#include <boost/foreach.hpp>

#include <QDockWidget>

namespace Tools {
namespace Commands {

CommandHistory::
        CommandHistory(ProjectState* project_state)
            :
    QWidget(),
    ui(new Ui::CommandHistory),
    project_state_(project_state)
{
    ui->setupUi(this);

    connect(project_state, SIGNAL(projectChanged(const Command*)), SLOT(redrawHistory()));


    ::Ui::SaweMainWindow* MainWindow = project_state->project()->mainWindow();
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetCommandHistory"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->resize(123, 150);
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    dock->setAllowedAreas(Qt::AllDockWidgetAreas);
    dock->setEnabled(true);
    dock->setWidget(this);
    dock->setWindowTitle("Command history");

    actionCommandHistory = new QAction( this );
    actionCommandHistory->setObjectName("actionCommandHistory");
    actionCommandHistory->setText("Command history");
    actionCommandHistory->setCheckable( true );
    actionCommandHistory->setChecked( true );

    MainWindow->getItems()->menu_Windows->addAction( actionCommandHistory );

    connect(actionCommandHistory, SIGNAL(toggled(bool)), dock, SLOT(setVisible(bool)));
    connect(actionCommandHistory, SIGNAL(triggered()), dock, SLOT(raise()));
    connect(dock, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibility(bool)));

    dock->setVisible( false );
    actionCommandHistory->setChecked( false );
}


CommandHistory::
        ~CommandHistory()
{
    delete ui;
}


void CommandHistory::
        redrawHistory()
{
    ui->listWidget->clear();
    const std::vector<CommandP>& list = project_state_->commandList().getCommandList();
    const Command* present = project_state_->commandList().present();

    BOOST_FOREACH(CommandP p, list)
    {
        if (p.get() == present)
            ui->listWidget->addItem(("-> " + p->toString()).c_str());
        else
            ui->listWidget->addItem(p->toString().c_str());
    }
}


} // namespace Commands
} // namespace Tools
