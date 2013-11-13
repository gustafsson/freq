#include "commandhistory.h"
#include "ui_commandhistory.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "commandinvoker.h"
#include "sawe/project.h"

#include <boost/foreach.hpp>

#include <QDockWidget>
#include <QTimer>

namespace Tools {
namespace Commands {

CommandHistory::
        CommandHistory(CommandInvoker* command_invoker)
            :
    QWidget(),
    ui(new Ui::CommandHistory),
    command_invoker_(command_invoker)
{
    ui->setupUi(this);

    connect(command_invoker, SIGNAL(projectChanged(const Command*)), SLOT(redrawHistory()));

    redrawHistory();

    ::Ui::SaweMainWindow* MainWindow = command_invoker->project()->mainWindow();
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetCommandHistory"));
    dock->setWidget (this);
    MainWindow->addDockWidget (Qt::RightDockWidgetArea, dock);
    dock->setWindowTitle("Command history");

    MainWindow->getItems()->menu_Windows->addAction( dock->toggleViewAction () );

    connect(dock->toggleViewAction (), SIGNAL(triggered()), dock, SLOT(raise()));
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
    const std::vector<pCommand>& list = command_invoker_->commandList().getCommandList();
    const Command* present = command_invoker_->commandList().present();

    BOOST_FOREACH(pCommand p, list)
    {
        if (p.get() == present)
            ui->listWidget->addItem(("-> " + p->toString()).c_str());
        else
            ui->listWidget->addItem(p->toString().c_str());
    }
}


} // namespace Commands
} // namespace Tools
