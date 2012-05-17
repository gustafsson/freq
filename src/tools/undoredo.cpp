#include "undoredo.h"

#include "tools/commands/commandinvoker.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

namespace Tools {

UndoRedo::UndoRedo(::Sawe::Project *project)
    :
    project_(project)
{
    ::Ui::MainWindow* ui = project->mainWindow()->getItems();

    connect(project->commandInvoker(), SIGNAL(projectChanged(const Command*)), SLOT(updateNames()));
    connect(ui->action_Undo, SIGNAL(triggered()), SLOT(undo()));
    connect(ui->action_Redo, SIGNAL(triggered()), SLOT(redo()));
}


void UndoRedo::
        updateNames()
{
    ::Ui::MainWindow* ui = project_->mainWindow()->getItems();
    std::string canUndo = project_->commandInvoker()->commandList().canUndo();
    std::string canRedo = project_->commandInvoker()->commandList().canRedo();
    ui->action_Undo->setText(QString("&Undo %1").arg(canUndo.c_str()));
    ui->action_Redo->setText(QString("&Redo %1").arg(canRedo.c_str()));
    ui->action_Undo->setEnabled( !canUndo.empty() );
    ui->action_Redo->setEnabled( !canRedo.empty() );
}


void UndoRedo::
        undo()
{
    project_->commandInvoker()->undo();
}


void UndoRedo::
        redo()
{
    project_->commandInvoker()->redo();
}

} // namespace Tools
