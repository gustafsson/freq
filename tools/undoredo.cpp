#include "undoredo.h"

#include "tools/commands/projectstate.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

namespace Tools {

UndoRedo::UndoRedo(::Sawe::Project *project)
    :
    project_(project)
{
    ::Ui::MainWindow* ui = project->mainWindow()->getItems();

    connect(project->projectState(), SIGNAL(projectChanged(const Command*)), SLOT(updateNames()));
    connect(ui->action_Undo, SIGNAL(triggered()), SLOT(undo()));
    connect(ui->action_Redo, SIGNAL(triggered()), SLOT(redo()));
}


void UndoRedo::
        updateNames()
{
    ::Ui::MainWindow* ui = project_->mainWindow()->getItems();
    std::string canUndo = project_->projectState()->commandList().canUndo();
    std::string canRedo = project_->projectState()->commandList().canRedo();
    ui->action_Undo->setText(QString("&Undo %1").arg(canUndo.c_str()));
    ui->action_Redo->setText(QString("&Redo %1").arg(canRedo.c_str()));
    ui->action_Undo->setEnabled( !canUndo.empty() );
    ui->action_Redo->setEnabled( !canRedo.empty() );
}


void UndoRedo::
        undo()
{
    project_->projectState()->undo();
}


void UndoRedo::
        redo()
{
    project_->projectState()->redo();
}

} // namespace Tools
