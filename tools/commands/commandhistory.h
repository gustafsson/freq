#ifndef COMMANDHISTORY_H
#define COMMANDHISTORY_H

#include <QWidget>
#include <QScopedPointer>

class QDockWidget;
class QAction;

namespace Tools {
namespace Commands {

    class ProjectState;
namespace Ui {
    class CommandHistory;
}

class CommandHistory : public QWidget
{
    Q_OBJECT

public:
    explicit CommandHistory(ProjectState* project_state);
    ~CommandHistory();

private slots:
    void redrawHistory();

private:
    Ui::CommandHistory *ui;
    ProjectState* project_state_;
    QDockWidget* dock;
    QAction* actionCommandHistory;
};


} // namespace Commands
} // namespace Tools
#endif // COMMANDHISTORY_H
