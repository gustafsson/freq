#ifndef COMMANDHISTORY_H
#define COMMANDHISTORY_H

#include <QWidget>
#include <QScopedPointer>

class QDockWidget;
class QAction;

namespace Tools {
namespace Commands {

    class CommandInvoker;
namespace Ui {
    class CommandHistory;
}

class CommandHistory : public QWidget
{
    Q_OBJECT
public:

    explicit CommandHistory(CommandInvoker* command_invoker);
    ~CommandHistory();

private slots:
    void redrawHistory();
    void checkVisibility(bool visible);

private:
    Ui::CommandHistory *ui;
    CommandInvoker* command_invoker_;
    QDockWidget* dock;
    QAction* actionCommandHistory;
};


} // namespace Commands
} // namespace Tools
#endif // COMMANDHISTORY_H
