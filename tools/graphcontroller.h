#ifndef GRAPHCONTROLLER_H
#define GRAPHCONTROLLER_H

#include "signal/chain.h"

#include <QObject>

class QDockWidget;
class QWidget;
class QVBoxLayout;
class QTreeWidget;
class QTreeWidgetItem;
class QAction;

namespace Sawe
{
    class Project;
}

namespace Tools
{

class RenderView;

class GraphController: public QObject
{
    Q_OBJECT
public:
    GraphController( RenderView* render_view );

    ~GraphController();

private slots:
    void redraw_operation_tree();
    void currentItemChanged(QTreeWidgetItem* current,QTreeWidgetItem* previous);
    void checkVisibilityOperations(bool visible);

    void removeSelected();
    void removeHidden();

private:
    void setupGui();

    QList<Signal::pChainHead> heads;

    RenderView* render_view_;
    Sawe::Project* project_;

    QAction *actionToggleOperationsWindow;
    QDockWidget *operationsWindow;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout;
    QTreeWidget *operationsTree;
};

} // namespace Tools

#endif // GRAPHCONTROLLER_H
