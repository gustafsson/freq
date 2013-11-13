#if 0
#ifndef GRAPHCONTROLLER_H
#define GRAPHCONTROLLER_H

#include <QObject>
#include <QTimer>
#include <QTreeWidget>

class QDockWidget;
class QWidget;
class QVBoxLayout;
class QTreeWidgetItem;
class QAction;
class QPushButton;
class QDropEvent;

namespace Sawe
{
    class Project;
}

namespace Tools
{

class RenderView;

class GraphTreeWidget: public QTreeWidget
{
    Q_OBJECT
public:
    GraphTreeWidget(QWidget*parent, Sawe::Project* project);

public slots:
    void currentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem*);

private:
    Sawe::Project* project_;
    QTreeWidgetItem* current_;

    virtual void dropEvent ( QDropEvent * event );
};

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
    void removeCaches();
    void updateContextMenu();

private:
    void setupGui();

    //QList<Signal::pChainHead> heads;

    RenderView* render_view_;
    Sawe::Project* project_;
    bool dontredraw_, removing_;

    QAction *actionToggleOperationsWindow;
    QDockWidget *operationsWindow;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout;
    QTreeWidget *operationsTree;
    QPushButton *removeSelectedButton;
    QTimer timerUpdateContextMenu;
};

} // namespace Tools

#endif // GRAPHCONTROLLER_H
#endif
