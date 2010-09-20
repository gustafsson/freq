#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "displaywidget.h"
#include "tfr/filter.h"
#include <QTreeWidgetItem>
#include <QComboBox>
#include <QAction>
#include <vector>
#include <QToolButton>

#ifdef Q_WS_MAC
void qt_mac_set_menubar_icons(bool enable);
#endif

class Ui_MainWindow;

namespace Ui {

class SaweMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    SaweMainWindow(const char* title, Sawe::Project* project, QWidget *parent = 0);
    ~SaweMainWindow();
    
    void connectLayerWindow(DisplayWidget *d);
    void setTimelineWidget( QWidget* );
    QWidget* getTimelineDock( );

    // These will be released by MainWindow::~QObject
    QGLWidget* displayWidget;
    QGLWidget* timelineWidget;

    //Signal::pWorkerCallback _displayWidgetCallback;
    //Signal::pWorkerCallback _timelineWidgetCallback;
protected:
    virtual void closeEvent(QCloseEvent *);
    struct ActionWindowPair
    {
        QWidget *w; QAction *a;
        ActionWindowPair(QWidget *iw, QAction *ia){w = iw; a = ia;}
    };
    std::vector<ActionWindowPair> controlledWindows;
    

public slots:
    void updateOperationsTree( Signal::pOperation s);
    //void updateLayerList( Signal::pOperation s );
    void slotDbclkFilterItem(QListWidgetItem*);
    void slotNewSelection(QListWidgetItem*);
    void slotDeleteSelection(void);
    void slotCheckWindowStates(bool);
    void slotCheckActionStates(bool);

signals:
    void sendCurrentSelection(int, bool);
    void sendRemoveItem(int);

private:
    Sawe::Project* project;
    Ui_MainWindow *ui;
    
    void add_widgets();
    void create_renderingwidgets();
    void connectActionToWindow(QAction *a, QWidget *b);
};


class QComboBoxAction: public QToolButton {
    Q_OBJECT
public:
    QComboBoxAction();
    void addActionItem( QAction* a );
    void decheckable(bool);
private slots:
    virtual void checkAction( QAction* a );
private:
    bool _decheckable;
};

/*class QDockArea: public QMainWindow {
    Q_OBJECT
public:
    explicit QMainWindow(QWidget *parent = 0, Qt::WindowFlags flags = 0);
    QDockArea();
};*/

} // namespace Ui

#endif // MAINWINDOW_H
