#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "displaywidget.h"
#include "tfr-filter.h"
#include <QTreeWidgetItem>
#include <QComboBox>
#include <QAction>
#include <vector>
#include <QToolButton>

#ifdef Q_WS_MAC
void qt_mac_set_menubar_icons(bool enable);
#endif


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(const char* title, QWidget *parent = 0);
    ~MainWindow();
    
    void connectLayerWindow(DisplayWidget *d);
    void setTimelineWidget( QWidget* );
    QWidget* getTimelineDock( );

protected:
//    virtual void keyPressEvent( QKeyEvent *e );
//    virtual void keyReleaseEvent ( QKeyEvent * e );
    virtual void closeEvent(QCloseEvent *);
    struct ActionWindowPair
    {
        QWidget *w; QAction *a;
        ActionWindowPair(QWidget *iw, QAction *ia){w = iw; a = ia;}
    };
    std::vector<ActionWindowPair> controlledWindows;
    

public slots:
    void updateOperationsTree( Signal::pSource s);
    void updateLayerList( Tfr::pFilter f );
    void slotDbclkFilterItem(QListWidgetItem*);
    void slotNewSelection(QListWidgetItem*);
    void slotDeleteSelection(void);
    void slotCheckWindowStates(bool);
    void slotCheckActionStates(bool);

signals:
    void sendCurrentSelection(int, bool);
    void sendRemoveItem(int);

private:
    class Ui_MainWindow *ui;
    
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

#endif // MAINWINDOW_H
