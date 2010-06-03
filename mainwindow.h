#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "displaywidget.h"
#include "tfr-filter.h"
#include <QTreeWidgetItem>

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

protected:
//    virtual void keyPressEvent( QKeyEvent *e );
//    virtual void keyReleaseEvent ( QKeyEvent * e );

public slots:
    void updateOperationsTree( Signal::pSource s);
    void updateLayerList( Tfr::pFilter f );
    void slotDbclkFilterItem(QListWidgetItem*);
    void slotNewSelection(QListWidgetItem*);
    void slotDeleteSelection(void);
    void slotToggleLayerWindow(bool);
    void slotClosedLayerWindow(bool visible);
    void slotToggleToolWindow(bool);
    void slotClosedToolWindow(bool visible);
    void slotToggleTimelineWindow(bool);
    void slotClosedTimelineWindow(bool visible);

signals:
    void sendCurrentSelection(int, bool);
    void sendRemoveItem(int);

private:
    class Ui_MainWindow *ui;
};

#endif // MAINWINDOW_H
