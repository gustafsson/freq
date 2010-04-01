#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "displaywidget.h"
#include "spectrogram-renderer.h"

#ifdef Q_WS_MAC
void qt_mac_set_menubar_icons(bool enable);
#endif

namespace Ui
{
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(const char* title, QWidget *parent = 0);
    ~MainWindow();
    
    void connectLayerWindow(DisplayWidget *d);

protected:
    virtual void keyPressEvent( QKeyEvent *e );
    virtual void keyReleaseEvent ( QKeyEvent * e );

public slots:
    void updateLayerList(pTransform t);
    void slotDbclkFilterItem(QListWidgetItem*);
    void slotNewSelection(QListWidgetItem*);
    void slotDeleteSelection(void);

signals:
    void sendCurrentSelection(int, bool);
    void sendRemoveItem(int);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
