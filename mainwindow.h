#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "displaywidget.h"
#include "spectrogram-renderer.h"

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
    void slotNewSelection(int);

signals:
    void sendCurrentSelection(int);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
