#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>

namespace Ui
{
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
  virtual void keyPressEvent( QKeyEvent *e );
  virtual void keyReleaseEvent ( QKeyEvent * e );

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
