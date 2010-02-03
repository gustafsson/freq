#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include "displaywidget.h"

MainWindow::MainWindow(const char* title, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle( title );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::keyPressEvent( QKeyEvent *e )
{
    if (e->isAutoRepeat())
        return;

    switch( e->key() )
    {
    case Qt::Key_Escape:
        close();
    default:
        DisplayWidget::gDisplayWidget->keyPressEvent(e);
    }
}

void MainWindow::keyReleaseEvent ( QKeyEvent * e )
{
    if (e->isAutoRepeat())
        return;

    DisplayWidget::gDisplayWidget->keyReleaseEvent(e);
}
