#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include "displaywidget.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::keyPressEvent( QKeyEvent *e )
{
    DisplayWidget::lastKey = e->key();
    switch( e->key() )
    {
    case Qt::Key_Escape:
        close();
    }
}

void MainWindow::keyReleaseEvent ( QKeyEvent *  )
{
    DisplayWidget::lastKey = 0;
}

