#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "displaywidget.h"
#include <QKeyEvent>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setCentralWidget( new DisplayWidget());
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

