#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include "displaywidget.h"
#include <boost/foreach.hpp>

MainWindow::MainWindow(const char* title, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle( title );
    void signalDbclkFilterItem(QListWidgetItem*);
    //connect(ui->layerWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(slotDbclkFilterItem(QListWidgetItem*)));
    connect(ui->layerWidget, SIGNAL(currentRowChanged(int)), this, SLOT(slotNewSelection(int)));
}

void MainWindow::slotDbclkFilterItem(QListWidgetItem *item)
{
    emit sendCurrentSelection(ui->layerWidget->row(item));
}

void MainWindow::slotNewSelection(int index)
{
    emit sendCurrentSelection(index);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::connectLayerWindow(DisplayWidget *d)
{
    connect(d, SIGNAL(filterChainUpdated(pTransform)), this, SLOT(updateLayerList(pTransform)));
    connect(this, SIGNAL(sendCurrentSelection(int)), d, SLOT(recieveCurrentSelection(int)));
}

void MainWindow::updateLayerList(pTransform t)
{
    ui->layerWidget->clear();
    
    BOOST_FOREACH( pFilter f, t->filter_chain ) {
        ui->layerWidget->addItem("Circle");
    }
    
    printf("#####Updating: Layers!\n");
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
