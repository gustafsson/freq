#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include "displaywidget.h"
#include <boost/foreach.hpp>
#include <sstream>
#include <iomanip>

using namespace std;

MainWindow::MainWindow(const char* title, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle( title );
    void signalDbclkFilterItem(QListWidgetItem*);
    //connect(ui->layerWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(slotDbclkFilterItem(QListWidgetItem*)));
    connect(ui->layerWidget, SIGNAL(currentRowChanged(int)), this, SLOT(slotNewSelection(int)));
    connect(ui->deleteFilterButton, SIGNAL(clicked(void)), this, SLOT(slotDeleteSelection(void)));
}

void MainWindow::slotDbclkFilterItem(QListWidgetItem *item)
{
    emit sendCurrentSelection(ui->layerWidget->row(item));
}

void MainWindow::slotNewSelection(int index)
{
    emit sendCurrentSelection(index);
}

void MainWindow::slotDeleteSelection(void)
{
    emit sendRemoveItem(ui->layerWidget->currentRow());
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::connectLayerWindow(DisplayWidget *d)
{
    connect(d, SIGNAL(filterChainUpdated(pTransform)), this, SLOT(updateLayerList(pTransform)));
    connect(this, SIGNAL(sendCurrentSelection(int)), d, SLOT(recieveCurrentSelection(int)));
    connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(recieveFilterRemoval(int)));
}

void MainWindow::updateLayerList(pTransform t)
{
    ui->layerWidget->clear();
    
    BOOST_FOREACH( pFilter f, t->filter_chain ) {
        stringstream title;
        stringstream tooltip;
        tooltip << fixed << setprecision(2) << " ";

        if (FilterChain *c = dynamic_cast<FilterChain*>(f.get())) {
            title << "Chain #" << c->size() << "";
            tooltip << "Chain contains " << c->size() << " subfilters";

        } else if (EllipsFilter* c = dynamic_cast<EllipsFilter*>(f.get())) {
            title << "Ellips, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)*M_PI) <<"";
            tooltip << "Ellips2 p(" << c->_t1 << ", " << c->_f1 << "), r(" << fabsf(c->_t2-c->_t1) << ", " << fabsf(c->_f2-c->_f1) << ")";

        } else if (SquareFilter* c = dynamic_cast<SquareFilter*>(f.get())) {
            title << "Square, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)) <<"";
            tooltip << "Square2 t[" << c->_t1 << ", " << c->_t2 << "], f[" << c->_f1 << ", " << c->_f2 << "]";

        }/* else if (SelectionFilter* c = dynamic_cast<SelectionFilter>(f.get())) {
            if (EllipsSelection* c = dynamic_cast<EllipsSelection>(c->selection)) {
                title << "Ellips, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)*M_PI) <<"";
                tooltip << "Ellips pos(" << c->_t1 << ", " << c->_f1 << "), radius(" << c->_t2-c->_t1 << ", " << c->_f2-c->_f1 << ")";

            } else if (SquareSelection* c = dynamic_cast<SquareSelection>(c->selection)) {
                title << "Square, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)) <<"";
                tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], f[" << c->_f1 << ", " << c->_f2 << "]";
        }*/
        else {
            title << typeid(*f).name() << ", unknown attributes";
        }

        QListWidgetItem* itm = new QListWidgetItem( title.str().c_str(), ui->layerWidget, 0 );
        itm->setToolTip( tooltip.str().c_str() );
        ui->layerWidget->addItem( itm );
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
