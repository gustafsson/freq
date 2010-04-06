#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include "displaywidget.h"
#include <boost/foreach.hpp>
#include <sstream>
#include <iomanip>

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

using namespace std;

MainWindow::MainWindow(const char* title, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
#ifdef Q_WS_MAC
    qt_mac_set_menubar_icons(false);
#endif
    ui->setupUi(this);
    this->setWindowTitle( title );
    void signalDbclkFilterItem(QListWidgetItem*);
    //connect(ui->layerWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(slotDbclkFilterItem(QListWidgetItem*)));
    connect(ui->layerWidget, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(slotNewSelection(QListWidgetItem*)));
    connect(ui->deleteFilterButton, SIGNAL(clicked(void)), this, SLOT(slotDeleteSelection(void)));
    connect(ui->actionToggleLayerWindow, SIGNAL(triggered(bool)), this, SLOT(slotToggleLayerWindow(bool)));
    connect(ui->layerWindow, SIGNAL(visibilityChanged(bool)), this, SLOT(slotClosedLayerWindow(bool)));
}

void MainWindow::slotToggleLayerWindow(bool a){
    if(!a) {
        ui->layerWindow->close();
    } else {
        ui->layerWindow->show();
    }
}
void MainWindow::slotClosedLayerWindow(bool visible){
    ui->actionToggleLayerWindow->setChecked(visible);
}

void MainWindow::slotDbclkFilterItem(QListWidgetItem */*item*/)
{
    //emit sendCurrentSelection(ui->layerWidget->row(item), );
}

void MainWindow::slotNewSelection(QListWidgetItem *item)
{
    int index = ui->layerWidget->row(item);
    if(index < 0){
        ui->deleteFilterButton->setEnabled(false);
        return;
    }else{
        ui->deleteFilterButton->setEnabled(true);
    }
    bool checked = false;
    if(ui->layerWidget->item(index)->checkState() == Qt::Checked){
        checked = true;
    }
    printf("Selecting new item: index:%d checked %d\n", index, checked);
    emit sendCurrentSelection(index, checked);
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
    connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(recieveCurrentSelection(int, bool)));
    connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(recieveFilterRemoval(int)));
    
    connect(this->ui->actionActivateSelection, SIGNAL(toggled(bool)), d, SLOT(recieveToggleSelection(bool)));
    connect(this->ui->actionActivateNavigation, SIGNAL(toggled(bool)), d, SLOT(recieveToggleNavigation(bool)));
    connect(this->ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), d, SLOT(recieveTogglePiano(bool)));
    connect(d, SIGNAL(setSelectionActive(bool)), this->ui->actionActivateSelection, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setNavigationActive(bool)), this->ui->actionActivateNavigation, SLOT(setChecked(bool)));
}

void MainWindow::updateLayerList(pTransform t)
{
    ui->layerWidget->clear();
    
    BOOST_FOREACH( pFilter f, t->filter_chain ) {
        stringstream title;
        stringstream tooltip;
        title << fixed << setprecision(1);
        tooltip << fixed << setprecision(2);

        if (FilterChain *c = dynamic_cast<FilterChain*>(f.get())) {
            title << "Chain #" << c->size() << "";
            tooltip << "Chain contains " << c->size() << " subfilters";

        } else if (EllipsFilter* c = dynamic_cast<EllipsFilter*>(f.get())) {
            float r = fabsf(c->_t1-c->_t2);
            title << "Ellips [" << c->_t1-r << ", " << c->_t1 + r << "]";
            tooltip << "Ellips p(" << c->_t1 << ", " << c->_f1 << "), "
                            << "r(" << r << ", " << fabsf(c->_f2-c->_f1) << "), "
                            << "area " << r*fabsf((c->_f1-c->_f2)*M_PI);

        } else if (SquareFilter* c = dynamic_cast<SquareFilter*>(f.get())) {
            title << "Square [" << c->_t1 << ", " << c->_t2 << "]";
            tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], "
                            << "f[" << c->_f1 << ", " << c->_f2 << "], "
                            << "area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2));

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
        itm->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsSelectable | Qt::ItemIsEnabled);
        itm->setCheckState( f->enabled? Qt::Checked:Qt::Unchecked);
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
