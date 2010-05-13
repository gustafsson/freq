#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include "displaywidget.h"
#include <boost/foreach.hpp>
#include <sstream>
#include <iomanip>
#include <demangle.h>
#include "tfr-filter.h"
#include "signal-operation-basic.h"
#include "signal-operation-composite.h"
#include "signal-microphonerecorder.h"
#include "signal-audiofile.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/unordered_set.hpp>
#include <boost/graph/adjacency_iterator.hpp>

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

using namespace std;
using namespace boost;

MainWindow::MainWindow(const char* title, QWidget *parent)
    : QMainWindow(parent), ui(new Ui_MainWindow)
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
    connect(ui->actionToggleToolWindow, SIGNAL(triggered(bool)), this, SLOT(slotToggleToolWindow(bool)));
    connect(ui->layerWindow, SIGNAL(visibilityChanged(bool)), this, SLOT(slotClosedLayerWindow(bool)));
    connect(ui->layerWindow, SIGNAL(visibilityChanged(bool)), this, SLOT(slotClosedLayerWindow(bool)));
}

void MainWindow::slotToggleLayerWindow(bool a){
    if(!a) {
        ui->layerWindow->close();
    } else {
        ui->layerWindow->show();
    }
}
void MainWindow::slotToggleToolWindow(bool a){
    if(!a) {
        ui->mainToolBar->close();
    } else {
        ui->mainToolBar->show();
    }
}
void MainWindow::slotClosedLayerWindow(bool visible){
    ui->actionToggleLayerWindow->setChecked(visible);
}
void MainWindow::slotClosedToolWindow(bool visible){
    ui->actionToggleToolWindow->setChecked(visible);
}

void MainWindow::slotDbclkFilterItem(QListWidgetItem * /*item*/)
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
    connect(d, SIGNAL(filterChainUpdated(Tfr::pFilter)), this, SLOT(updateLayerList(Tfr::pFilter)));
    connect(d, SIGNAL(operationsUpdated(Signal::pSource)), this, SLOT(updateOperationsTree(Signal::pSource)));
    connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(receiveCurrentSelection(int, bool)));
    connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(receiveFilterRemoval(int)));
    
    connect(this->ui->actionActivateSelection, SIGNAL(toggled(bool)), d, SLOT(receiveToggleSelection(bool)));
    connect(this->ui->actionActivateNavigation, SIGNAL(toggled(bool)), d, SLOT(receiveToggleNavigation(bool)));
    connect(this->ui->actionPlaySelection, SIGNAL(triggered()), d, SLOT(receivePlaySound()));
    connect(this->ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), d, SLOT(receiveTogglePiano(bool)));
    connect(this->ui->actionToggle_hz_grid, SIGNAL(toggled(bool)), d, SLOT(receiveToggleHz(bool)));
    connect(this->ui->actionActionAdd_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddSelection(bool)));
    connect(this->ui->actionActionRemove_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddClearSelection(bool)));
    connect(this->ui->actionCropSelection, SIGNAL(triggered()), d, SLOT(receiveCropSelection()));
    connect(this->ui->actionMoveSelection, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelection(bool)));
    connect(this->ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelectionInTime(bool)));
    connect(d, SIGNAL(setSelectionActive(bool)), this->ui->actionActivateSelection, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setNavigationActive(bool)), this->ui->actionActivateNavigation, SLOT(setChecked(bool)));

    ui->actionActivateNavigation->setChecked(true);
    d->setWorkerSource();
}

struct TitleAndTooltip {
    std::string title, tooltip;
};

typedef boost::adjacency_list<vecS,vecS, bidirectionalS, TitleAndTooltip> OperationGraph;
const OperationGraph::vertex_descriptor vertex_descriptor_null = (OperationGraph::vertex_descriptor)-1;

OperationGraph::vertex_descriptor populateGraph( Tfr::pFilter f, OperationGraph& graph )
{
    stringstream title;
    stringstream tooltip;
    title << fixed << setprecision(1);
    tooltip << fixed << setprecision(2);

    Tfr::FilterChain* filter_chain=0;

    if ( 0 != (filter_chain = dynamic_cast<Tfr::FilterChain*>(f.get())))
    {
        title << "Chain #" << filter_chain->size() << "";
        tooltip << "Chain contains " << filter_chain->size() << " subfilters";

    } else if (Tfr::EllipsFilter* c = dynamic_cast<Tfr::EllipsFilter*>(f.get())) {
            float r = fabsf(c->_t1-c->_t2);
            title << "Ellips [" << c->_t1-r << ", " << c->_t1 + r << "]";
            tooltip << "Ellips p(" << c->_t1 << ", " << c->_f1 << "), "
                            << "r(" << r << ", " << fabsf(c->_f2-c->_f1) << "), "
                            << "area " << r*fabsf((c->_f1-c->_f2)*M_PI);

    } else if (Tfr::SquareFilter* c = dynamic_cast<Tfr::SquareFilter*>(f.get())) {
        title << "Square [" << c->_t1 << ", " << c->_t2 << "]";
        tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], "
                        << "f[" << c->_f1 << ", " << c->_f2 << "], "
                        << "area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2));
    } else if (f) {
        title << demangle(typeid(*f).name()) << ", unknown filter";
    } else {
        title << "No filter";
    }

    TitleAndTooltip tat;
    tat.title = title.str();
    tat.tooltip = tooltip.str();

    OperationGraph::vertex_descriptor v = add_vertex(tat, graph);

    if (filter_chain) {
        BOOST_FOREACH( Tfr::pFilter f, *filter_chain ) {
            OperationGraph::vertex_descriptor c;
            c = populateGraph( f, graph );

            add_edge(v,c,graph);
        }
    }

    return v;
}

OperationGraph::vertex_descriptor populateGraph( Signal::pSource s, OperationGraph& graph )
{
    stringstream title;
    stringstream tooltip;
    title << fixed << setprecision(1);
    tooltip << fixed << setprecision(2);

    Signal::pSource childSource;
    Tfr::pFilter childFilter;
    if (Signal::MicrophoneRecorder* mic = dynamic_cast<Signal::MicrophoneRecorder*>(s.get()))
    {
        title << "Microphone";
        tooltip << "Microphone recording, FS=" << mic->sample_rate();
    }
    else if (Signal::Audiofile* file = dynamic_cast<Signal::Audiofile*>(s.get()))
    {
        title << file->filename();
        tooltip << "Reading from file: " << file->filename();
    }
    else if ( Signal::FilterOperation* filter_operation = dynamic_cast<Signal::FilterOperation*>(s.get()))
    {
        title << "Filter";
        tooltip << "Filter Operation";

        childFilter = filter_operation->filter();
    }
    else if (Signal::OperationSubOperations* sub_operations = dynamic_cast<Signal::OperationSubOperations*>(s.get()))
    {
        title << demangle( typeid(*s).name() );
        tooltip << "Composite operation" << demangle(typeid(*s).name());

        childSource = sub_operations->subSource();
    }
    else if (Signal::OperationSuperposition* super = dynamic_cast<Signal::OperationSuperposition*>(s.get()))
    {
        title << "Sum";
        tooltip << "Sum of two signals by superpositioning";

        childSource = super->source2();
    }
    else
    {
        title << demangle( typeid(*s).name() );
        tooltip << "Source not further described: " << demangle(typeid(*s).name());
    }

    TitleAndTooltip tat;
    tat.title = title.str();
    tat.tooltip = tooltip.str();

    OperationGraph::vertex_descriptor v = add_vertex(tat, graph);

    if ( Signal::Operation* op = dynamic_cast<Signal::Operation*>(s.get())) {
        OperationGraph::vertex_descriptor c;
        c = populateGraph( op->source(), graph );

        add_edge(v,c,graph);
    }

    if (childFilter || childSource) {
        OperationGraph::vertex_descriptor c =
             (0!=childFilter)
             ? populateGraph( childFilter, graph )
             : populateGraph( childSource, graph );

        add_edge(v,c,graph);
    }

    return v;
}

template<typename vertex_descriptor>
bool first_common_vertex_up(const vertex_descriptor& search, const vertex_descriptor& stop, const vertex_descriptor& target, OperationGraph graph)
{
    if (search == stop)
        return false;
    if (search == target)
        return true;

    // Find all parents of a
    pair<OperationGraph::inv_adjacency_iterator, OperationGraph::inv_adjacency_iterator>
        children = boost::inv_adjacent_vertices(search, graph);

    for (OperationGraph::inv_adjacency_iterator c = children.first; c!=children.second; ++c)
    {
        if (first_common_vertex_up(*c, stop, target, graph))
            return true;
    }

    return false;
}

/**
  Assume that the graph doesn't contain any loops. Whatever is that called now again?
  Search for 'target' in the graph that can be reached from 'source' by going down and then up.
  */
template<typename vertex_descriptor>
vertex_descriptor first_common_vertex(const vertex_descriptor& source, const vertex_descriptor& target, OperationGraph graph)
{
    // Find all childrens of this graph
    pair<OperationGraph::adjacency_iterator, OperationGraph::adjacency_iterator>
            children = adjacent_vertices(source, graph);

    for (OperationGraph::adjacency_iterator c = children.first; c!=children.second; ++c)
    {
        if (first_common_vertex_up(*c, source, target, graph))
            return *c;
        first_common_vertex(*c, target, graph);
    }
    return vertex_descriptor_null;
}
/*OperationGraph::adjacency_iterator
template<typename Iterator>
typename std::iterator_traits<Iterator>::difference_type
mycount(Iterator first, Iterator second)
{
    for ()
}*/

void updateOperationsTree( OperationGraph::vertex_descriptor v, OperationGraph graph, QTreeWidgetItem* w, OperationGraph::vertex_descriptor stop )
{
    BOOST_ASSERT( w );

    QTreeWidgetItem* child = new QTreeWidgetItem( 0 );
    child->setText(0, QString::fromStdString( graph[v].title ));
    child->setToolTip(0, QString::fromStdString( graph[v].tooltip ));
    child->setFlags( Qt::ItemIsSelectable | Qt::ItemIsEnabled );
    w->addChild( child );

    // Find all childrens of this graph
    pair<OperationGraph::adjacency_iterator, OperationGraph::adjacency_iterator>
            children = adjacent_vertices(v, graph);

    // Count them
    // size_t numChildren = std::count( children.first, children.second );
    size_t numChildren = children.second - children.first;

    // If only one, do nothing special
    if (1==numChildren) {
        if (*children.first != stop)
            updateOperationsTree( *children.first, graph, w, stop );
        return;
    }

    // If more than one however:
    //  1. count number of disconnected subgraphcs (children usually converge to have some
    //     common node further down)
    //  2. for each subgraph
    //  3. create a subtree with the title subgraph (the first subgraph is promoted to stay
    //     on the same level with 'w' as parent)
    //  4. do 1 again
    //
    //  1 search all paths and look for the first common node (first connection point)
    //  2 if a child is found who is not connected  doesn't have .... blaj
    boost::unordered_set<OperationGraph::vertex_descriptor> commons;
    for (OperationGraph::adjacency_iterator c = children.first; c!=children.second; c++)
    {
        if (*c == stop) continue;

        OperationGraph::adjacency_iterator d = c;
        d++;
        for (; d!=children.second; d++)
        {
            if (*d == stop) continue;
            OperationGraph::vertex_descriptor common = first_common_vertex( *c, *d, graph );
            commons.insert( common );
        }
    }

    boost::unordered_set<OperationGraph::vertex_descriptor>::iterator itr;
    for (OperationGraph::adjacency_iterator c = children.first; c!=children.second; c++)
    {
        if (*c == stop) continue;

        OperationGraph::vertex_descriptor stop_vertex = vertex_descriptor_null;
        for (itr=commons.begin(); itr!=commons.end(); itr++ )
        {
            OperationGraph::vertex_descriptor common = first_common_vertex( *c, *itr, graph );
            if (common != vertex_descriptor_null)
                stop_vertex = common;
        }

        if (c == children.first) {
            updateOperationsTree( *c, graph, w, stop_vertex );
        } else {
            updateOperationsTree( *c, graph, child, stop_vertex );
        }
    }
}

void MainWindow::updateOperationsTree( Signal::pSource s )
{
    TaskTimer tt("Updating operations tree");

    OperationGraph graph;

    OperationGraph::vertex_descriptor head = populateGraph( s, graph );

    ui->operationsTree->clear();

    QTreeWidgetItem* w = ui->operationsTree->invisibleRootItem();
    ::updateOperationsTree( head, graph, w, vertex_descriptor_null );
}

void MainWindow::updateLayerList( Tfr::pFilter f )
{
    ui->layerWidget->clear();

    Tfr::FilterChain* filter_chain = dynamic_cast<Tfr::FilterChain*>(f.get());
    if (0 == filter_chain )
    {
        return;
    }

    BOOST_FOREACH( Tfr::pFilter f, *filter_chain ) {
        stringstream title;
        stringstream tooltip;
        title << fixed << setprecision(1);
        tooltip << fixed << setprecision(2);

        if (Tfr::FilterChain *c = dynamic_cast<Tfr::FilterChain*>(f.get())) {
            title << "Chain #" << c->size() << "";
            tooltip << "Chain contains " << c->size() << " subfilters";

        } else if (Tfr::EllipsFilter* c = dynamic_cast<Tfr::EllipsFilter*>(f.get())) {
            float r = fabsf(c->_t1-c->_t2);
            title << "Ellips [" << c->_t1-r << ", " << c->_t1 + r << "]";
            tooltip << "Ellips p(" << c->_t1 << ", " << c->_f1 << "), "
                            << "r(" << r << ", " << fabsf(c->_f2-c->_f1) << "), "
                            << "area " << r*fabsf((c->_f1-c->_f2)*M_PI);

        } else if (Tfr::SquareFilter* c = dynamic_cast<Tfr::SquareFilter*>(f.get())) {
            title << "Square [" << c->_t1 << ", " << c->_t2 << "]";
            tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], "
                            << "f[" << c->_f1 << ", " << c->_f2 << "], "
                            << "area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2));

        }/* else if (Tfr::SelectionFilter* c = dynamic_cast<Tfr::SelectionFilter>(f.get())) {
            if (EllipsSelection* c = dynamic_cast<EllipsSelection>(c->selection)) {
                title << "Ellips, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)*M_PI) <<"";
                tooltip << "Ellips pos(" << c->_t1 << ", " << c->_f1 << "), radius(" << c->_t2-c->_t1 << ", " << c->_f2-c->_f1 << ")";

            } else if (Tfr::SquareSelection* c = dynamic_cast<Tfr::SquareSelection>(c->selection)) {
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
