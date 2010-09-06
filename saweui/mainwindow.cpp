#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QKeyEvent>
#include <QSlider>
#include "displaywidget.h"
#include <boost/foreach.hpp>
#include <sstream>
#include <iomanip>
#include <demangle.h>
#include "tfr/filter.h"
#include "signal/operation-basic.h"
#include "signal/operation-composite.h"
#include "signal/microphonerecorder.h"
#include "signal/audiofile.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/unordered_set.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include "sawe/application.h"
#include "sawe/timelinewidget.h"
#include "saweui/propertiesselection.h"

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

using namespace std;
using namespace boost;

MainWindow::MainWindow(const char* title, QWidget *parent)
:   QMainWindow(parent),
    ui(new Ui_MainWindow)
{
#ifdef Q_WS_MAC
//    qt_mac_set_menubar_icons(false);
#endif
    ui->setupUi(this);
    QString qtitle = QString::fromLocal8Bit(title);
    this->setWindowTitle( qtitle );

    //connect(ui->layerWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(slotDbclkFilterItem(QListWidgetItem*)));
    connect(ui->layerWidget, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(slotNewSelection(QListWidgetItem*)));
    connect(ui->deleteFilterButton, SIGNAL(clicked(void)), this, SLOT(slotDeleteSelection(void)));
    connectActionToWindow(ui->actionToggleTopFilterWindow, ui->topFilterWindow);
    connectActionToWindow(ui->actionToggleOperationsWindow, ui->operationsWindow);
    connectActionToWindow(ui->actionToggleHistoryWindow, ui->historyWindow);
    connectActionToWindow(ui->actionToggleTimelineWindow, ui->dockWidgetTimeline);

    this->addDockWidget( Qt::RightDockWidgetArea, ui->toolPropertiesWindow );
    this->addDockWidget( Qt::RightDockWidgetArea, ui->operationsWindow );
    this->addDockWidget( Qt::RightDockWidgetArea, ui->topFilterWindow );
    this->addDockWidget( Qt::RightDockWidgetArea, ui->historyWindow );

    this->tabifyDockWidget(ui->operationsWindow, ui->topFilterWindow);
    this->tabifyDockWidget(ui->operationsWindow, ui->historyWindow);
    ui->topFilterWindow->raise();

    this->addToolBar( Qt::TopToolBarArea, ui->toolBarTool );
    this->addToolBar( Qt::TopToolBarArea, ui->toolBarOperation );
    this->addToolBar( Qt::BottomToolBarArea, ui->toolBarPlay );

    //new Saweui::PropertiesSelection( ui->toolPropertiesWindow );
    //ui->toolPropertiesWindow-
    new Saweui::PropertiesSelection( ui->frameProperties );

    connect(ui->actionToggleToolToolBox, SIGNAL(toggled(bool)), ui->toolBarTool, SLOT(setVisible(bool)));
    connect(ui->actionNew_recording, SIGNAL(triggered(bool)), Sawe::Application::global_ptr(), SLOT(slotNew_recording()));
    connect(ui->actionOpen, SIGNAL(triggered(bool)), Sawe::Application::global_ptr(), SLOT(slotOpen_file()));

    /*QComboBoxAction * qb = new QComboBoxAction();
    qb->addActionItem( ui->actionActivateSelection );
    qb->addActionItem( ui->actionActivateNavigation );
    ui->toolBarTool->addWidget( qb );*/

    /*ui->actionToolSelect->setEnabled( true );
    ui->actionActivateSelection->setEnabled( true );
    ui->actionSquareSelection->setEnabled( true );
    ui->actionSplineSelection->setEnabled( true );
    ui->actionPolygonSelection->setEnabled( true );
    ui->actionPeakSelection->setEnabled( true );*/

    ui->actionPeakSelection->setChecked( false );

    {   QComboBoxAction * qb = new QComboBoxAction();
        qb->addActionItem( ui->actionActivateSelection );
        qb->addActionItem( ui->actionSquareSelection );
        qb->addActionItem( ui->actionSplineSelection );
        qb->addActionItem( ui->actionPolygonSelection );
        qb->addActionItem( ui->actionPeakSelection );

        ui->toolBarTool->addWidget( qb );
    }

    {   QComboBoxAction * qb = new QComboBoxAction();
        qb->addActionItem( ui->actionAmplitudeBrush );
        qb->addActionItem( ui->actionAirbrush );
        qb->addActionItem( ui->actionSmoothBrush );
        qb->setEnabled( false );
        ui->toolBarTool->addWidget( qb );
    }

    {   QToolButton * tb = new QToolButton();

        tb->setDefaultAction( ui->actionToolSelect );

        ui->toolBarTool->addWidget( tb );
        connect( tb, SIGNAL(triggered(QAction *)), tb, SLOT(setDefaultAction(QAction *)));
    }

    {   QComboBoxAction * qb = new QComboBoxAction();
        qb->addActionItem( ui->actionToggle_hz_grid );
        qb->addActionItem( ui->actionToggle_piano_grid );
        ui->toolBarPlay->addWidget( qb );
    }

    {   QComboBoxAction * qb = new QComboBoxAction();
        qb->decheckable( false );
        qb->addActionItem( ui->actionSet_rainbow_colors );
        qb->addActionItem( ui->actionSet_grayscale );
        ui->toolBarPlay->addWidget( qb );
    }

    {   QComboBoxAction * qb = new QComboBoxAction();
        qb->addActionItem( ui->actionTransform_Cwt );
        qb->addActionItem( ui->actionTransform_Stft );
        qb->addActionItem( ui->actionTransform_Cwt_phase );
        qb->addActionItem( ui->actionTransform_Cwt_reassign );
        qb->addActionItem( ui->actionTransform_Cwt_ridge );
        qb->decheckable( false );
        ui->toolBarPlay->addWidget( qb );
    }
}

void MainWindow::slotCheckWindowStates(bool)
{
    unsigned int size = controlledWindows.size();
    for(unsigned int i = 0; i < size; i++)
    {
        controlledWindows[i].a->setChecked(!(controlledWindows[i].w->isHidden()));
    }
}
void MainWindow::slotCheckActionStates(bool)
{
    unsigned int size = controlledWindows.size();
    for(unsigned int i = 0; i < size; i++)
    {
        controlledWindows[i].w->setVisible(controlledWindows[i].a->isChecked());
    }
}

void MainWindow::connectActionToWindow(QAction *a, QWidget *b)
{
    connect(a, SIGNAL(toggled(bool)), this, SLOT(slotCheckActionStates(bool)));
    connect(b, SIGNAL(visibilityChanged(bool)), this, SLOT(slotCheckWindowStates(bool)));
    controlledWindows.push_back(ActionWindowPair(b, a));
}

MainWindow::~MainWindow()
{
    TaskTimer tt("~MainWindow");
    delete ui;
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

void MainWindow::connectLayerWindow(DisplayWidget *d)
{
    connect(d, SIGNAL(operationsUpdated(Signal::pSource)), this, SLOT(updateLayerList(Signal::pSource)));
    connect(d, SIGNAL(operationsUpdated(Signal::pSource)), this, SLOT(updateOperationsTree(Signal::pSource)));
    connect(this, SIGNAL(sendCurrentSelection(int, bool)), d, SLOT(receiveCurrentSelection(int, bool)));
    connect(this, SIGNAL(sendRemoveItem(int)), d, SLOT(receiveFilterRemoval(int)));
    
    connect(this->ui->actionActivateSelection, SIGNAL(toggled(bool)), d, SLOT(receiveToggleSelection(bool)));
    connect(this->ui->actionActivateNavigation, SIGNAL(toggled(bool)), d, SLOT(receiveToggleNavigation(bool)));
    connect(this->ui->actionActivateInfoTool, SIGNAL(toggled(bool)), d, SLOT(receiveToggleInfoTool(bool)));
    connect(this->ui->actionPlaySelection, SIGNAL(triggered()), d, SLOT(receivePlaySound()));
    connect(this->ui->actionFollowPlayMarker, SIGNAL(triggered(bool)), d, SLOT(receiveFollowPlayMarker(bool)));
    connect(this->ui->actionToggle_piano_grid, SIGNAL(toggled(bool)), d, SLOT(receiveTogglePiano(bool)));
    connect(this->ui->actionToggle_hz_grid, SIGNAL(toggled(bool)), d, SLOT(receiveToggleHz(bool)));
    connect(this->ui->actionActionAdd_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddSelection(bool)));
    connect(this->ui->actionActionRemove_selection, SIGNAL(triggered(bool)), d, SLOT(receiveAddClearSelection(bool)));
    connect(this->ui->actionCropSelection, SIGNAL(triggered()), d, SLOT(receiveCropSelection()));
    connect(this->ui->actionMoveSelection, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelection(bool)));
    connect(this->ui->actionMoveSelectionTime, SIGNAL(triggered(bool)), d, SLOT(receiveMoveSelectionInTime(bool)));
    connect(this->ui->actionMatlabOperation, SIGNAL(triggered(bool)), d, SLOT(receiveMatlabOperation(bool)));
    connect(this->ui->actionMatlabFilter, SIGNAL(triggered(bool)), d, SLOT(receiveMatlabFilter(bool)));
    connect(this->ui->actionTonalizeFilter, SIGNAL(triggered(bool)), d, SLOT(receiveTonalizeFilter(bool)));
    connect(this->ui->actionReassignFilter, SIGNAL(triggered(bool)), d, SLOT(receiveReassignFilter(bool)));
    connect(this->ui->actionRecord, SIGNAL(triggered(bool)), d, SLOT(receiveRecord(bool)));
    connect(d, SIGNAL(setSelectionActive(bool)), this->ui->actionActivateSelection, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setNavigationActive(bool)), this->ui->actionActivateNavigation, SLOT(setChecked(bool)));
    connect(d, SIGNAL(setInfoToolActive(bool)), this->ui->actionActivateInfoTool, SLOT(setChecked(bool)));
    connect(this->ui->actionSet_rainbow_colors, SIGNAL(triggered()), d, SLOT(receiveSetRainbowColors()));
    connect(this->ui->actionSet_grayscale, SIGNAL(triggered()), d, SLOT(receiveSetGrayscaleColors()));
    connect(this->ui->actionSet_heightlines, SIGNAL(toggled(bool)), d, SLOT(receiveSetHeightlines(bool)));

    connect(this->ui->actionTransform_Cwt, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt()));
    connect(this->ui->actionTransform_Stft, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Stft()));
    connect(this->ui->actionTransform_Cwt_phase, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt_phase()));
    connect(this->ui->actionTransform_Cwt_reassign, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt_reassign()));
    connect(this->ui->actionTransform_Cwt_ridge, SIGNAL(triggered()), d, SLOT(receiveSetTransform_Cwt_ridge()));

    {   QSlider * qs = new QSlider();
        qs->setOrientation( Qt::Horizontal );
        qs->setValue( 50 );
        qs->setToolTip( "Intensity level" );
        connect(qs, SIGNAL(valueChanged(int)), d, SLOT(receiveSetYScale(int)));

        ui->toolBarPlay->addWidget( qs );
    }

    {   QSlider * qs = new QSlider();
        qs->setOrientation( Qt::Horizontal );
        qs->setValue( 50 );
        qs->setToolTip( "Time/frequency resolution. If set higher than the middle, the audio reconstruction will be incorrect." );
        connect(qs, SIGNAL(valueChanged(int)), d, SLOT(receiveSetTimeFrequencyResolution(int)));

        ui->toolBarPlay->addWidget( qs );
    }


    ui->actionActivateNavigation->setChecked(true);

    updateOperationsTree( d->worker()->source() );
    d->getCwtFilter();

    if (d->isRecordSource()) {
        this->ui->actionRecord->setEnabled(true);
    } else {
        this->ui->actionRecord->setEnabled(false);
    }
}

void MainWindow::
        setTimelineWidget( QWidget* w )
{
    ui->dockWidgetTimeline->setWidget( w );
    ui->dockWidgetTimeline->show();
}

QWidget* MainWindow::
        getTimelineDock( )
{
    return ui->dockWidgetTimeline;
}

void MainWindow::
        closeEvent(QCloseEvent * e)
{
    // TODO add dialog asking user to save the project
    e->ignore();
    Sawe::Application::global_ptr()->slotClosed_window( this );
}



struct TitleAndTooltip {
    std::string title, tooltip;
    void* ptrData;
};

typedef boost::adjacency_list<vecS,vecS, bidirectionalS, TitleAndTooltip> OperationGraph;
const OperationGraph::vertex_descriptor vertex_descriptor_null = (OperationGraph::vertex_descriptor)-1;

OperationGraph::vertex_descriptor populateGraph( Tfr::pFilter f, OperationGraph& graph )
{
    // Search for vertex in graph
    OperationGraph::vertex_iterator i, end;
    for (tie(i,end) = vertices( graph ); i!=end; ++i) {
        if (f.get() == graph[*i].ptrData)
            return *i;
    }

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
    tat.ptrData = f.get();

    OperationGraph::vertex_descriptor v = add_vertex(tat, graph);

    TaskTimer tt("%d %s", (unsigned)v, tat.title.c_str()); tt.suppressTiming();

    if (filter_chain) {
        BOOST_FOREACH( Tfr::pFilter f, *filter_chain ) {
            OperationGraph::vertex_descriptor c;
            c = populateGraph( f, graph );

            add_edge(v,c,graph);
            tt.info("%d -> %d", v,c);
        }
    }

    return v;
}

OperationGraph::vertex_descriptor populateGraph( Signal::pSource s, OperationGraph& graph )
{
    // Search for vertex in graph
    OperationGraph::vertex_iterator i, end;
    for (tie(i,end) = vertices( graph ); i!=end; ++i) {
        if (s.get() == graph[*i].ptrData)
            return *i;
    }

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
    else if ( Tfr::CwtFilter* filter_operation = dynamic_cast<Tfr::CwtFilter*>(s.get()))
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
    tat.ptrData = s.get();

    OperationGraph::vertex_descriptor v = add_vertex(tat, graph);

    TaskTimer tt("%d %s", (unsigned)v, tat.title.c_str()); tt.suppressTiming();

    if ( Signal::Operation* op = dynamic_cast<Signal::Operation*>(s.get())) {
        OperationGraph::vertex_descriptor c;
        c = populateGraph( op->source(), graph );

        add_edge(v,c,graph);
        tt.info("%d -> %d", v,c);
    }

    if (childFilter || childSource) {
        OperationGraph::vertex_descriptor c =
             (0!=childFilter)
             ? populateGraph( childFilter, graph )
             : populateGraph( childSource, graph );

        add_edge(v,c,graph);
        tt.info("%d -> %d", v,c);
    }

    return v;
}

template<typename vertex_descriptor>
bool first_common_vertex_up(const vertex_descriptor& search, const vertex_descriptor& target, const vertex_descriptor& stop, OperationGraph graph)
{
    if (search == stop)
        return false;
    if (search == target)
        return true;

    // Find all parents of a
    pair<OperationGraph::inv_adjacency_iterator, OperationGraph::inv_adjacency_iterator>
        parents = boost::inv_adjacent_vertices(search, graph);

    for (OperationGraph::inv_adjacency_iterator c = parents.first; c!=parents.second; ++c)
    {
        if (first_common_vertex_up(*c, target, stop, graph))
            return true;
    }

    return false;
}

/**
  Assume that the graph doesn't contain any loops. Whatever is that called now again?
  Search for 'target' in the graph that can be reached from 'source' by going down and then up.
  */
template<typename vertex_descriptor>
vertex_descriptor first_common_vertex(const vertex_descriptor& source, const vertex_descriptor& target, const vertex_descriptor& stop, OperationGraph graph)
{
    if (first_common_vertex_up(source, target, stop, graph))
        return source;

    // Find all childrens of this graph
    pair<OperationGraph::adjacency_iterator, OperationGraph::adjacency_iterator>
            children = adjacent_vertices(source, graph);

    for (OperationGraph::adjacency_iterator c = children.first; c!=children.second; ++c)
    {
        vertex_descriptor v = first_common_vertex(*c, target, stop, graph);
        if (vertex_descriptor_null != v)
            return v;

    }
    return vertex_descriptor_null;
}

void updateOperationsTree( OperationGraph::vertex_descriptor v, OperationGraph& graph, QTreeWidgetItem* w, OperationGraph::vertex_descriptor stop )
{
    BOOST_ASSERT( w );

    const TitleAndTooltip& tat = graph[v];
    QTreeWidgetItem* child = new QTreeWidgetItem( 0 );
    child->setText(0, QString::fromLocal8Bit( tat.title.c_str() ));
    child->setToolTip(0, QString::fromLocal8Bit( tat.tooltip.c_str() ));
    child->setFlags( Qt::ItemIsSelectable | Qt::ItemIsEnabled );
    w->addChild( child );

    // Find all childrens of this vertex
    pair<OperationGraph::adjacency_iterator, OperationGraph::adjacency_iterator>
            children = adjacent_vertices(v, graph);

    // Count them
    // size_t numChildren = std::count( children.first, children.second );
    size_t numChildren = children.second - children.first;

    TaskTimer tt("%d %s (%d children)", (unsigned)v, tat.title.c_str(), numChildren);
    tt.suppressTiming();

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
        OperationGraph::adjacency_iterator d = c;
        d++;
        for (; d!=children.second; d++)
        {
            OperationGraph::vertex_descriptor common = first_common_vertex( *c, *d, v, graph );
            if (common != vertex_descriptor_null)
                commons.insert( common );
        }
    }

    BOOST_FOREACH( const OperationGraph::vertex_descriptor& common, commons ) {
        tt.info("Children converge at: %d %s", common, graph[common].title.c_str());
    }

    boost::unordered_set<OperationGraph::vertex_descriptor>::iterator itr;
    for (OperationGraph::adjacency_iterator c = children.first; c!=children.second; c++)
    {
        if (*c == stop) continue;

        OperationGraph::vertex_descriptor stop_vertex = vertex_descriptor_null;
        for (itr=commons.begin(); itr!=commons.end(); itr++ )
        {
            OperationGraph::vertex_descriptor common = first_common_vertex( *c, *itr, v, graph );
            if (common != vertex_descriptor_null) {
                stop_vertex = common;
                // TODO it feels like there could be a bug around here... think this through again and explain the procedure
            }
        }

        if (stop_vertex != vertex_descriptor_null)
            tt.info("Following children until: %d %s", stop_vertex, graph[stop_vertex].title.c_str());
        else
            tt.info("Following children to the end of the graph");

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

void MainWindow::updateLayerList( Signal::pSource s )
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
            title << demangle(typeid(*f).name()) << ", unknown attributes";
        }

        QListWidgetItem* itm = new QListWidgetItem( title.str().c_str(), ui->layerWidget, 0 );
        itm->setToolTip( tooltip.str().c_str() );
        itm->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsSelectable | Qt::ItemIsEnabled);
        itm->setCheckState( f->enabled? Qt::Checked:Qt::Unchecked);
        ui->layerWidget->addItem( itm );
    }
    
    printf("#####Updating: Layers!\n");
}
/*
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
}*/

QComboBoxAction::
        QComboBoxAction()
            :   _decheckable(true)
{
    connect( this, SIGNAL(triggered(QAction *)), this, SLOT(checkAction(QAction *)));
    this->setContextMenuPolicy( Qt::ActionsContextMenu );
}

void QComboBoxAction::
        addActionItem( QAction* a )
{
    addAction( a );
    if (0 == defaultAction())
        setDefaultAction(a);
}

void QComboBoxAction::
        decheckable( bool a )
{
    _decheckable = a;
    if (false == _decheckable)
        setChecked( true );
}

void QComboBoxAction::
        checkAction( QAction* a )
{
    if (a->isChecked())
    {
        QList<QAction*> l = actions();
        for (QList<QAction*>::iterator i = l.begin(); i!=l.end(); i++)
            if (*i != a)
                (*i)->setChecked( false );
    }

    if (false == _decheckable)
        a->setChecked( true );

    setDefaultAction( a );
}
