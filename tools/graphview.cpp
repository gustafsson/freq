#if 0

#include <boost/graph/adjacency_list.hpp>
#include <boost/unordered_set.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include "signal/operation-basic.h"

// TODO implement

struct TitleAndTooltip {
    std::string title, tooltip;
    void* ptrData;
};

typedef boost::adjacency_list<vecS,vecS, bidirectionalS, TitleAndTooltip> OperationGraph;
const OperationGraph::vertex_descriptor vertex_descriptor_null = (OperationGraph::vertex_descriptor)-1;

/*OperationGraph::vertex_descriptor populateGraph( Signal::pOperation f, OperationGraph& graph )
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

//    TODO ta bort
//    Tfr::FilterChain* filter_chain=0;

//    if ( 0 != (filter_chain = dynamic_cast<Tfr::FilterChain*>(f.get())))
//    {
//        title << "Chain #" << filter_chain->size() << "";
//        tooltip << "Chain contains " << filter_chain->size() << " subfilters";

//    } else

    if (Filters::EllipseFilter* c = dynamic_cast<Filters::EllipseFilter*>(f.get())) {
            float r = fabsf(c->_t1-c->_t2);
            title << "Ellipse [" << c->_t1-r << ", " << c->_t1 + r << "]";
            tooltip << "Ellipse p(" << c->_t1 << ", " << c->_f1 << "), "
                            << "r(" << r << ", " << fabsf(c->_f2-c->_f1) << "), "
                            << "area " << r*fabsf((c->_f1-c->_f2)*M_PI);

    } else if (Filters::SquareFilter* c = dynamic_cast<Filters::SquareFilter*>(f.get())) {
        title << "Square [" << c->_t1 << ", " << c->_t2 << "]";
        tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], "
                        << "f[" << c->_f1 << ", " << c->_f2 << "], "
                        << "area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2));
    } else if (f) {
        title << vartype(*f) << ", unknown filter";
    } else {
        title << "No filter";
    }

    TitleAndTooltip tat;
    tat.title = title.str();
    tat.tooltip = tooltip.str();
    tat.ptrData = f.get();

    OperationGraph::vertex_descriptor v = add_vertex(tat, graph);

    TaskTimer tt("%d %s", (unsigned)v, tat.title.c_str()); tt.suppressTiming();

//  TODO remove
//  if (filter_chain) {
//        foreach( Tfr::pFilter f, *filter_chain ) {
//            OperationGraph::vertex_descriptor c;
//            c = populateGraph( f, graph );

//            add_edge(v,c,graph);
//            tt.info("%d -> %d", v,c);
//        }
//    }

    return v;
}*/

OperationGraph::vertex_descriptor populateGraph( Signal::pOperation s, OperationGraph& graph )
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

    Signal::pOperation childSource;
    // Signal::pOperation childFilter;
    if (Adapters::MicrophoneRecorder* mic = dynamic_cast<Adapters::MicrophoneRecorder*>(s.get()))
    {
        title << "Microphone";
        tooltip << "Microphone recording, FS=" << mic->sample_rate();
    }
    else if (Adapters::Audiofile* file = dynamic_cast<Adapters::Audiofile*>(s.get()))
    {
        title << file->filename();
        tooltip << "Reading from file: " << file->filename();
    }
    else if ( Tfr::CwtFilter* filter_operation = dynamic_cast<Tfr::CwtFilter*>(s.get()))
    {
        filter_operation;
        title << "Filter " << vartype(*s);
        tooltip << "Filter Operation " << vartype(*s);

        // childFilter = filter_operation->filter();
    }
    else if (Signal::OperationSubOperations* sub_operations = dynamic_cast<Signal::OperationSubOperations*>(s.get()))
    {
        title << demangle( typeid(*s).name() );
        tooltip << "Composite operation " << vartype(*s);

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
        tooltip << "Source not further described: " << vartype(*s);
    }

    TitleAndTooltip tat;
    tat.title = title.str();
    tat.tooltip = tooltip.str();
    tat.ptrData = s.get();

    OperationGraph::vertex_descriptor v = add_vertex(tat, graph);

    TaskTimer tt("%d %s", (unsigned)v, tat.title.c_str()); tt.suppressTiming();

    if ( s->source() ) {
        OperationGraph::vertex_descriptor c;
        c = populateGraph( s->source(), graph );

        add_edge(v,c,graph);
        tt.info("%d -> %d", v,c);
    }

//    if (childFilter || childSource) {
//        OperationGraph::vertex_descriptor c =
//             (0!=childFilter)
//             ? populateGraph( childFilter, graph )
//             : populateGraph( childSource, graph );

//        add_edge(v,c,graph);
//        tt.info("%d -> %d", v,c);
//    }

    if (childSource) {
        OperationGraph::vertex_descriptor c =
             populateGraph( childSource, graph );

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

static void updateOperationsTree( OperationGraph::vertex_descriptor v, OperationGraph& graph, QTreeWidgetItem* w, OperationGraph::vertex_descriptor stop )
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

    foreach( const OperationGraph::vertex_descriptor& common, commons ) {
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

void SaweMainWindow::updateOperationsTree( Signal::pOperation s )
{
    TaskTimer tt("Updating operations tree");

    OperationGraph graph;

    OperationGraph::vertex_descriptor head = populateGraph( s, graph );

    ui->operationsTree->clear();

    QTreeWidgetItem* w = ui->operationsTree->invisibleRootItem();

    ::Ui::updateOperationsTree( head, graph, w, vertex_descriptor_null );
}

/*
void SaweMainWindow::updateLayerList( Signal::pOperation s )
{
    ui->layerWidget->clear();

    Tfr::FilterChain* filter_chain = dynamic_cast<Tfr::FilterChain*>(f.get());
    if (0 == filter_chain )
    {
        return;
    }

    foreach( Tfr::pFilter f, *filter_chain ) {
        stringstream title;
        stringstream tooltip;
        title << fixed << setprecision(1);
        tooltip << fixed << setprecision(2);

        if (Tfr::FilterChain *c = dynamic_cast<Tfr::FilterChain*>(f.get())) {
            title << "Chain #" << c->size() << "";
            tooltip << "Chain contains " << c->size() << " subfilters";

        } else if (Tfr::EllipseFilter* c = dynamic_cast<Tfr::EllipseFilter*>(f.get())) {
            float r = fabsf(c->_t1-c->_t2);
            title << "Ellipse [" << c->_t1-r << ", " << c->_t1 + r << "]";
            tooltip << "Ellipse p(" << c->_t1 << ", " << c->_f1 << "), "
                            << "r(" << r << ", " << fabsf(c->_f2-c->_f1) << "), "
                            << "area " << r*fabsf((c->_f1-c->_f2)*M_PI);

        } else if (Tfr::SquareFilter* c = dynamic_cast<Tfr::SquareFilter*>(f.get())) {
            title << "Square [" << c->_t1 << ", " << c->_t2 << "]";
            tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], "
                            << "f[" << c->_f1 << ", " << c->_f2 << "], "
                            << "area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2));

        }
//        else if (Tfr::SelectionFilter* c = dynamic_cast<Tfr::SelectionFilter>(f.get())) {
//            if (EllipseSelection* c = dynamic_cast<EllipseSelection>(c->selection)) {
//                title << "Ellipse, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)*M_PI) <<"";
//                tooltip << "Ellipse pos(" << c->_t1 << ", " << c->_f1 << "), radius(" << c->_t2-c->_t1 << ", " << c->_f2-c->_f1 << ")";

//            } else if (Tfr::SquareSelection* c = dynamic_cast<Tfr::SquareSelection>(c->selection)) {
//                title << "Square, area " << fabsf((c->_t1-c->_t2)*(c->_f1-c->_f2)) <<"";
//                tooltip << "Square t[" << c->_t1 << ", " << c->_t2 << "], f[" << c->_f1 << ", " << c->_f2 << "]";
//        }
        else {
            title << vartype(*f) << ", unknown attributes";
        }

        QListWidgetItem* itm = new QListWidgetItem( title.str().c_str(), ui->layerWidget, 0 );
        itm->setToolTip( tooltip.str().c_str() );
        itm->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsSelectable | Qt::ItemIsEnabled);
        itm->setCheckState( f->enabled? Qt::Checked:Qt::Unchecked);
        ui->layerWidget->addItem( itm );
    }

    printf("#####Updating: Layers!\n");
}
*/

#endif // if 0
