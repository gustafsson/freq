#ifndef SIGNAL_DAG_H
#define SIGNAL_DAG_H

// DAG, Directed_acyclic_graph
// http://en.wikipedia.org/wiki/Directed_acyclic_graph
//
// Describes a graph of signal processing nodes and operations on it.

#include "dagcommand.h"

#include <QWaitCondition>

namespace Signal {
namespace Dag {

/**
 * @brief The Dag class
 * Dag TODO
 * Assuming for now that the list of nodes is linear, and not at all a dag.
 * I.e a chain... the chain class was just removed...
 */
class SignalDag
{
public:
    typedef boost::shared_ptr<SignalDag> Ptr;

    SignalDag (Node::Ptr root);

    int sampleRate();
    int numberOfSamples();
    int numberOfChannels();

private:
    friend class DagHead;

    // The nodes build a linked bi-directional list.
    Node::Ptr tip_; // There's only one 'tip'
    const Node::Ptr root_; // 'root' is never changed during the lifetime of a dag

    // Each node can access the entire tree through children and parents.
    // There might be multiple tips. Connected to different points in the tree.

    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/) {
        TaskInfo ti("ChainHead::serialize");
        ar & BOOST_SERIALIZATION_NVP(root_); // Save it early to make it more readable.
        ar & BOOST_SERIALIZATION_NVP(tip_);
    }
};


} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_H
