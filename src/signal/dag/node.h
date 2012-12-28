#ifndef SIGNAL_DAG_NODE_H
#define SIGNAL_DAG_NODE_H

#include "signal/sinksource.h"
#include "signal/operationcache.h"

#include <boost/weak_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include <QReadWriteLock>

namespace Signal {
namespace Dag {

class Node: boost::noncopyable {
public:
    typedef boost::shared_ptr<Node> Ptr;

    class NodeData: boost::noncopyable
    {
    public:
        // (1) An Operation should not required any mutexes (exceptions occur
        //     for operations with shared resources, such as source data).
        // (2) An Operation might modify internal states while computing.
        // 1 and 2 implies that different threads must use different instances
        // of Operation.

        // OperationDesc should be cheap to copy and be sufficient to describe
        // a new instance of Operation.
        //
        // OperationDesc can change in a node. But only from a non-const
        // instance of Node. Such instances should only be available when the
        // Dag is in a write lock.

        // Asynchronous operations (i.e recordings and script operations) will
        // change their state during the lifetime of an Operation. To
        // invalidate samples in parent caches


        bool                    hidden() { return hidden_; }

        // Signal::SinkSource is thread safe
        Signal::SinkSource&     cache() { return cache_; }

        Signal::Operation::Ptr  operation(void* p, ComputingEngine* e=0);
        void                    removeOperation(void* p);

    private:
        typedef std::map<void*, Signal::Operation::Ptr> OperationMap;

        friend class Node;
        NodeData(Signal::OperationDesc::Ptr operationdesc, bool hidden);

        // Make sure operationDesc is only changed when Node is non-const
        const OperationDesc&        operationDesc() const;
        void                        operationDesc(Signal::OperationDesc::Ptr desc);

        QReadWriteLock              operations_lock_;
        OperationMap                operations_;
        Signal::OperationDesc::Ptr  desc_;
        bool                        hidden_;
        Signal::SinkSource          cache_;
    };


    // const interface
    NodeData&               data () const;
    QString                 name () const;
    const OperationDesc&    operationDesc() const;
    void                    invalidateSamples(const Intervals& I) const;

    // Meant to enforce constness for the entire DAG when reached from a const Node.
    const Node&             getChild (int i=0) const;
    int                     numChildren () const;


    // none-const interface
    explicit Node(Signal::OperationDesc::Ptr operationdesc, bool hidden = false);
    ~Node();

    // NodeData has non-const access from a const Node. But OperationDesc has
    // non-const access only from a non-const Node.
    void                    operationDesc(Signal::OperationDesc::Ptr);

    void                    setChild (Node::Ptr p, int i = 0);
    Node::Ptr               getChildPtr (int i=0);
    void                    detachParents ();
    std::set<Node*>         parents ();
    QReadWriteLock*         lock();

    static void test ();

private:
    QReadWriteLock lock_;

    // A const Node refers to the DAG references, but properties of the payload
    // data can be changed within the restrictions of NodeData.
    boost::scoped_ptr<NodeData> data_;

    std::vector<Node::Ptr> children_;
    std::set<Node*> parents_;

    friend class boost::serialization::access;
    Node() {} /// only used by deserialization
    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        TaskInfo ti("Serializing %s", name().toStdString ().c_str());

        Signal::OperationDesc::Ptr operationdesc;
        bool hidden;
        if (typename archive::is_saving())
        {
            operationdesc = operationDesc().copy ();
            hidden = data ().hidden ();
        }
        ar & BOOST_SERIALIZATION_NVP(operationdesc);
        ar & BOOST_SERIALIZATION_NVP(hidden);
        if (typename archive::is_loading())
        {
            data_.reset (new NodeData(operationdesc, hidden));
        }

        ar & BOOST_SERIALIZATION_NVP(children_);
    }
};


} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_NODE_H
