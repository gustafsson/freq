#ifndef SIGNAL_DAG_NODE_H
#define SIGNAL_DAG_NODE_H

#include "signal/sinksource.h"
#include "signal/operationcache.h"
#include "volatileptr.h"
#include <boost/weak_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <QReadWriteLock>

namespace Signal {
namespace Dag {


class Node: public boost::enable_shared_from_this<volatile Node>, public VolatilePtr<Node> {
public:

    /**
     * @brief Node and NodeData could have separate locks as consistency
     * within NodeData is managed independently of Node.
     */
    class NodeData
    {
    public:
        typedef boost::shared_ptr<NodeData> Ptr;

        // (1) Operations An Operation should not required any mutexes for normal fast
        //     processing. This (exceptions occur
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

        NodeData(Signal::OperationDesc::Ptr operationdesc);

        const OperationDesc&        operationDesc() const;
        void                        operationDesc(Signal::OperationDesc::Ptr desc);

        Signal::Operation::Ptr      operation(void* p, ComputingEngine* e=0);
        void                        removeOperation(void* p);

        Signal::Intervals           current_processing; // Expected output
        Signal::Intervals           intervals_to_invalidate;
        Signal::SinkSource          cache;

    private:
        typedef std::map<void*, Signal::Operation::Ptr> OperationMap;

        OperationMap                operations_;
        Signal::OperationDesc::Ptr  desc_;
    };


    explicit Node(Signal::OperationDesc::Ptr operationdesc);
    ~Node();


    NodeData*               data ();
    const NodeData*         data () const;
    QString                 name () const;


    void                    invalidateSamples(Intervals I) volatile;
    void                    startSampleProcessing(Interval expected_output) volatile;
    void                    validateSamples(Signal::pBuffer output) volatile;


    void                    setChild (Node::Ptr p, int i = 0) volatile;
    Ptr                     getChild (int i=0) volatile;
    ConstPtr                getChild (int i=0) const volatile;
    int                     numChildren () const;
    std::set<Ptr>           parents ();
    std::set<ConstPtr>      parents () const;

    static void test ();

private:
    NodeData::Ptr data_;

    std::vector<Ptr> children_;
    std::set<WeakPtr> parents_;

    friend class boost::serialization::access;
    Node() {} /// only used by deserialization
    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        TaskInfo ti("Serializing %s", name().toStdString ().c_str());

        Signal::OperationDesc::Ptr operationdesc;
        if (typename archive::is_saving())
        {
            operationdesc = Node::ReadPtr(this)->data()->operationDesc().copy ();
        }
        ar & BOOST_SERIALIZATION_NVP(operationdesc);
        if (typename archive::is_loading())
        {
            data_.reset (new NodeData(operationdesc));
        }

        ar & BOOST_SERIALIZATION_NVP(children_);
    }
};


} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_NODE_H
