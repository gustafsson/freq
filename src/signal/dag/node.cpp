#include "node.h"

namespace Signal {
namespace Dag {


Node::NodeData::
NodeData(Signal::OperationDesc::Ptr operationdesc, bool hidden)
    :
      desc_(operationdesc),
      hidden_(hidden),
      cache_(0)
{
}

Signal::Operation::Ptr Node::NodeData::
        operation(void* p, ComputingEngine* e)
{
    QWriteLocker l(&operations_lock_);
    Signal::Operation::Ptr& r = operations_[p];
    if (!r)
        r = desc_->createOperation (e);
    return r;
}

void Node::NodeData::
        removeOperation(void* p)
{
    QWriteLocker l(&operations_lock_);
    operations_.erase(p);
}


const OperationDesc& Node::NodeData::
        operationDesc() const
{
    return *desc_;
}


void Node::NodeData::
        operationDesc(Signal::OperationDesc::Ptr desc)
{
    QWriteLocker l(&operations_lock_);
    for (OperationMap::iterator i = operations_.begin();
         i != operations_.end();
         ++i)
    {
        i->second = desc->recreateOperation (i->second);
    }
}


Node::
        Node(Signal::OperationDesc::Ptr operationdesc, bool hidden)
    :
      data_(new NodeData(operationdesc->copy(), hidden))
{
    children_.resize (operationdesc->getNumberOfSources());
}


Node::
        ~Node()
{
    // Detach self from children. This effectively delets any child that
    // doesn't have any other reference.
    for (int j=0; j<numChildren(); ++j)
        setChild(Node::Ptr(), j);

    // Doesn't need to detach self from parents. Since parents keep a smart
    // pointer reference to this the destructor wouldn't be called if there
    // would be any parent to detach.
}


void Node::
        detachParents()
{
    // Detach self from parents
    // Replace self with the first child among the parents.
    // This will invalidate parents if 'newChildNode' is not set.
    Ptr newChildNode;
    if (numChildren()>0)
        newChildNode = children_[0];

    for (std::set<Node*>::iterator itr = parents_.begin ();
         itr != parents_.end ();
         itr++)
    {
        Node* p = *itr;
        for (int j=0; j<p->numChildren(); ++j)
        {
            if (&p->getChild(j) == this)
                p->setChild(newChildNode, j);
        }
    }
    // Parents should be detached here.
    EXCEPTION_ASSERT(parents_.empty ());
}


void Node::
        setChild(Node::Ptr p, int i)
{
    if (children_[i])
        children_[i]->parents_.erase(this);

    children_[i] = p;

    if (children_[i])
        children_[i]->parents_.insert(this);
}


Node::Ptr Node::
        getChildPtr (int i)
{
    return children_[i];
}


const Node& Node::
        getChild( int i ) const
{
    return *children_[i];
}


int Node::
        numChildren() const
{
    return children_.size ();
}


Node::NodeData& Node::
        data() const
{
    return *data_;
}


QString Node::
        name() const
{
    return operationDesc().toString ();
}


const OperationDesc& Node::
        operationDesc() const
{
    return data().operationDesc();
}


void Node::
        invalidate_samples(const Intervals& I) const
{
    data().cache ().invalidate_samples (I);

    for (std::set<Node*>::iterator i = parents_.begin ();
         i != parents_.end ();
         ++i)
    {
        (*i)->invalidate_samples (I);
    }
}


void Node::
        operationDesc(Signal::OperationDesc::Ptr p)
{
    data().operationDesc(p);
}


const std::set<Node*>& Node::
        parents()
{
    return parents_;
}


} // namespace Dag
} // namespace Signal

#include "signal/buffersource.h"
#include "boost/format.hpp"

using namespace Signal;
using namespace boost;

namespace Signal {
namespace Dag {

class SimpleOperationDesc: public Signal::OperationDesc {
public:
    SimpleOperationDesc(Signal::Operation::Ptr p) : p(p) {}

    Signal::Operation::Ptr createOperation(ComputingEngine* engine) const {
        engine=engine;
        return p;
    }

    OperationDesc::Ptr copy() const { return Signal::OperationDesc::Ptr(new SimpleOperationDesc(p)); }
    QString toString() const { return "SimpleOperationDesc"; }

    bool operator==(const OperationDesc& d) const {
        if (SimpleOperationDesc const* s = dynamic_cast<SimpleOperationDesc const*>(&d))
            return s->p == p;
        return false;
    }

private:
    Signal::Operation::Ptr p;
};

void Node::
        test()
{


    pBuffer b(new Buffer(Interval(0, 10), 1, 1));
    Signal::Operation::Ptr o( new BufferSource(b));
    Node n(Signal::OperationDesc::Ptr(new SimpleOperationDesc(o)), true);
    EXCEPTION_ASSERTX(n.data ().hidden () == true, str(format("n.hidden () = %d") % n.data ().hidden ()));


}
} // namespace Dag
} // namespace Signal
