#include "node.h"
#include "test/operationmockups.h"

#include "tools/support/timer.h"

namespace Signal {
namespace Dag {


Node::NodeData::
        NodeData(Signal::OperationDesc::Ptr operationdesc)
    :
      desc_(operationdesc)
{
}


Signal::Operation::Ptr Node::NodeData::
        operation(ComputingEngine* e)
{
    Signal::Operation::Ptr& r = operations_[e];
    if (!r)
        r = desc_->createOperation (e);
    return r;
}


void Node::NodeData::
        removeOperation(ComputingEngine* p)
{
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
    for (OperationMap::iterator i = operations_.begin();
         i != operations_.end();
         ++i)
    {
        i->second = desc->recreateOperation (i->second);
    }
}


Node::
        Node(Signal::OperationDesc::Ptr operationdesc)
    :
      data_(new NodeData(operationdesc->copy()))
{
    children_.resize (operationdesc->getNumberOfSources());
}


Node::
        ~Node()
{
    // Detach self from children.
    for (int j=0; j<numChildren(); ++j)
    {
        // This detachment isn't strictly necessary as Node::parents()
        // makes sure only valid parents are returned anyways. But I
        // guess it counts as good housekeeping.

        // Can't use 'setChild(Node::Ptr(), j)' because setChild relies
        // on 'shared_from_this' which isn't available in the destructor.

        if (!children_[j])
            continue;

        WritePtr c(children_[j]);

        for (std::set<WeakPtr>::iterator i=c->parents_.begin();
             i != c->parents_.end ();
             ++i)
        {
            if (!i->lock())
            {
                // Assume that 'i' was the one we're removing now.
                c->parents_.erase(i);
                break;
            }
        }
    }

    // Doesn't need to detach self from parents. Since parents keep a
    // shared_ptr (Node::Ptr) reference to this, the destructor wouldn't be
    // called if there was any parent to detach here.
    // Parents should be detached here.
    //EXCEPTION_ASSERT(parents_.empty ());

    // This effectively delets any instance of a child Node that does not
    // have any other references through shared_ptr (Node::Ptr).
    // children_.clear (); // implicit by std::vector::~vector
}


void Node::
        setChild(Node::Ptr newchild, int i) volatile
{
    // Avoid locking multiple objects at once as it may cause deadlocks.
    // But it is required here to keep consistency during the transaction.
    WritePtr self(this);

    Ptr oldchild = self->children_[i];
    self->children_[i] = newchild;

    WeakPtr weakself = self->shared_from_this();

    if (oldchild)
        WritePtr(oldchild)->parents_.erase(weakself);
    if (newchild)
        WritePtr(newchild)->parents_.insert(weakself);
}


Node::Ptr Node::
        getChild (int i) volatile
{
    ReadPtr r(this);
    return r->children_[i];
}


Node::ConstPtr Node::
        getChild( int i ) const volatile
{
    ReadPtr r(this);
    return r->children_[i];
}


int Node::
        numChildren() const
{
    return children_.size ();
}


const Node::NodeData* Node::
        data () const
{
    return data_.get ();
}


Node::NodeData* Node::
        data ()
{
    return data_.get ();
}


QString Node::
        name() const
{
    return data()->operationDesc().toString ();
}


void Node::
        invalidateSamples(Intervals I) volatile
{
    {
        Node::WritePtr w(this);
        Node::NodeData* data = w->data();
        data->intervals_to_invalidate |= data->current_processing & I;
        I -= data->current_processing;
        if (!I)
            return;
        data->cache.invalidate_samples (I);
    }

    invalidateParentSamples (I);
}


void Node::
        invalidateParentSamples(Intervals I) volatile
{
    std::set<Node::Ptr> P = Node::WritePtr (this)->parents();

    for (std::set<Node::Ptr>::iterator i = P.begin ();
         i != P.end ();
         ++i)
    {
        (*i)->invalidateSamples (I);
    }
}


bool Node::
        startSampleProcessing(Interval expected_output) volatile
{
    Node::WritePtr self(this);
    if (self->data()->current_processing & expected_output)
    {
        // If someone else has just started working on this.
        return false;
    }

    self->data()->current_processing |= expected_output;
    return true;
}


void Node::
        validateSamples(Signal::pBuffer output) volatile
{
    Interval I = output->getInterval ();

    {
        Node::WritePtr w(this);
        Node::NodeData* data = w->data();

        if (data->intervals_to_invalidate & I)
        {
            // discard results
        }
        else
        {
            // Merge results
            data->cache.put (output);
        }

        // Done processing
        data->current_processing -= I;
        // Doesn't need to invalidate parents as they shouldn't have been
        // validated either. And if the parents somehow aren't invalid this
        // result might still not be needed. So there's no imminent need to
        // call invalidateSamples which will trigger pretty much everything.
        data->cache.invalidate_samples (data->intervals_to_invalidate);
        data->intervals_to_invalidate.clear ();

    }

    invalidateParentSamples (I);
}


std::set<Node::Ptr> Node::
        parents()
{
    std::set<Ptr> P;
    for (std::set<WeakPtr>::iterator i = parents_.begin ();
         i != parents_.end ();
         ++i)
    {
        Ptr p = i->lock();
        if (p)
            P.insert (p);
    }

    return P;
}


std::set<Node::ConstPtr> Node::
        parents() const
{
    std::set<ConstPtr> P;
    for (std::set<WeakPtr>::iterator i = parents_.begin ();
         i != parents_.end ();
         ++i)
    {
        ConstPtr p = i->lock();
        if (p)
            P.insert (p);
    }

    return P;
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

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const {
        if (expectedOutput)
            *expectedOutput = I;
        return I;
    }

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
    Signal::OperationDesc::Ptr desc( new BufferSource(b));
    Node::Ptr np(new Node(desc));
    {
        Node::ReadPtr n(np);
        EXCEPTION_ASSERTX(n->data ()->operationDesc () == BufferSource(b),
                          str(format("n.hidden () = %1%") % n->data ()->operationDesc ()));
    }
    Signal::OperationDesc::Ptr d(new Test::TransparentOperationDesc);
    Node::Ptr
            p1(new Node(d)),
            p2(new Node(d)),
            p3(new Node(d)),
            p4(new Node(d));
    p2->setChild (p3);
    p1->setChild (p2);
    p3->setChild (p4);
    p4->setChild (np);

    {
        Node::ReadPtr
                nr1(p1),
                nr3(p3);
        Node::WritePtr
                nr4(p4);

        const Node::NodeData *r1 = nr1->data();
        const Node::NodeData *r3 = nr3->data();
        Node::NodeData *r4 = nr4->data();

        EXCEPTION_ASSERT_EQUALS( *d, *d );
        EXCEPTION_ASSERT_EQUALS( d.get (), d.get ());
        EXCEPTION_ASSERT_EQUALS( r1->operationDesc (), *d );
        EXCEPTION_ASSERT_NOTEQUALS( &r1->operationDesc (), d.get () );
        EXCEPTION_ASSERT_EQUALS( r1->operationDesc (), r3->operationDesc () );
        EXCEPTION_ASSERT_NOTEQUALS( &r1->operationDesc (), &r3->operationDesc () );

        r4->cache = Cache(); //Signal::SinkSource(b->number_of_channels ());
        r4->cache.put (b);
        EXCEPTION_ASSERT_EQUALS( b->getInterval (), r4->cache.samplesDesc () );
    }

    np->invalidateSamples (Interval(0,6));
    Signal::Intervals I = Node::WritePtr (p4)->data ()->cache.samplesDesc ();
    EXCEPTION_ASSERT_EQUALS( Interval(6,10), I );
}

} // namespace Dag
} // namespace Signal
