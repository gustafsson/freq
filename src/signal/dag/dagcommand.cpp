#include "dagcommand.h"

namespace Signal {
namespace Dag {


CommandAddUnaryOperation::
        CommandAddUnaryOperation(Signal::OperationDesc::Ptr operation)
    : node_(new Node(operation))
{
    EXCEPTION_ASSERT(1 == operation->getNumberOfSources());
}


Node::Ptr CommandAddUnaryOperation::
        node()
{
    return node_;
}


Node::Ptr CommandAddUnaryOperation::
        execute(Node::Ptr head)
{
    node_->setChild(head);
    return node_;
}


CommandRemoveNode::
        CommandRemoveNode(Node::Ptr node)
    : node_(node)
{
    // Can not remove source nodes.
    EXCEPTION_ASSERT_LESS ( 0, Node::ReadPtr(node)->numChildren() );
}


Node::Ptr CommandRemoveNode::
        execute(Node::Ptr oldheadp)
{
    Node::Ptr newhead = oldheadp->getChild();
    Node::WritePtr oldhead(oldheadp);
    // ReadPtr will make sure that parents isn't changed during the scope of this method.

    // Move parents around head, new head is the first child
    std::set<Node::Ptr> P = oldhead->parents ();
    for (std::set<Node::Ptr>::iterator itr = P.begin();
         itr != P.end();
         ++itr)
    {
        Node::WritePtr p(*itr);
        for (int j=0; j < p->numChildren (); ++j)
        {
            if (p->getChild (j) == oldheadp)
                p->setChild (newhead, j);
        }
    }

    // Parents should be detached here.
    EXCEPTION_ASSERT(oldhead->parents().empty ());

    return newhead;
}


CommandReplaceOperation::
        CommandReplaceOperation (Node::Ptr node, Signal::OperationDesc::Ptr operationdesc)
    :
        node_(node),
        operationdesc_(operationdesc)
{}


Node::Ptr CommandReplaceOperation::
        execute(Node::Ptr head)
{
    Node::WritePtr(node_)->data()->operationDesc ( operationdesc_ );
    return head;
}


Node::Ptr CommandUpdateNode::
        execute (Node::Ptr head)
{
    Node::WritePtr(node_)->invalidateSamples (I_);
    return head;
}


void ICommand::
        test()
{
    CommandAddUnaryOperation::test ();
    CommandRemoveNode::test ();
    CommandReplaceOperation::test ();
    CommandUpdateNode::test ();
}


void CommandAddUnaryOperation::
        test()
{
}


void CommandRemoveNode::
        test()
{
}


void CommandReplaceOperation::
        test()
{
}


void CommandUpdateNode::
        test()
{
}


} // namespace Dag
} // namespace Signal
