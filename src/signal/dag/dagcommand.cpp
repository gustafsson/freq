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
}


Node::Ptr CommandRemoveNode::
        execute(Node::Ptr oldhead)
{
    Node::Ptr newhead = oldhead->getChildPtr();

    // Move parents around head, new head is the first child
    std::set<Node*> P = oldhead->parents();
    for (std::set<Node*>::iterator itr = P.begin();
         itr != P.end();
         ++itr)
    {
        Node* p = *itr;
        for (int j=0; j<p->numChildren(); ++j)
        {
            if (p->getChildPtr(j) == oldhead)
                p->setChild(newhead, j);
        }
    }

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
    node_->operationDesc ( operationdesc_ );
    return head;
}


Node::Ptr CommandUpdateNode::
        execute (Node::Ptr head)
{
    node_->invalidateSamples (I_);
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
