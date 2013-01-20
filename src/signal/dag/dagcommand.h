#ifndef SIGNAL_DAG_COMMAND_H
#define SIGNAL_DAG_COMMAND_H

#include "node.h"

namespace Signal {
namespace Dag {


class ICommand
{
public:
    typedef boost::shared_ptr<ICommand> Ptr;

    virtual ~ICommand() {}

    static void test();

private:
    friend class DagHead;

    virtual Node::Ptr execute(Node::Ptr head)=0;
};


/**
 * @brief The CommandAddUnaryOperation class appends a unary operation to a dag.
 */
class CommandAddUnaryOperation: public ICommand
{
public:
    CommandAddUnaryOperation(Signal::OperationDesc::Ptr operation);

    Node::Ptr node();

    static void test();

private:
    Node::Ptr node_;

    Node::Ptr execute(Node::Ptr head);
};


/**
 * @brief The CommandRemoveNode class removes a node from a dag.
 */
class CommandRemoveNode: public ICommand
{
public:
    CommandRemoveNode(Node::Ptr node);

    static void test();

private:
    Node::Ptr node_;

    Node::Ptr execute(Node::Ptr oldhead);
};


/**
 * @brief The CommandReplaceOperation class is used to replace a node.
 * This command is usually used to just change a parameter in an operation description.
 */
class CommandReplaceOperation: public ICommand
{
public:
    CommandReplaceOperation (Node::Ptr node, Signal::OperationDesc::Ptr operationdesc);

    static void test();

private:
    Node::Ptr node_;
    Signal::OperationDesc::Ptr operationdesc_;

    Node::Ptr execute(Node::Ptr head);
};


/**
 * @brief The CommandUpdateNode class is used when some asynchronously shared
 * resource has changed. This change might make calling read on the node again
 * result in different data than was previously returned. The Operation is
 * responsible for keeping the shared resource safe in multithreading.
 *
 * Examples include newly recorded data and results from scripts.
 */
class CommandUpdateNode: public ICommand
{
public:
    CommandUpdateNode (Node::Ptr node, Signal::Intervals I);

    static void test();

private:
    Node::Ptr node_;
    Signal::Intervals I_;

    Node::Ptr execute(Node::Ptr head);
};

} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_COMMAND_H
