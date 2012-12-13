#include "dag.h"

namespace Signal {
namespace Dag {

Dag::
        Dag (Node::Ptr head)
    :
      head_(head),
      tip_(head)
{}


Processor Dag::
        getProcessor()
{
    return Processor(&dag_lock_, &head_);
}


void Dag::
        queueCommand(ICommand::Ptr cmd)
{
    QWriteLocker l (&cmdqueue_lock_);
    cmdqueue_.push_back (cmd);
}


void Dag::
        executeQueue()
{
    QWriteLocker l1 (&dag_lock_);
    QWriteLocker l2 (&cmdqueue_lock_);
    for (int i=0; i<(int)cmdqueue_.size (); ++i)
    {
        bool movetip = (head_ == tip_);
        Node::Ptr oldhead = head_;
        head_ = cmdqueue_[i]->execute (head_);

        // Move parents if head has changed.
        if (oldhead != head_)
        {
            // if oldhead is unique P should be empty
            std::set<Node*> P = oldhead->parents();
            for (std::set<Node*>::iterator itr = P.begin();
                 itr != P.end();
                 ++itr)
            {
                Node* p = *itr;
                for (int j=0; j<p->numChildren(); ++j)
                {
                    if (&p->getChild(j) == oldhead.get ())
                        p->setChild(head_, j);
                }
            }
        }

        if (movetip)
            tip_ = head_;
    }

    cmdqueue_.clear ();
}

} // namespace Dag
} // namespace Signal
