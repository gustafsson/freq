#include "daghead.h"

#include <QReadWriteLock>

namespace Signal {
namespace Dag {


DagHead::
        DagHead(Dag::Ptr dag, Signal::OperationDesc::Ptr headprocessor)
    :
      dag_(dag),
      head_(new Node(headprocessor, true))
{
    head_->setChild (dag_->root_);
}


DagHead::
        DagHead(DagHead::Ptr daghead, Signal::OperationDesc::Ptr headprocessor)
    :
      dag_(daghead->dag ()),
      head_(new Node(headprocessor, true))
{
    head_->setChild ( daghead->head ()->getChildPtr () );
}


void DagHead::
        queueCommand(ICommand::Ptr cmd)
{
    QWriteLocker l (&cmdqueue_lock_);
    cmdqueue_.push_back (cmd);

}


void DagHead::
        setInvalidSamples(Signal::Intervals invalid, Signal::IntervalType center)
{

}


void DagHead::
        executeQueue()
{
    QWriteLocker l2 (&cmdqueue_lock_);

    for (int i=0; i<(int)cmdqueue_.size (); ++i)
    {
        Node::Ptr oldhead = head_;
        head_ = cmdqueue_[i]->execute (head_);

        // Move parents if head has changed.
        /*if (oldhead != head_)
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
        }*/

       // if oldhead==tip && tip.usecount
        //if (tip_->parents ().count (head_.get ()))
          //  tip_ = head_;
    }

    cmdqueue_.clear ();
}


} // namespace Dag
} // namespace Signal
