#ifndef SIGNAL_DAG_H
#define SIGNAL_DAG_H

// DAG, Directed_acyclic_graph
// http://en.wikipedia.org/wiki/Directed_acyclic_graph
//
// Describes a graph of signal processing nodes and operations on it.

#include "dagcommand.h"
#include "processor.h"

#include <QReadWriteLock>

namespace Signal {
namespace Dag {

class Dag
{
public:
    typedef boost::shared_ptr<Dag> Ptr;

    Dag (Node::Ptr head) : head_(head), tip_(head) {}

    Processor getProcessor() {
        return Processor(&dag_lock_, &head_);
    }

    void queueCommand(ICommand::Ptr cmd) {
        QWriteLocker l (&cmdqueue_lock_);
        cmdqueue_.push_back (cmd);
    }

private:
    void executeQueue() {
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

    QReadWriteLock dag_lock_;
    Node::Ptr head_; // There's only one 'head'.
    Node::Ptr tip_; // There's only one 'tip'. 'head' and 'tip' might be the same.
    // Multiple Dags are out-of-scope.
    // Multiple tips are out-of-scope. Or?

    QReadWriteLock cmdqueue_lock_;
    std::vector<ICommand::Ptr> cmdqueue_;
};


} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_H
