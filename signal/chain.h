#ifndef CHAIN_H
#define CHAIN_H

#include "postsink.h"

#include <QObject>

#include <boost/shared_ptr.hpp>

namespace Signal {

class Chain : public QObject
{
    Q_OBJECT
public:
    explicit Chain(Signal::pOperation root);

    /**
      Returns the source from which the chain was initiated.
      */
    Signal::pOperation root_source() const;

    /**
      Returns the top of the tree.
      */
    Signal::pOperation tip_source() const;

    /**
      Sets a new tip.
      */
    void tip_source(Signal::pOperation);

    std::string name;
signals:
    void chainChanged();

private:
    /**
      */
    Signal::pOperation root_source_, tip_source_;
};
typedef boost::shared_ptr<Chain> pChain;


/**
Current work point in a chain. Compare HEAD in a git repo which
points to the point in history that current edits are based upon.
*/
class ChainHead : public QObject
{
    Q_OBJECT
public:
    ChainHead( pChain chain, pOperation postSink );


    /**
      appendOperation moves head and discards whatever chain->tip_source()
      previously was pointing to.
      */
    void appendOperation(Signal::pOperation s);


    /**
      Current HEAD.
      */
    Signal::pOperation     head_source() const;
    Signal::PostSink*      post_sink() const;


    /**
      Set HEAD to some arbitrary other operation. It is an error to set HEAD
      to an operation that is not in the Chain.
    */
    void head_source(Signal::pOperation s);

    pChain chain() { return chain_; }

signals:
    void headChanged();

private slots:
    void chainChanged();

private:
    pChain chain_;
    Signal::pOperation head_;
    Signal::pOperation post_sink_;
};
typedef boost::shared_ptr<Chain> pChainHead;

} // namespace Signal

#endif // CHAIN_H
