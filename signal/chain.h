#ifndef CHAIN_H
#define CHAIN_H

#include "postsink.h"

#include <QObject>
#include <QString>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>

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

    /**
      Name can be used as a title in the GUI.
      */
    QString name;


    bool isInChain(Signal::pOperation) const;
signals:
    void chainChanged();

private:
    /**
      */
    Signal::pOperation root_source_, tip_source_;

//    friend class boost::serialization::access;
//    Chain() { } // used by serialization, root is read from archive instead
//    template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/) const {
//        ar & BOOST_SERIALIZATION_NVP(root_source_);
//        ar & BOOST_SERIALIZATION_NVP(tip_source_);
//    }

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
    ChainHead( pChain chain );


    /**
      appendOperation moves head and discards whatever chain->tip_source()
      previously was pointing to.
      */
    void appendOperation(Signal::pOperation s);


    /**
      Current HEAD, As head_source(s) changes head_source->source this
      pOperation will always be the same.
      */
    Signal::pOperation     head_source_ref() const;
    Signal::pOperation     head_source() const;
    Signal::PostSink&      post_sink();


    /**
      Set HEAD to some arbitrary other operation. It is an error to set HEAD
      to an operation that is not in the Chain.
    */
    void head_source(Signal::pOperation s);

    pChain chain();

signals:
    void headChanged();

private slots:
    void chainChanged();

private:
    pChain chain_;
    PostSink post_sink_;
    Signal::pOperation head_source_;
};
typedef boost::shared_ptr<ChainHead> pChainHead;

} // namespace Signal

#endif // CHAIN_H
