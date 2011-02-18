#ifndef LAYERS_H
#define LAYERS_H

#include "chain.h"
#include "rewirechannels.h"

#include <boost/serialization/set.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace Sawe { class Project; }

namespace Signal {

class Layers
{
public:
    Layers(Sawe::Project* project);
    ~Layers();

    Sawe::Project* project();

    std::set<pChain> layers();

    virtual void addLayer(pChain);
    virtual void removeLayer(pChain);

    bool isInSet(pChain) const;

private:
    std::set<pChain> layers_;
    Sawe::Project* project_;

//    friend class boost::serialization::access;
//    template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/) const {
//        ar & BOOST_SERIALIZATION_NVP( layers_);
//    }
};


/**
  post_sink() is given to worker.
  post_sink can be asked for invalid samples so that worker knows what to work on.
  worker still needs to be told if worker should start from 0 or start at the middle.

  chain has a post_sink in which targets register themselves as sinks so that
  invalidate samples can propagate through.

  when worker reads from post_sink post_sink has its source setup so that it
  will merge the results from all chains using OperationSuperposition.

  it could also make a setup so that it puts the results from each layer in its
  own channel.

  the OperationSuperposition structure is rebuilt whenever addLayer, removeLayer
  or addAsChannels is called.
  */
class Target {
public:
    Target(Layers* all_layers);

    /**
      It is an error to add a layer that is not in 'all_layers_'
      */
    void addLayerHead(pChainHead);

    /**
      It is an error to remove a layer that is not in 'layerHeads'
      */
    void removeLayerHead(pChainHead);

    /**
      Returns the chain head that correspons to a chain, if there is one.
      */
    pChainHead findHead( pChain );

    /**
      layer merging doesn't have to be done by superpositioning, it could also
      make a setup so that it puts the results from each layer in its
      own channel
      */
    void addAsChannels(bool add_as_channels);

    /**
      Add sinks to this target group through this PostSink
      */
    PostSink*   post_sink() const;

    /**
      Select which channels that gets to propagate through with anything else
      than silence.
      */
    RewireChannels* channels() const;

    /**
      Will process all active channels (as wired by channels()) and return the
      buffer from the last channel.
      */
    pBuffer readChannels( const Interval& I );

private:
    void rebuildSource();

    Signal::pOperation post_sink_;
    Signal::pOperation rewire_channels_;
    Signal::pOperation forall_channels_;
    Signal::pOperation update_view_;
    bool add_as_channels_;
    Layers* all_layers_;
    std::set<pChainHead> layerHeads;

    bool isInSet(pChain) const;
};
typedef boost::shared_ptr<Target> pTarget;

} // namespace Signal

#endif // LAYERS_H
