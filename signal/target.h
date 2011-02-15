#ifndef LAYERS_H
#define LAYERS_H

#include "chain.h"

#include <list>

namespace Signal {

class Layers
{
public:
    std::vector<pChain> layers();

    virtual void addLayer(pChain);
    virtual void removeLayer(pChain);

private:
    std::vector<pChain> layers_;
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
class Target : public Layers {
public:
    Target(Layers* all_layers_);

    /**
      It is an error to and a layer that is not in 'all_layers_'
      */
    virtual void addLayer(pChain);

    /**
      It is an error to and a layer that is not in 'layers()'
      */
    virtual void removeLayer(pChain);

    /**
      layer merging doesn't have to be done by superpositioning, it could also
      make a setup so that it puts the results from each layer in its
      own channel
      */
    void addAsChannels(bool add_as_channels);

    /**
      Add sinks to this target group through this PostSink
      */
    Signal::PostSink*   post_sink() const;

private:
    void rebuildSource();

    Signal::pOperation post_sink_;
    bool add_as_channels_;
    Layers* all_layers_;
};


} // namespace Signal

#endif // LAYERS_H
