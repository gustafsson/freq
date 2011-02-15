#include "target.h"

#include "operation-basic.h"

#include <boost/foreach.hpp>

namespace Signal {

class OperationAddChannels: public Operation
{
public:
    OperationAddChannels( pOperation source, pOperation source2 )
        :
        Operation(source),
        source2_(source2),
        current_channel_(0)
    {
    }

    virtual pBuffer read( const Interval& I )
    {
        if (current_channel_< _source->num_channels())
            return _source->read( I );
        else
            return source2_->read( I );
    }

    virtual pOperation source2() const { return _source2; }

    virtual unsigned num_channels() { return _source->num_channels() + source2_->num_channels(); }
    virtual void set_channel(unsigned c) {
        BOOST_ASSERT( c < num_channels() );
        if (c < _source->num_channels())
            _source->set_channel(c);
        else
            source2_->set_channel(c - _source->num_channels());
        current_channel_ = c;
    }
    virtual unsigned get_channel() { return current_channel_; }

private:
    pOperation source2_;
    unsigned current_channel_;
};



std::vector<pChain> Layers::
        layers()
{
    return layers_;
}


void Layers::
        addLayer(pChain p )
{
    BOOST_ASSERT( !isInSet(p) );
    layers_.insert( p );
}


void Layers::
        removeLayer(pChain p)
{
    BOOST_ASSERT( isInSet(p) );
    layers_.erase( p );
}


bool Layers::
        isInSet(pChain) const
{
    return layers_.find( p ) != layers_.end();
}


Target::
        Target(Layers* all_layers)
            :
            post_sink_(new PostSink ),
            add_as_channels_(false),
            all_layers_(all_layers)
{
}


void Target::
        addLayerHead(pChainHead p)
{
    BOOST_ASSERT( all_layers_ );
    BOOST_ASSERT( !isInSet(p->chain()) );
    BOOST_ASSERT( all_layers_->isInSet(p->chain()) );

    layerHeads.insert( p );

    // Add this target (by its post_sink_) to the ChainHead
    Signal::PostSink& ps = p->post_sink();
    std::vector<pOperation> sinks = ps.sinks();
    sinks.push_back( post_sink_ );
    ps.sinks( sinks );

    rebuildSource();

    post_sink()->invalidate_samples( ~p->head_source()->zeroed_samples_recursive() );
}


void Target::
        removeLayerHead(pChainHead p)
{
    BOOST_ASSERT( all_layers_ );
    BOOST_ASSERT( isInSet(p->chain()) );
    BOOST_ASSERT( all_layers_->isInSet(p->chain()) );

    layerHeads.erase( p );

    // Remove this target (by its post_sink_) from the ChainHead
    Signal::PostSink& ps = p->post_sink();
    std::vector<pOperation> sinks = ps.sinks();
    sinks.resize( std::remove( sinks.begin(), sinks.end(), post_sink_ )-sinks.begin() );
    ps.sinks( sinks );

    rebuildSource();

    post_sink()->invalidate_samples( ~p->head_source()->zeroed_samples_recursive() );
}


void Target::
        addAsChannels(bool add_as_channels)
{
    if (add_as_channels_ == add_as_channels)
        return;

    add_as_channels_ = add_as_channels;

    rebuildSource();

    post_sink()->invalidate_samples( ~post_sink()->zeroed_samples_recursive() );
}


PostSink* Target::
        post_sink() const
{
    return dynamic_cast<PostSink*>(post_sink_.get());
}


void Target::
        rebuildSource()
{
    pOperation s;

    BOOST_FOREACH( pChainHead p, layerHeads )
    {
        if (!s)
        {
            s = p->head_source_ref();
        }

        if (add_as_channels_)
            s.reset(new OperationAddChannels(s, p->head_source_ref()));
        else
            s.reset(new OperationSuperposition(s, p->head_source_ref()));
    }

    post_sink_->source( s );
}


bool Target::
        isInSet(pChain c) const
{
    BOOST_FOREACH( pChainHead p, layerHeads )
    {
        if (p->chain() == c)
            return true;
    }
    return false;
}

} // namespace Signal
