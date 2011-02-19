#include "target.h"

#include "operation-basic.h"

#include "sawe/project.h"
#include "tools/renderview.h"

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
        if (current_channel_< source()->num_channels())
            return source()->read( I );
        else
            return source2_->read( I );
    }

    virtual pOperation source2() const { return source2_; }

    virtual unsigned num_channels() { return source()->num_channels() + source2_->num_channels(); }
    virtual void set_channel(unsigned c) {
        BOOST_ASSERT( c < num_channels() );
        if (c < source()->num_channels())
            source()->set_channel(c);
        else
            source2_->set_channel(c - source()->num_channels());
        current_channel_ = c;
    }
    virtual unsigned get_channel() { return current_channel_; }

private:
    pOperation source2_;
    unsigned current_channel_;
};


class ForAllChannelsOperation: public Operation
{
public:
    ForAllChannelsOperation
        (
            Signal::pOperation o
        )
            :
        Operation(o)
    {
    }


    virtual pBuffer read( const Interval& I )
    {
        unsigned N = num_channels();
        Signal::pBuffer r;
        for (unsigned i=0; i<N; ++i)
        {
            set_channel( i );
            r = Signal::Operation::read( I );
        }

        return r;
    }
};


class UpdateView: public Operation
{
public:
    UpdateView(Sawe::Project* project)
        :
        Operation(pOperation()),
        render_view_(project->tools().render_view())
    {

    }


    virtual void invalidate_samples(const Intervals& I)
    {
        render_view_->userinput_update( false );

        Operation::invalidate_samples(I);
    }


private:
    Tools::RenderView* render_view_;
};


Layers::
        Layers(Sawe::Project* project)
            :
            project_(project)
{
}


Layers::
        ~Layers()
{
    TaskInfo ti("~Layers");
    layers_.clear();
}


Sawe::Project* Layers::
        project()
{
    return project_;
}


std::set<pChain> Layers::
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
        isInSet(pChain p) const
{
    return layers_.find( p ) != layers_.end();
}


std::string Layers::
        toString() const
{
    std::string s;
    for (std::set<pChain>::iterator itr = layers_.begin(); itr != layers_.end(); ++itr)
    {
        s += (*itr)->tip_source()->toString();
        s += "\n\n";
    }
    return s;
}


Target::
        Target(Layers* all_layers)
            :
            post_sink_( new PostSink ),
            rewire_channels_( new RewireChannels(pOperation()) ),
            forall_channels_( new ForAllChannelsOperation(pOperation()) ),
            update_view_( new UpdateView( all_layers->project() )),
            add_as_channels_(false),
            all_layers_(all_layers)
{
    BOOST_FOREACH( pChain c, all_layers_->layers() )
    {
        addLayerHead( pChainHead(new ChainHead(c)));
    }

    rewire_channels_->source( post_sink_ );
    forall_channels_->source( rewire_channels_ );
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


pChainHead Target::
        findHead( pChain c )
{
    BOOST_FOREACH( pChainHead p, layerHeads )
    {
        if (c == p->chain())
            return p;
    }

    return pChainHead();
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


RewireChannels* Target::
        channels() const
{
    return dynamic_cast<RewireChannels*>(rewire_channels_.get());
}


pBuffer Target::
        readChannels( const Interval& I )
{
    return forall_channels_->read( I );
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
