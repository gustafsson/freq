#include "target.h"

#include "operation-basic.h"

#include "sawe/project.h"
#include "tools/renderview.h"
#include "tfr/filter.h"

#include "neat_math.h"

#include <boost/foreach.hpp>
#include <boost/noncopyable.hpp>

#define DEBUG_Target if(0)
//#define DEBUG_Target

namespace Signal {


class ForAllChannelsOperation: public Operation, public boost::noncopyable
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


    virtual void invalidate_samples(const Intervals& I)
    {
        unsigned N = num_channels();
        if (0 < N)
        {
            if (get_channel() >= N)
                set_channel(N - 1);
        }

        Operation::invalidate_samples( I );
    }
};


class UpdateView: public Operation, public boost::noncopyable
{
public:
    UpdateView(Sawe::Project* project, std::string targetname)
        :
        Operation(pOperation()),
        project_(project),
        targetname_(targetname)
    {
    }


    virtual std::string name()
    {
        return Operation::name() + " for '" + targetname_ + "' in '" + project_->project_title() + "'";
    }


    virtual void invalidate_samples(const Intervals& I)
    {
        if (project_->areToolsInitialized())
            project_->tools().render_view()->userinput_update( false );

        Operation::invalidate_samples(I);

        project_->setModified();
    }


private:
    Sawe::Project* project_;
    std::string targetname_;
};


/**
  This is a premature optimization that saves quite a few function calls.
  */
class CacheVars: public Operation, public boost::noncopyable
{
public:
    CacheVars() : Operation(pOperation()), FS(-1) {}

    virtual Signal::pBuffer read(const Interval& I)
    {
        update();
        return Operation::read( I );
    }


    virtual IntervalType number_of_samples()
    {
        if (0>FS)
            return Operation::number_of_samples();
        return number_of_samples_;
    }

    virtual float sample_rate()
    {
        if (0>FS)
            update();
        return FS;
    }


    virtual void invalidate_samples(const Intervals& I)
    {
        update();

        Operation::invalidate_samples(I);
    }


private:
    void update()
    {
        number_of_samples_ = Operation::number_of_samples();
        FS = Operation::sample_rate();
    }

    Signal::IntervalType number_of_samples_;
    float FS;
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
    for (std::set<pChain>::const_iterator itr = layers_.begin(); itr != layers_.end(); ++itr)
    {
        s += (*itr)->tip_source()->toString();
        s += "\n\n";
    }
    return s;
}


Target::
        Target(Layers* all_layers, std::string name, bool allow_cheat_resolution, bool autocreate_chainheads)
            :
            name_( name ),
            post_sink_( new PostSink ),
            reroute_channels_( new RerouteChannels(pOperation()) ),
            forall_channels_( new ForAllChannelsOperation(pOperation()) ),
            update_view_( new UpdateView( all_layers->project(), name )),
            cache_vars_( new CacheVars ),
            add_as_channels_(false),
            allow_cheat_resolution_(allow_cheat_resolution),
            all_layers_(all_layers)
{
    TaskInfo("Target name %s", name_.c_str());
    // all_layers_ might not actually be needed, but, project() is for update_view
    BOOST_ASSERT( all_layers_ );

    post_sink_->source( reroute_channels_ );
    forall_channels_->source( post_sink_ );
    update_view_->source( forall_channels_ );
    cache_vars_->source(update_view_);
    read_ = cache_vars_;

    if (autocreate_chainheads && all_layers_)
    {
        BOOST_FOREACH( pChain c, all_layers_->layers() )
        {
            /*if (all_layers->project()->head->chain() == c)
                addLayerHead(all_layers->project()->head);
            else*/
                addLayerHead( pChainHead(new ChainHead(c)));
        }
    }
}


Target::
        ~Target()
{
    TaskInfo("~Target name %s", name_.c_str());
}


void Target::
        addLayerHead(pChainHead p)
{
    BOOST_ASSERT( p );
    BOOST_ASSERT( !isInSet(p->chain()) );
    //BOOST_ASSERT( all_layers_->isInSet(p->chain()) );

    Signal::Intervals was_zero = read_->zeroed_samples_recursive();

    layerHeads.insert( p );
    rebuildSource();

    Signal::Intervals is_zero = read_->zeroed_samples_recursive();
    Signal::Intervals need_update = (was_zero&is_zero).inverse();

    post_sink()->invalidate_samples( need_update );
}


void Target::
        removeLayerHead(pChainHead p)
{
    BOOST_ASSERT( isInSet(p->chain()) );
    //BOOST_ASSERT( all_layers_->isInSet(p->chain()) );

    Signal::Intervals was_zero = read_->zeroed_samples_recursive();

    layerHeads.erase( p );
    rebuildSource();

    Signal::Intervals is_zero = read_->zeroed_samples_recursive();
    Signal::Intervals need_update = (was_zero&is_zero).inverse();

    post_sink()->invalidate_samples( need_update );
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


RerouteChannels* Target::
        channels() const
{
    return dynamic_cast<RerouteChannels*>(reroute_channels_.get());
}


pBuffer Target::
        read( const Interval& I )
{
    DEBUG_Target TaskInfo("Target \"%s\", reading\n%s", name_.c_str(), read_->toString().c_str());
    return read_->read( I );
}


std::string Target::
        name()
{
    return name_;
}


pOperation Target::
        source() const
{
    return read_;
}


bool Target::
        allow_cheat_resolution() const
{
    return allow_cheat_resolution_;
}


void Target::
        allow_cheat_resolution(bool value)
{
    allow_cheat_resolution_ = value;
}


template<typename T>
T* findType(Signal::pOperation p)
{
    for(Signal::pOperation r = p; r!=0; r = r->source())
    {
        if (T* t = dynamic_cast<T*>(r.get()))
            return t;

        std::string n = r->name();
        PostSink* ps = dynamic_cast<PostSink*>(r.get());
        if (ps) foreach(Signal::pOperation o, ps->sinks())
        {
            std::string m = o->name();
            if (T* t = findType<T>(o))
                return t;
        }
    }

    return 0;
}

unsigned Target::
        next_good_size( unsigned current_valid_samples_per_chunk )
{
    if (Tfr::Filter* f = findType<Tfr::Filter>(source()))
        return f->next_good_size(current_valid_samples_per_chunk);

    unsigned minsize = 64;
    if (current_valid_samples_per_chunk<minsize)
        return minsize;

    return spo2g(align_up(current_valid_samples_per_chunk, minsize)/minsize)*minsize;
}


unsigned Target::
        prev_good_size( unsigned current_valid_samples_per_chunk )
{
    if (Tfr::Filter* f = findType<Tfr::Filter>(source()))
        return f->prev_good_size(current_valid_samples_per_chunk);

    unsigned minsize = 64;
    if (current_valid_samples_per_chunk<2*minsize)
        return minsize;

    return lpo2s(align_up(current_valid_samples_per_chunk, minsize)/minsize)*minsize;
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
        else
        {
            if (add_as_channels_)
                s.reset(new OperationAddChannels(s, p->head_source_ref()));
            else
                s.reset(new OperationSuperposition(s, p->head_source_ref()));
        }
    }

    reroute_channels_->source( s );

    DEBUG_Target TaskInfo("Target::rebuildSource created\n%s", read_->toString().c_str());
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


OperationTarget::
        OperationTarget(Layers* all_layers, pOperation operation, std::string name, bool allow_cheat_resolution)
            :
            Target(all_layers, name, allow_cheat_resolution, false)
{
    addOperation(operation);
}


void OperationTarget::
        addOperation(pOperation operation)
{
    Signal::pChain chain( new Signal::Chain(operation) );
    Signal::pChainHead ch( new Signal::ChainHead(chain));

    // Add cache layer
    ch->appendOperation(Signal::pOperation(new Signal::Operation(Signal::pOperation())));

    // Invalidate cache
    ch->head_source()->invalidate_samples(ch->head_source()->getInterval() );

    addLayerHead( ch );
}

} // namespace Signal
