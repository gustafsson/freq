#include "chain.h"

#include "operationcache.h"

namespace Signal {

Chain::
        Chain(Signal::pOperation root)
            :
            root_source_(root),
            tip_source_(root)
{
    name = "Chain";
}


Signal::pOperation Chain::
        root_source() const
{
    return root_source_;
}


Signal::pOperation Chain::
        tip_source() const
{
    return tip_source_;
}


void Chain::
        tip_source(Signal::pOperation tip)
{
    if (tip_source_ != tip)
    {
        tip_source_ = tip;
        emit chainChanged();
    }
}


bool Chain::
        isInChain(Signal::pOperation t) const
{
    Signal::pOperation s = tip_source();
    while( s )
    {
        if (s == t )
            return true;
        s = s->source();
    }

    return false;
}


class ChainHeadReference: public Signal::Operation
{
public:
    ChainHeadReference( Signal::pOperation o ): Operation(o) {}
};

ChainHead::
        ChainHead( pChain chain )
            :
            chain_(chain),
            head_source_( new ChainHeadReference(chain_->tip_source()) )
{
    connect( chain.get(), SIGNAL(chainChanged()), SLOT(chainChanged()));
}


void ChainHead::
        appendOperation(Signal::pOperation s)
{
    BOOST_ASSERT( s );

    TaskInfo tt("ChainHead::appendOperation( %s )", vartype(*s).c_str());

    // Check that this operation is not already in the list. Can't move into
    // composite operations yet as there is no operation iterator implemented.
    unsigned i = 0;
    Signal::pOperation itr = head_source_ref();
    while(itr)
    {
        if ( itr == s )
        {
            std::stringstream ss;
            ss << "ChainHead::appendOperation( " << vartype(*s) << ", " << std::hex << s.get() << ") "
                    << "is already in operation chain at postion " << i;
            throw std::invalid_argument( ss.str() );
        }
        itr = itr->source();
        ++i;
    }

    s->source( head_source() );
    Signal::Intervals was_zeros = head_source()->zeroed_samples_recursive();
    Signal::Intervals new_zeros = s->zeroed_samples_recursive();
    Signal::Intervals affected = s->affected_samples();
    Signal::Intervals still_zeros = was_zeros & new_zeros;
    TaskInfo("Was zero samples %s", was_zeros.toString().c_str());
    TaskInfo("New zero samples %s", new_zeros.toString().c_str());
    TaskInfo("Still zero samples %s", still_zeros.toString().c_str());
    TaskInfo("Affected samples %s", affected.toString().c_str());

    Signal::Intervals signal_length( 0, std::max( 1lu, s->number_of_samples() ));
    head_source_->invalidate_samples( ( affected - still_zeros ) & signal_length );

    //_min_samples_per_chunk = Tfr::Cwt::Singleton().next_good_size( 1, _source->sample_rate());
    //_highest_fps = _min_fps;
    //_max_samples_per_chunk = (unsigned)-1;

    head_source( pOperation( new OperationCacheLayer(s) ));

    tt.tt().info("Worker::appendOperation, worker tree:\n%s", head_source_ref()->toString().c_str());
}


Signal::pOperation ChainHead::
        head_source() const
{
    return head_source_->source();
}


Signal::pOperation ChainHead::
        head_source_ref() const
{
    return head_source_;
}


void ChainHead::
        head_source(Signal::pOperation s)
{
    if (!chain_->isInChain(s))
        chain_->tip_source( s );

    if (head_source() != s)
    {
        head_source_->source( s );
        emit headChanged();
    }
}


pChain ChainHead::
        chain()
{
    return chain_;
}


void ChainHead::
        chainChanged()
{
    if (head_source() == chain_->tip_source()->source())
        head_source( chain_->tip_source() );
    else
    {
        if( !chain_->isInChain(head_source()) )
            head_source( chain_->tip_source() );
    }
}


} // namespace Signal
