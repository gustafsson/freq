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
        TaskInfo("Chain '%s' is now\n%s",
                 name.toStdString().c_str(),
                 tip_source_->toString().c_str());
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

    TaskInfo tt("ChainHead::appendOperation '%s' on\n%s",
                s->name().c_str(), head_source_ref()->toString().c_str());

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

    head_source( pOperation( new OperationCacheLayer(s) ));

    TaskInfo("Worker::appendOperation, worker tree:\n%s", head_source_ref()->toString().c_str());
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
        Signal::Intervals new_data( 0, head_source()->number_of_samples() );
        Signal::Intervals old_data( 0, s->number_of_samples() );
        Signal::Intervals invalid = new_data | old_data;

        Signal::Intervals was_zeros = head_source()->zeroed_samples_recursive();
        Signal::Intervals new_zeros = s->zeroed_samples_recursive();
        Signal::Intervals still_zeros = was_zeros & new_zeros;
        invalid -= still_zeros;

        invalid &= s->affected_samples_until( head_source() );
        invalid &= head_source()->affected_samples_until( s );

        head_source_->source( s );
        head_source_->invalidate_samples( invalid );

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
    Signal::pOperation test = chain_->tip_source();
    if (dynamic_cast<Signal::OperationCacheLayer*>(test.get()) && test->source() != head_source())
        test = test->source();

    if (head_source() == test->source())
        head_source( chain_->tip_source() );
    else
    {
        if( !chain_->isInChain(head_source()) )
            head_source( chain_->tip_source() );
    }
}


} // namespace Signal
