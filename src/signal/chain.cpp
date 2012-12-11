#include "chain.h"

#include "tfr/filter.h"

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
                 name.c_str(),
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


class ChainHeadReference: public Signal::DeprecatedOperation
{
public:
    ChainHeadReference( Signal::pOperation o ): DeprecatedOperation(o) {}
};

ChainHead::
        ChainHead( pChain chain )
            :
            chain_(chain)
{
    connectChain();
}


void ChainHead::
        appendOperation(Signal::pOperation s)
{
    EXCEPTION_ASSERT( s );

    // Check that this operation is not already in the list. Can't move into
    // composite operations yet as there is no operation iterator implemented.
    unsigned i = 0;
    for(Signal::pOperation itr = head_source_ref(); itr; itr = itr->source() )
    {
        if ( itr == s )
        {
            std::stringstream ss;
            ss << "ChainHead::appendOperation( " << vartype(*s) << ", " << std::hex << s.get() << ") "
                    << "is already in operation chain at postion " << i;
            throw std::invalid_argument( ss.str() );
        }

        ++i;
    }

    s->source( head_source() );
    TaskInfo tt("ChainHead::appendOperation '%s' on\n%s",
                s->name().c_str(), head_source()->toString().c_str());

    pOperation new_head = s;

    // Cache all calculations
    if (0 == dynamic_cast<Signal::OperationCache*>( s.get() ))
        new_head.reset( new OperationCacheLayer(s) );

    // Inject this operation in the middle
    Signal::pOperation o = Signal::DeprecatedOperation::findParentOfSource( chain_->tip_source(), head_source() );
    if (o)
        o->source( new_head );

    head_source( new_head );

    s->invalidate_samples( s->affected_samples() );
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
        Signal::Intervals diff = Signal::DeprecatedOperation::affectedDiff( head_source(), s );
        head_source_->source( s );
        head_source_->invalidate_samples( diff );

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


void ChainHead::
        connectChain()
{
    head_source_.reset( new ChainHeadReference(chain_->tip_source()) );
    connect( chain_.get(), SIGNAL(chainChanged()), SLOT(chainChanged()), Qt::QueuedConnection);
}


} // namespace Signal
