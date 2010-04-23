#include "signal-filteroperation.h"

namespace Signal {

FilterOperation::
FilterOperation(pSource source, Tfr::pFilter filter)
:   Operation(source),
    _filter( filter )
{
}

pBuffer FilterOperation::
read( unsigned firstSample, unsigned numberOfSamples )
{
    meldFilters();

    unsigned wavelet_std_samples = cwt.wavelet_std_samples( _source->sample_rate());

    // wavelet_std_samples gets stored in c so that inverse_cwt can take it
    // into account and create an inverse that is of the desired size.
    if (firstSample < wavelet_std_samples) firstSample = 0;
    else firstSample -= wavelet_std_samples;

    if (numberOfSamples<.5f*wavelet_std_samples)
        numberOfSamples=.5f*wavelet_std_samples;

    pBuffer b = _source->read( firstSample, numberOfSamples + 2*wavelet_std_samples );

    Tfr::pChunk c = cwt( b );
    if (_filter) (*_filter)( *c );
    pBuffer r = inverse_cwt( *c );

    _previous_chunk = c;

    return r;
}

void FilterOperation::
meldFilters()
{
    FilterOperation* f = dynamic_cast<FilterOperation*>( _source.get());
    if (0==f) return;

    f->meldFilters();

    Tfr::FilterChain* c = dynamic_cast<Tfr::FilterChain*>(_filter.get());
    if (0==c) {        
        if (_filter) {
            c = new Tfr::FilterChain;
            c->push_back( _filter );
        }
        _filter = Tfr::pFilter( c );
    }

    Tfr::FilterChain* c2 = dynamic_cast<Tfr::FilterChain*>(f->filter().get());
    if (0==c2) {
        if(f->filter()) c->push_back( f->filter() );
    } else {
        c->insert(c->end(), c2->begin(), c2->end());
    }

    // Remove _source (this effectively prevents two subsequent FilterOperation to
    // have different parameters for Cwt and InverseCwt
    _source = f->source();
}


} // namespace Signal
