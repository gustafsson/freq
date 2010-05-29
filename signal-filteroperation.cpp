#include "signal-filteroperation.h"
#include <CudaException.h>
#include <memory.h>

namespace Signal {

FilterOperation::
        FilterOperation(pSource source, Tfr::pFilter filter)
:   OperationCache(source),
    _filter( filter ),
    _save_previous_chunk( false )
{
}

pBuffer FilterOperation::
        readRaw( unsigned firstSample, unsigned numberOfSamples )
{
    unsigned wavelet_std_samples = cwt.wavelet_std_samples( _source->sample_rate());

    // wavelet_std_samples gets stored in cwt so that inverse_cwt can take it
    // into account and create an inverse that is of the desired size.
    unsigned first_valid_sample = 0;
    if (firstSample < wavelet_std_samples) first_valid_sample = firstSample;
    else first_valid_sample = wavelet_std_samples;
    firstSample -= first_valid_sample;

    if (numberOfSamples<.5f*wavelet_std_samples)
        numberOfSamples=.5f*wavelet_std_samples;

    _previous_chunk.reset();

    // Decrease the amount of memory required
    pBuffer r;
    while(true) {
        try {
            unsigned L = numberOfSamples + 2*wavelet_std_samples;

            // If filter would make all these samples zero, return immediately
            if (!_save_previous_chunk)
            {
                if (_filter) {
                    SamplesIntervalDescriptor work(firstSample, firstSample + L );
                    work -= _filter->getZeroSamples( _source->sample_rate() );
                    if (work.isEmpty()) {
                        pBuffer b( new Buffer( firstSample, L, _source->sample_rate() ));
                        ::memset( b->waveform_data->getCpuMemory(), 0, L);
                        return b;
                    }
                }
            }

            pBuffer b = _source->readFixedLength( firstSample, L );

            // If filter would leave these samples untouched, there is nothing to do; return
            if (!_save_previous_chunk)
            {
                if (_filter) {
                    SamplesIntervalDescriptor work(firstSample, firstSample + L );
                    work -= _filter->getUntouchedSamples( _source->sample_rate() );
                    if (work.isEmpty())
                        return b;
                } else {
                    // If there is no filter to apply, there is nothing to do; return
                    return b;
                }
            }

            // Compute the continous wavelet transform
            Tfr::pChunk c = cwt( b );

            c->n_valid_samples += c->first_valid_sample - first_valid_sample;
            c->first_valid_sample = first_valid_sample;

            // Apply filter
            if (_filter)
            {
                (*_filter)( *c );
            }

            // Compute the inverse, unless it is the chunk that is the interesting part
            if (!_save_previous_chunk)
                r = inverse_cwt( *c );
            else
                r = b;

            if (_save_previous_chunk)
                _previous_chunk = c;

            break;
        } catch (const CufftException &) {
            if (numberOfSamples>1) {
                numberOfSamples/=2;
                continue;
            }
            throw;
        } catch (const CudaException &x) {
            if (x.getCudaError() == cudaErrorMemoryAllocation && numberOfSamples>1) {
                numberOfSamples/=2;
                continue;
            }
            throw;
        }
    }

    _save_previous_chunk = false;

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
            _filter = Tfr::pFilter( c );
        } else {
            _filter = f->filter();
        }
    }

    if (0!=c) {
        Tfr::FilterChain* c2 = dynamic_cast<Tfr::FilterChain*>(f->filter().get());
        if (0==c2) {
            if(f->filter())
                c->push_back( f->filter() );
        } else {
            c->insert(c->end(), c2->begin(), c2->end());
        }
    }

    // Remove _source (this effectively prevents two subsequent FilterOperation to
    // have different parameters for Cwt and InverseCwt)
    _source = f->source();
}

Tfr::pChunk FilterOperation::
        previous_chunk()
{
    _save_previous_chunk = true;
    return _previous_chunk;
}

void FilterOperation::
        release_previous_chunk()
{
    _previous_chunk.reset();
}

void FilterOperation::
        filter( Tfr::pFilter f )
{
    // Start with the assumtion that everything will have to be recomputed
    SamplesIntervalDescriptor a = SamplesIntervalDescriptor::SamplesIntervalDescriptor_ALL;

    if (_filter)
    {
        // Don't recompute what would still be zero
        a -= (_filter->getZeroSamples( sample_rate() ) &= f->getZeroSamples( sample_rate() ));

        // Don't recompute what would still be unaffected
        a -= (_filter->getUntouchedSamples( sample_rate() ) &= f->getUntouchedSamples( sample_rate() ));
    }

    // The samples that were previously invalid are still invalid, merge
    _invalid_samples |= a;

    // Change filter
    _filter = f;
}

} // namespace Signal
