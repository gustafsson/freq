#include "layer.h"

LayerWaveform::LayerWaveform( pWaveform src ): _src(src) {}
bool LayerWaveform::operator()( Waveform_chunk& chunk )
{
    /* A = src->get_chunk( chunk.start, chunk.length );
       if chunk->data != 0
        chunk += A;
       else
        chunk = A;
       */
    return false;
}

LayerFilter::LayerFilter( pFilter filter ): _filter(filter) {}
bool LayerFilter::operator()( Waveform_chunk& chunk ) {
    /* compute transform of waveform chunk
       apply filter on transform chunk
       store inverse transform in waveform chunk
    */
}

class merge_layer
{
    Waveform_chunk& w;
    bool r;
public:
    merge_layer( Waveform_chunk& w):w(w),r(true) {}

    void operator()( pLayer p) {
        r |= (*p)( w );
    }

    operator bool() { return r; }
};

bool LayerMerge::operator()( Waveform_chunk& w) {
    return std::for_each(begin(), end(), merge_layer( w ));
}
