#include "layer.h"

#if 0 // TODO Convert layer to Signal::Operation

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

class layer_sequence
{
    Waveform_chunk& w;
    bool r;
public:
    layer_sequence( Waveform_chunk& w):w(w),r(true) {}

    void operator()( pLayer p) {
        r |= (*p)( w );
    }

    operator bool() { return r; }
};

bool LayerSequence::operator()( Waveform_chunk& w) {
    return std::for_each(begin(), end(), layer_sequence( w ));
}

LayerMerge::LayerMerge( pLayer layer ): _layer(layer) {}

bool LayerMerge::operator()( Waveform_chunk& w) {
    Waveform_chunk t(sizeof w));
    bool r = (*layer)( t );

    Waveform wf;
    wf.setBehind( t );
    LayerWaveform l( wf )
    r |= l(w);
    return r;
}
#endif
