void Filter::invalidateWaveform( const Transform& t, Signal::Buffer& w)
{
    float start,end;
    range(start,end);
    for (unsigned n = t.getChunkIndex( std::max(0.f,start)*t.original_waveform()->sample_rate());
         n <= t.getChunkIndex( std::max(0.f,end)*t.original_waveform()->sample_rate());
         n++)
    {
        w.valid_transform_chunks.erase(n);
    }
}

class invalidate_waveform
{
    const Transform& t;
    Signal::Buffer& w;
public:
    invalidate_waveform( const Transform& t, Signal::Buffer& w):t(t),w(w) {}

    void operator()( pFilter p) {
        p->invalidateWaveform(t,w);
    }
};

void FilterChain::invalidateWaveform( const Transform& t, Signal::Buffer& w) {
    std::for_each(begin(), end(), invalidate_waveform( t,w ));
}
