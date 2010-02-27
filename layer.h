#ifndef LAYER_H
#define LAYER_H

#include <list>
#include <boost/shared_ptr.hpp>

class Transform_chunk;
typedef boost::shared_ptr<class Layer> pLayer;

class Layer
{
public:
    virtual bool operator()( Waveform_chunk& ) = 0;
};

class LayerWaveform: public Layer
{
    LayerWaveform( pWaveform src );
    virtual bool operator()( Waveform_chunk& );
private:
    pWaveform _src;
};

class LayerFilter: public Layer
{
    LayerFilter( pFilter filter );
    bool virtual operator()( Waveform_chunk& chunk );
private:
    pFilter _filter;
};

class LayerSequence: public Layer, public std::list<pFilter>
{
    virtual bool operator()( Waveform_chunk& ) = 0;
};

class LayerMerge: public Layer, public std::list<pFilter>
{
    virtual bool operator()( Waveform_chunk& ) = 0;
};



#endif // LAYER_H
