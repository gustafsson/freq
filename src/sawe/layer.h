#ifndef LAYER_H
#define LAYER_H

#include <list>
#include <boost/shared_ptr.hpp>

#if 0 // TODO Convert layer to Signal::Operation

class Transform_chunk;
typedef boost::shared_ptr<class Layer> pLayer;

class Layer
{
public:
    virtual bool operator()( Waveform_chunk& ) = 0;
};

class LayerWaveform: public Layer
{
public:
    LayerWaveform( pWaveform src );
    virtual bool operator()( Waveform_chunk& );
private:
    pWaveform _src;
};

class LayerFilter: public Layer
{
public:
    LayerFilter( pFilter filter );
    bool virtual operator()( Waveform_chunk& chunk );
private:
    pFilter _filter;
};

class LayerSequence: public Layer, public std::list<pFilter>
{
public:
    virtual bool operator()( Waveform_chunk& ) = 0;
};

class LayerMerge: public Layer
{
public:
    LayerMerge( pLayer layer );
    virtual bool operator()( Waveform_chunk& ) = 0;
private:
    pLayer _layer;
};

#endif

#endif // LAYER_H
