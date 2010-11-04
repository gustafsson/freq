#include "brushmodel.h"
#include <demangle.h>

#include "tfr/cwt.h"
#include "support/brushpaint.cu.h"

namespace Tools {


BrushModel::
        BrushModel( Sawe::Project* project )
            :
            brush_factor(0)
{
    filter_.reset( new Support::MultiplyBrush );
    filter_->source( project->head_source() );
    project->head_source( filter_ );
}


Support::BrushFilter* BrushModel::
        filter()
{
    return dynamic_cast<Support::BrushFilter*>(filter_.get());
}


Signal::Interval BrushModel::
        paint( Heightmap::Reference ref, Heightmap::Position pos )
{
    Heightmap::Position a, b;
    ref.getArea(a, b);

    Signal::Intervals r;
    r |= addGauss(ref, pos);
    r |= addGauss(ref.sibblingRight(), pos);
    if (a.time>0)
        r |= addGauss(ref.sibblingLeft(), pos);
    if (a.scale>0)
        r |= addGauss(ref.sibblingBottom(), pos);
    if (b.scale<1)
        r |= addGauss(ref.sibblingTop(), pos);
    return r.coveredInterval();
}


Signal::Interval BrushModel::
        addGauss( Heightmap::Reference ref, Heightmap::Position pos )
{
    Support::BrushFilter::BrushImageDataP& img = (*filter()->images)[ ref ];

    if (!img)
    {
        img.reset( new GpuCpuData<float>(
            0,
            make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
            GpuCpuVoidData::CudaGlobal ) );
        cudaMemset( img->getCudaGlobal().ptr(), 0, img->getSizeInBytes1D() );
    }

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    float fs = filter()->sample_rate();
    float hz = cwt.compute_frequency2( fs, pos.scale );
    float deltasample = Tfr::Cwt::Singleton().morlet_sigma_t( fs, hz );
    float deltascale = cwt.sigma() / cwt.nScales(fs);
    float deltat = deltasample/fs;
    deltat*=10;
    deltascale*=0.01;

    Heightmap::Position a, b;
    ref.getArea( a, b );

    ::addGauss( make_float4(a.time, a.scale, b.time, b.scale),
                   img->getCudaGlobal(),
                   make_float2( pos.time, pos.scale ),
                   make_float2( 1.f/deltat, 1.f/deltascale ),
                   brush_factor );

    Heightmap::pBlock block = ref.collection()->getBlock( ref );
    GpuCpuData<float>* blockData = block->glblock->height()->data.get();
    ::multiplyGauss( make_float4(a.time, a.scale, b.time, b.scale),
                   blockData->getCudaGlobal(),
                   make_float2( pos.time, pos.scale ),
                   make_float2( 1.f/deltat, 1.f/deltascale ),
                   brush_factor );
    ref.collection()->computeSlope( block, 0 );

    return Signal::Interval(a.time*fs, b.time*fs);
}


} // namespace Tools
