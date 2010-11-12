#include "brushmodel.h"
#include <demangle.h>

#include "tfr/cwt.h"

namespace Tools {


BrushModel::
        BrushModel( Sawe::Project* project )
            :
            brush_factor(0),
            xscale_(10)
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

    xscale_ = b.time-a.time;
    if (xscale_ < 0.5)
        xscale_ = 0.5;

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    float fs = filter()->sample_rate();
    float hz = cwt.compute_frequency2( fs, pos.scale );
    float deltasample = Tfr::Cwt::Singleton().morlet_sigma_t( fs, hz );
    float deltascale = cwt.sigma() / cwt.nScales(fs);
    float deltat = deltasample/fs;
    deltat *= 10*xscale_;
    deltascale *= 0.1;

    Gauss gauss(
            make_float2( pos.time, pos.scale ),
            make_float2( deltat, deltascale ),
            brush_factor);

    Heightmap::Reference
            right = ref,
            left = ref,
            top = ref,
            bottom = ref;

    float threshold = 0.001f;
    for (unsigned &x = left.block_index[0]; ; --x)
    {
        left.getArea(a, b);
        if (0 == a.time || threshold > gauss.gauss_value(make_float2( a.time, pos.scale )))
            break;
    }
    for (unsigned &x = right.block_index[0]; ; ++x)
    {
        right.getArea(a, b);
        if (threshold > gauss.gauss_value(make_float2( b.time, pos.scale )))
            break;
    }
    for (unsigned &y = bottom.block_index[1]; y>0; --y)
    {
        bottom.getArea(a, b);
        if (0 == a.scale || threshold > gauss.gauss_value(make_float2( pos.time, a.scale )))
            break;
    }
    for (unsigned &y = top.block_index[1]; ; ++y)
    {
        top.getArea(a, b);
        if (1 >= b.scale || threshold > gauss.gauss_value(make_float2( pos.time, b.scale )))
            break;
    }

    Signal::Intervals r;

    unsigned &x = ref.block_index[0],
             &y = ref.block_index[1];

    for (x = left.block_index[0]; x<=right.block_index[0]; ++x )
        for (y = bottom.block_index[1]; y<=top.block_index[1]; ++y )
            r |= addGauss(ref, gauss);

    return r.coveredInterval();
}


Signal::Interval BrushModel::
        addGauss( Heightmap::Reference ref, Gauss gauss )
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

    Heightmap::Position a, b;
    ref.getArea( a, b );

    TaskTimer tt("Painting guass [(%g %g), (%g %g)]", a.time, a.scale, b.time, b.scale );

    ::addGauss( make_float4(a.time, a.scale, b.time, b.scale),
                   img->getCudaGlobal(),
                   gauss );

    Heightmap::pBlock block = ref.collection()->getBlock( ref );
    GpuCpuData<float>* blockData = block->glblock->height()->data.get();
    ::multiplyGauss( make_float4(a.time, a.scale, b.time, b.scale),
                   blockData->getCudaGlobal(),
                   gauss );
    ref.collection()->computeSlope( block, 0 );

    return ref.getInterval();
}


} // namespace Tools
