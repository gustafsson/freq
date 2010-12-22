#include "brushmodel.h"
#include <demangle.h>

#include "tfr/cwt.h"

#define TIME_BRUSH
//#define TIME_BRUSH if(0)

namespace Tools {


BrushModel::
        BrushModel( Sawe::Project* project, RenderModel* render_model )
            :
            brush_factor(0),
            render_model_(render_model)
{
    filter_ = project->head_source();
	if (0 == dynamic_cast<Support::MultiplyBrush*>(filter_.get()))
	{
		filter_.reset( new Support::MultiplyBrush );
		filter_->source( project->head_source() );
		project->head_source( filter_ );
	} 

    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, render_model->collections )
        filter()->validateRefs( collection.get() );
}


Support::BrushFilter* BrushModel::
        filter()
{
    return dynamic_cast<Support::BrushFilter*>(filter_.get());
}


Signal::Interval BrushModel::
        paint( Heightmap::Reference ref, Heightmap::Position pos )
{
    TIME_BRUSH TaskTimer tt("BrushModel::paint( %s, (%g, %g) )", ref.toString().c_str(), pos.time, pos.scale );
    Heightmap::Position a, b;
    ref.getArea(a, b);

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    float fs = filter()->sample_rate();
    float hz = cwt.compute_frequency2( fs, pos.scale );
    float deltasample = Tfr::Cwt::Singleton().morlet_sigma_samples( fs, hz );
    float deltascale = cwt.sigma() / cwt.nScales(fs);
    float deltat = deltasample/fs;
    deltat *= 1;
    deltascale *= 0.1;
    float xscale = b.time-a.time;
    float yscale = b.scale-a.scale;
    if (deltat < xscale*0.05)
        deltat = xscale*0.05;
    if (deltascale < yscale*0.05)
        deltascale = yscale*0.05;

    Gauss gauss(
            make_float2( pos.time, pos.scale ),
            make_float2( deltat, deltascale ),
            brush_factor);

    TIME_BRUSH TaskInfo("Created gauss mu=(%g, %g), sigma=(%g, %g), height=%g, xscale=%g",
                        pos.time, pos.scale, gauss.sigma().x, gauss.sigma().y, brush_factor, xscale);

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
    Heightmap::Position a, b;
    ref.getArea( a, b );

    TIME_BRUSH TaskInfo("Painting gauss to ref=%s",
                        ref.toString().c_str());

    // TODO filter()->images should be way smaller compared to the high
    // resolution of collection refs.
    Support::BrushFilter::BrushImageDataP& img = (*filter()->images)[ ref ];

    if (!img)
    {
        img.reset( new GpuCpuData<float>(
            0,
            make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
            GpuCpuVoidData::CudaGlobal ) );
        cudaMemset( img->getCudaGlobal().ptr(), 0, img->getSizeInBytes1D() );
    }

    ::addGauss( make_float4(a.time, a.scale, b.time, b.scale),
                   img->getCudaGlobal(),
                   gauss );

    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, render_model_->collections )
    {
        Heightmap::pBlock block = collection->getBlock( ref );
        if (block)
        {
            GpuCpuData<float>* blockData = block->glblock->height()->data.get();
            ::multiplyGauss( make_float4(a.time, a.scale, b.time, b.scale),
                           blockData->getCudaGlobal(),
                           gauss );
            // collection->invalidate_samples is called by brushcontroller on mouse release
        }
    }

    return ref.getInterval();
}


} // namespace Tools
