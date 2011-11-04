#include "brushmodel.h"

#include "sawe/project.h"
#include "tfr/cwt.h"
#include "cudaglobalstorage.h"

#include <demangle.h>


#define TIME_BRUSH
//#define TIME_BRUSH if(0)

namespace Tools {


BrushModel::
        BrushModel( Sawe::Project* project, RenderModel* render_model )
            :
            brush_factor(0),
            std_t(1),
            render_model_(render_model),
            project_(project)
{
/*    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, render_model_->collections )
        brush->validateRefs( collection.get() );*/
    foreach (Signal::pChain c, project->layers.layers())
    {
        Signal::pOperation o = c->tip_source();
        while(o)
        {
            Support::BrushFilter* b = dynamic_cast<Support::BrushFilter*>(o.get());
            if (b && b->images)
            {
                Support::BrushFilter::BrushImagesP imgcopy( new Support::BrushFilter::BrushImages );

                foreach (Support::BrushFilter::BrushImages::value_type v, *b->images)
                {
                    Heightmap::Reference ref = v.first;
                    ref.setCollection( render_model->collections[0].get() );
                    (*imgcopy)[ ref ] = v.second;
                }
                b->images = imgcopy;
            }
            o = o->source();
        }
    }
}


Support::BrushFilter* BrushModel::
        filter()
{
    Support::MultiplyBrush* brush = dynamic_cast<Support::MultiplyBrush*>(filter_.get());
    if (0 == brush)
    {
        filter_.reset( brush = new Support::MultiplyBrush );
        filter_->source( project_->head->head_source() );
    }

    return brush;
}


void BrushModel::
        finished_painting()
{
    if (filter()->images && filter()->images->size())
        project_->appendOperation( filter_ ); // Insert cache layer

    filter_.reset();
}


Gauss BrushModel::
       getGauss( Heightmap::Reference ref, Heightmap::Position pos )
{
    TIME_BRUSH TaskTimer tt("BrushModel::paint( %s, (%g, %g) )", ref.toString().c_str(), pos.time, pos.scale );
    Heightmap::Position a, b;
    ref.getArea(a, b);

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    float fs = filter()->sample_rate();
    float hz = render_model_->display_scale().getFrequency( pos.scale );
    float deltasample = cwt.morlet_sigma_samples( fs, hz ) * cwt.wavelet_default_time_support()*0.9f;
    float deltascale = cwt.sigma() * cwt.wavelet_scale_support()*0.9f / cwt.nScales(fs);
    float deltat = deltasample / fs;
    deltat *= std_t;
    deltascale *= 0.05;
    float xscale = b.time-a.time;
    float yscale = b.scale-a.scale;
    if (deltat < xscale*0.02)
        deltat = xscale*0.02;
    if (deltascale < yscale*0.01)
        deltascale = yscale*0.01;

    Gauss gauss(
            ResamplePos( pos.time, pos.scale ),
            ResamplePos( deltat, deltascale ),
            brush_factor);

    TIME_BRUSH TaskInfo("Created gauss mu=(%g, %g), sigma=(%g, %g), height=%g, xscale=%g",
                        pos.time, pos.scale, gauss.sigma().x, gauss.sigma().y, brush_factor, xscale);

    return gauss;
}


Signal::Interval BrushModel::
        paint( Heightmap::Reference ref, Heightmap::Position pos )
{
    Gauss gauss = getGauss( ref, pos );

    Heightmap::Position a, b;

    Heightmap::Reference
            right = ref,
            left = ref,
            top = ref,
            bottom = ref;

    float threshold = 0.001f;
    for (unsigned &x = left.block_index[0]; ; --x)
    {
        left.getArea(a, b);
        if (0 == a.time || threshold > fabsf(gauss.gauss_value( a.time, pos.scale )))
            break;
    }
    for (unsigned &x = right.block_index[0]; ; ++x)
    {
        right.getArea(a, b);
        if (threshold > fabsf(gauss.gauss_value( b.time, pos.scale )))
            break;
    }
    for (unsigned &y = bottom.block_index[1]; ; --y)
    {
        bottom.getArea(a, b);
        if (0 == a.scale || threshold > fabsf(gauss.gauss_value( pos.time, a.scale )))
            break;
    }
    for (unsigned &y = top.block_index[1]; ; ++y)
    {
        top.getArea(a, b);
        if (1 <= b.scale || threshold > fabsf(gauss.gauss_value( pos.time, b.scale )))
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
        img.reset( new DataStorage<float>(
                ref.samplesPerBlock(), ref.scalesPerBlock(), 1));
        cudaMemset( CudaGlobalStorage::WriteAll<1>(img).device_ptr(), 0, img->numberOfBytes() );
    }

    ImageArea area = {a.time, a.scale, b.time, b.scale};
    ::addGauss( area,
                   img,
                   gauss );

    foreach( const boost::shared_ptr<Heightmap::Collection>& collection, render_model_->collections )
    {
        Heightmap::pBlock block = collection->getBlock( ref );
        if (block)
        {
            Heightmap::Block::pData blockData = block->glblock->height()->data;

            ::multiplyGauss( area,
                           blockData,
                           gauss, render_model_->amplitude_axis() );
            // collection->invalidate_samples is called by brushcontroller on mouse release
        }
    }

    return ref.getInterval();
}


} // namespace Tools
