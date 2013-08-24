#include "brushmodel.h"

#include "heightmap/block.h"
#include "heightmap/glblock.h"
#include "heightmap/collection.h"
#include "heightmap/referenceinfo.h"

#include "sawe/project.h"
#include "tfr/cwt.h"

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
    EXCEPTION_ASSERTX(false, "Use Signal::Processing namespace");
/*
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
                    (*imgcopy)[ ref ] = v.second;
                }
                b->images = imgcopy;
            }
            o = o->source();
        }
    }
*/
}


Support::BrushFilter* BrushModel::
        filter()
{
    Support::MultiplyBrush* brush = dynamic_cast<Support::MultiplyBrush*>(filter_.get());
    if (0 == brush)
    {
        filter_.reset( brush = new Support::MultiplyBrush(project_->tools().render_model.tfr_mapping ()) );
        filter_->source( project_->head->head_source() );
    }

    return brush;
}


void BrushModel::
        finished_painting()
{
    if (filter()->images && filter()->images->size())
    {
        project_->appendOperation( filter_ ); // Insert cache layer
    }

    filter_.reset();
}


Gauss BrushModel::
       getGauss( Heightmap::Reference ref, Heightmap::Position pos )
{
    TIME_BRUSH TaskTimer tt("BrushModel::paint( %s, (%g, %g) )", ref.toString().c_str(), pos.time, pos.scale );
    Heightmap::Region region = Heightmap::ReferenceRegion( tfr_mapping() )( ref);

    Tools::Support::TransformDescs::WritePtr td (render_model_->transform_descs ());
    Tfr::Cwt& cwt = td->getParam<Tfr::Cwt>();
    float fs = filter()->sample_rate();
    float hz = render_model_->display_scale().getFrequency( pos.scale );
    float deltasample = cwt.morlet_sigma_samples( fs, hz ) * cwt.wavelet_default_time_support()*0.9f;
    float deltascale = cwt.sigma() * cwt.wavelet_scale_support()*0.9f / cwt.nScales(fs);
    float deltat = deltasample / fs;
    deltat *= std_t;
    deltascale *= 0.05;
    float xscale = region.time();
    float yscale = region.scale();
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

    Heightmap::Reference
            right = ref,
            left = ref,
            top = ref,
            bottom = ref;

    float threshold = 0.001f;
    Heightmap::ReferenceRegion rr( tfr_mapping() );
    for (unsigned &x = left.block_index[0]; ; --x)
    {
        Heightmap::Region region = rr( left);
        if (0 == region.a.time || threshold > fabsf(gauss.gauss_value( region.a.time, pos.scale )))
            break;
    }
    for (unsigned &x = right.block_index[0]; ; ++x)
    {
        Heightmap::Region region = rr( right );
        if (threshold > fabsf(gauss.gauss_value( region.b.time, pos.scale )))
            break;
    }
    for (unsigned &y = bottom.block_index[1]; ; --y)
    {
        Heightmap::Region region = rr( bottom );
        if (0 == region.a.scale || threshold > fabsf(gauss.gauss_value( pos.time, region.a.scale )))
            break;
    }
    for (unsigned &y = top.block_index[1]; ; ++y)
    {
        Heightmap::Region region = rr( top );
        if (1 <= region.b.scale || threshold > fabsf(gauss.gauss_value( pos.time, region.b.scale )))
            break;
    }

    Signal::Intervals r;

    unsigned &x = ref.block_index[0],
             &y = ref.block_index[1];

    for (x = left.block_index[0]; x<=right.block_index[0]; ++x )
        for (y = bottom.block_index[1]; y<=top.block_index[1]; ++y )
            r |= addGauss(ref, gauss);

    return r.spannedInterval();
}


const Heightmap::TfrMapping BrushModel::
        tfr_mapping()
{
    return render_model_->tfr_mapping();
}


Signal::Interval BrushModel::
        addGauss( Heightmap::Reference ref, Gauss gauss )
{
    Heightmap::Region r = Heightmap::ReferenceRegion( tfr_mapping() )( ref );

    TIME_BRUSH TaskInfo("Painting gauss to ref=%s",
                        ref.toString().c_str());

    // TODO filter()->images should be way smaller compared to the high
    // resolution of collection refs.
    Support::BrushFilter::BrushImageDataP& img = (*filter()->images)[ ref ];

    if (!img)
    {
        Heightmap::BlockSize block_size = tfr_mapping().block_size;
        img.reset( new DataStorage<float>(
                       block_size.texels_per_row (), block_size.texels_per_column (), 1));
    }

    ResampleArea area( r.a.time, r.a.scale, r.b.time, r.b.scale );
    ::addGauss( area,
                   img,
                   gauss );

    foreach( const Heightmap::Collection::Ptr& collection, render_model_->collections() )
    {
        Heightmap::pBlock block = write1(collection)->getBlock( ref );
        if (block)
        {
            Heightmap::BlockData::WritePtr block_data( block->block_data() );
            Heightmap::BlockData::pData data = block_data->cpu_copy;
            //Heightmap::BlockData::pData data = block->glblock->height()->data;

            ::multiplyGauss( area,
                           data,
                           gauss, render_model_->amplitude_axis() );
            // collection->invalidate_samples is called by brushcontroller on mouse release
        }
    }

    return Heightmap::ReferenceInfo(ref, tfr_mapping()).getInterval ();
}


} // namespace Tools
