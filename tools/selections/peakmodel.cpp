#include "peakmodel.h"
#include "support/peakfilter.h"
#include "tools/renderview.h"
#include "support/watershead.cu.h"

namespace Tools { namespace Selections
{

PeakModel::PeakModel()
    :
    filter( new Filters::PeakFilter )
{
}


Filters::PeakFilter* PeakModel::
        peak_filter()
{
    return dynamic_cast<Filters::PeakFilter*>(filter.get());
}


void PeakModel::
        findAddPeak( Heightmap::Reference ref, Heightmap::Position pos )
{
/*    Heightmap::pBlock block = ref.collection()->getBlock( ref );
    GpuCpuData<float>* blockData = block->glblock->height()->data.get();

    Heightmap::Position a, b;
    ref.getArea( a, b );

    GpuCpuData<float> intermediate(
        0,
        make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
        GpuCpuVoidData::CudaGlobal );
    cudaMemset( intermediate.getCudaGlobal().ptr(), 0,
                intermediate.getSizeInBytes1D() );

    ::watershed(
            make_float4(a.time, a.scale, b.time, b.scale),
            make_float2(pos.scale, pos.time),
            blockData->getCudaGlobal(),
            intermediate.getCudaGlobal());


    Support::BrushFilter::BrushImageDataP& img = (*peak_filter()->brush.images)[ ref ];

    if (!img)
    {
        img.reset( new GpuCpuData<float>(
            0,
            make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
            GpuCpuVoidData::CudaGlobal ) );
        cudaMemset( img->getCudaGlobal().ptr(), 0, img->getSizeInBytes1D() );
    }

    ::watershedApplyIntermediate(
            intermediate.getCudaGlobal(),
            img->getCudaGlobal() );*/
}

}} // Tools::Selections
