#include "peakmodel.h"

#include "support/peakfilter.h"
#include "tools/renderview.h"
#include "support/watershead.cu.h"
#include "tfr/cwt.h"
#include "tools/support/brushpaint.cu.h"

#include <boost/foreach.hpp>

namespace Tools { namespace Selections
{

PeakModel::PeakModel()
    :
    filter( new Support::SplineFilter )
{
}


Support::SplineFilter* PeakModel::
        peak_filter()
{
    return dynamic_cast<Support::SplineFilter*>(filter.get());
}


PeakModel::PeakAreaP PeakModel::
        getPeakArea(Heightmap::Reference ref)
{
    PeakAreaP& area = classifictions[ ref ];

    if (!area)
    {
        area.reset( new GpuCpuData<bool>(
            0,
            make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
            GpuCpuVoidData::CpuMemory ) );
        memset( area->getCpuMemory(), 0, area->getSizeInBytes1D() );
    }

    return area;
}

/*
PeakModel::PeakAreaP PeakModel::
        getPeakAreaGauss(Heightmap::Reference ref)
{
    PeakAreaP& area = gaussed_classifictions[ ref ];

    if (!area)
    {
        area.reset( new GpuCpuData<float>(
            0,
            make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
            GpuCpuVoidData::CpuMemory ) );
        memset( area->getCpuMemory(), 0, area->getSizeInBytes1D() );
    }

    return area;
}
*/

bool PeakModel::
        classifiedVal(unsigned x, unsigned y, unsigned w, unsigned h)
{
    Heightmap::Reference ref = classifictions.begin()->first;
    ref.block_index[0] = x/w;
    ref.block_index[1] = y/h;

    PeakAreas::iterator itr = classifictions.find( ref );
    if (itr == classifictions.end())
        return 0;

    return itr->second->getCpuMemory()[ x + y*w ];
}

/*
float& PeakModel::
        gaussedVal(unsigned x, unsigned y, unsigned w, unsigned h)
{
    Heightmap::Reference ref = classifictions.begin()->first;
    ref.block_index[0] = x/w;
    ref.block_index[1] = y/h;

    float* p = getPeakAreaGauss(ref)->getCpuMemory();
    return p[ x + y*w ];
}
*/

void PeakModel::
        findAddPeak( Heightmap::Reference ref, Heightmap::Position pos )
{
    Heightmap::Position a, b;
    ref.getArea( a, b );
    unsigned h = ref.scalesPerBlock();
    unsigned w = ref.samplesPerBlock();
    unsigned y0 = (pos.scale-a.scale)/(b.scale-a.scale)*(h-1) + .5f;
    unsigned x0 = (pos.time-a.time)/(b.time-a.time)*(w-1) + .5f;

    classifictions.clear();

    recursivelyClassify(
            ref,
            ref.samplesPerBlock(), ref.scalesPerBlock(),
            x0, y0, PS_Increasing, -FLT_MAX );

    findBorder();
}


void PeakModel::
        findBorder()
{
    std::vector<Support::SplineFilter::SplineVertex>& vertices = peak_filter()->v;
    vertices.clear();

    // Find range of classified pixels
    BOOST_ASSERT(!classifictions.empty());

    Heightmap::Reference ref = classifictions.begin()->first;
    unsigned
            w = ref.samplesPerBlock(),
            h = ref.scalesPerBlock();

    uint2 start_point;
    if (!anyBorderPixel(start_point, w, h))
        return;

    uint2 pos = start_point;
    uint2 lastnode = start_point;
    peak_filter()->v
    do
    {

        pos = nextBorderPixel(pos, w, h);

        if ()

    } while(pos!=start_point);
}


bool PeakModel::
        anyBorderPixel( uint2& pos, unsigned w, unsigned h )
{
    BOOST_FOREACH(PeakAreas::value_type v, classifictions)
    {
        bool *b = v.second->getCpuMemory();

        for (unsigned y=0; y<h; ++y)
        {
            for (unsigned x=0; x<h; ++x)
            {
                if (b[ x + y*w ])
                {
                    pos = make_uint2( x + v.first.block_index[0]*w,
                                      y + v.first.block_index[1]*h);
                    return true;
                }
            }
        }
    }

    return false;
}


uint2 PeakModel::
        nextBorderPixel( uint2 v, unsigned w, unsigned h )
{
    uint2 p[] =
    { // walk clockwise
        {v.x+1, v.y+0},
        {v.x+1, v.y+1},
        {v.x+0, v.y+1},
        {v.x-1, v.y+1},
        {v.x-1, v.y+0},
        {v.x-1, v.y-1},
        {v.x+0, v.y-1},
        {v.x+1, v.y-1}
    };

    unsigned N = sizeof(p)/sizeof(p[0]);
    bool prev = classifiedVal(p[N-1].x, p[N-1].y, w, h);
    for (unsigned i=0; i<N; ++i)
    {
        bool v = classifiedVal(p[i].x, p[i].y, w, h);
        if (v && !prev)
            return p[i];
    }
    return v;
}


/*void PeakModel::
        findAddPeak( Heightmap::Reference ref, Heightmap::Position pos )
{
    Heightmap::Position a, b;
    ref.getArea( a, b );
    unsigned h = ref.scalesPerBlock();
    unsigned w = ref.samplesPerBlock();
    unsigned y0 = (pos.scale-a.scale)/(b.scale-a.scale)*(h-1) + .5f;
    unsigned x0 = (pos.time-a.time)/(b.time-a.time)*(w-1) + .5f;

    classifictions.clear();

    recursivelyClassify(
            ref,
            ref.samplesPerBlock(), ref.scalesPerBlock(),
            x0, y0, PS_Increasing, -FLT_MAX );

    smearGauss();

    BOOST_FOREACH(PeakAreas::value_type v, gaussed_classifictions)
    {
        Support::BrushFilter::BrushImageDataP img = peak_filter()->brush.getImage( v.first );
        PeakAreaP gauss = v.second;

        float* p = img->getCpuMemory();
        float* q = gauss->getCpuMemory();

        for (unsigned y=0; y<h; ++y)
            for (unsigned x=0; x<w; ++x)
                p[x + y*w] = std::max( p[x + y*w], q[x + y*w]);
    }
}

void PeakModel::
        smearGauss()
{
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(peak_filter()->transform().get());


    BOOST_ASSERT(!classifictions.empty());
    Heightmap::Reference ref = classifictions.begin()->first;

    Heightmap::Reference
            min_ref = ref,
            max_ref = ref;


    BOOST_FOREACH(PeakAreas::value_type v, classifictions)
    {
        for (int i=0; i<2; ++i)
        {
            if (min_ref.block_index[i] > v.first.block_index[i])
                min_ref.block_index[i] = v.first.block_index[i];
            if (max_ref.block_index[i] < v.first.block_index[i])
                max_ref.block_index[i] = v.first.block_index[i];
        }
    }

    Heightmap::Position a,b;
    min_ref.getArea(a,b);
    Tfr::FreqAxis const& fa = min_ref.collection()->display_scale();
    float min_hz = fa.getFrequency(a.scale);
    float t_sigma = cwt->morlet_sigma_t( 1, min_hz );
    float f_sigma = log2f(1.0 / (2*M_PI* cwt->sigma() ))/log2f(fa.getFrequency(1.f)/fa.getFrequency(0.f));
    float t_support = t_sigma * cwt->wavelet_time_support();
    float f_support = f_sigma + log2f( cwt->wavelet_scale_support());
    float min_scale = a.scale - f_support;
    float min_t = a.time - t_support;

    while(a.scale > 0 && a.scale > min_scale)
    {
        min_ref = min_ref.sibblingBottom();
        min_ref.getArea(a,b);
    }
    while(a.time > 0 && a.time > min_t)
    {
        min_ref = min_ref.sibblingLeft();
        min_ref.getArea(a,b);
    }
    max_ref.getArea(a,b);
    float max_scale = b.scale + f_support;
    float max_t = b.time + t_support;
    while(b.scale < 1 && b.scale < max_scale)
    {
        max_ref = max_ref.sibblingTop();
        max_ref.getArea(a,b);
    }
    while(b.time < max_t)
    {
        max_ref = min_ref.sibblingRight();
        max_ref.getArea(a,b);
    }

    unsigned
            w = ref.samplesPerBlock(),
            h = ref.scalesPerBlock();
    // Terribly nested loop, but whatever, in runs once every mouse click
    // Its ok to take a few milliseconds to compute
    unsigned
            global_min_t = min_ref.block_index[0] * w,
            global_max_t = max_ref.block_index[0] * w-1,
            global_min_s = min_ref.block_index[1] * h,
            global_max_s = max_ref.block_index[1] * h-1;

    // Compute convolution with Gauss function
    Gauss g(make_float2(0,0),
            make_float2( t_sigma * ldexpf(1.f,ref.log2_samples_size[0]),
                         f_sigma * ldexpf(1.f,ref.log2_samples_size[1])));

    for (unsigned y=global_min_s; y < global_max_s; ++y )
        for (unsigned x=global_min_t; x < global_max_t; ++x )
        {
            g.pos = make_float2( x, y );

            float sum = 0;
            for (unsigned v=global_min_s; v < global_max_s; ++v )
                for (unsigned u=global_min_t; u < global_max_t; ++u )
                    sum += classifiedVal( u, v, w, h ) * g.gauss_value( u, v );

            gaussedVal(x, y, w, h) = sum;
        }
}
*/

void PeakModel::
        recursivelyClassify( Heightmap::Reference ref,
                             unsigned w, unsigned h,
                             unsigned x, unsigned y,
                             PropagationState prevState, float prevVal
                             )
{
    Heightmap::pBlock block = ref.collection()->getBlock( ref );
    GpuCpuData<float>* blockData = block->glblock->height()->data.get();
    float* data = blockData->getCpuMemory();

    PeakAreaP area = getPeakArea(ref);
    bool* classification = area->getCpuMemory();

    recursivelyClassify(ref, data, classification,
                        w, h, x, y, prevState, prevVal );
}


void PeakModel::
        recursivelyClassify( Heightmap::Reference ref,
                             float *data, bool* classification,
                             unsigned w, unsigned h,
                             unsigned x, unsigned y,
                             PropagationState prevState, float prevVal )
{
    if (x>=w)
    {
        recursivelyClassify(ref.sibblingRight(), w, h, x-w, y, prevState, prevVal );
        return;
    }
    if (y>=h)
    {
        recursivelyClassify(ref.sibblingTop(), w, h, x, y-h, prevState, prevVal );
        return;
    }
    if (prevState==PS_Out)
        return;

    float val = data[x + y*w];
    PropagationState state;
    if (val>prevVal)
        state = PS_Increasing;
    else if (val==prevVal)
        state = prevState;
    else
        state = PS_Decreasing;

    if (prevState>state)
        state = PS_Out;

    classification[x+y*w] = state != PS_Out;

    if (state != PS_Out)
    {
        recursivelyClassify( ref, data, classification,
                             w, h,
                             x+1, y, state, val );
        recursivelyClassify( ref, data, classification,
                             w, h,
                             x, y+1, state, val );

        if (val!=prevVal)
        {
            if (0==x) {
                if (0<ref.block_index[0])
                    recursivelyClassify( ref.sibblingLeft(),
                                         w, h,
                                         w-1, y, state, val );

            } else {
                recursivelyClassify( ref, data, classification,
                                     w, h,
                                     x-1, y, state, val );
            }
            if (0==x) {
                if (0<ref.block_index[1])
                    recursivelyClassify( ref.sibblingBottom(),
                                         w, h,
                                         x, h-1, state, val );
            } else {
                recursivelyClassify( ref, data, classification,
                                     w, h,
                                     x, y-1, state, val );
            }
        }
    }
}

}} // Tools::Selections
