#include "peakmodel.h"

#include "support/peakfilter.h"
#include "tools/renderview.h"
#include "support/watershead.cu.h"
#include "tfr/cwt.h"
#include "tools/support/brushpaint.cu.h"

#include <boost/foreach.hpp>

namespace Tools { namespace Selections
{

PeakModel::PeakModel( Tfr::FreqAxis const& fa )
    :   spline_model( fa )
{
}


Support::SplineFilter* PeakModel::
        peak_filter()
{
    return dynamic_cast<Support::SplineFilter*>(spline_model.filter.get());
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

    return itr->second->getCpuMemory()[ (x%w) + (y%h)*w ];
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

    pixel_count = 0;
    recursivelyClassify(
            ref,
            ref.samplesPerBlock(), ref.scalesPerBlock(),
            x0, y0, PS_Increasing, -FLT_MAX );

    // Discard image data from CPU
    BOOST_FOREACH( PeakAreas::value_type const& v, classifictions )
    {
        Heightmap::pBlock block = ref.collection()->getBlock( v.first );
        GpuCpuData<float>* blockData = block->glblock->height()->data.get();
        blockData->getCudaGlobal( false );
    }

    findBorder();

    // Translate nodes to scale and time

    Heightmap::Position elementSize(
            ldexpf(1.f,ref.log2_samples_size[0]),
            ldexpf(1.f,ref.log2_samples_size[1]));

    std::vector<Heightmap::Position> &v = spline_model.v;
    unsigned N=border_nodes.size();
    v.resize(N);

    for (unsigned i=0; i<N; ++i)
    {
        Heightmap::Position p;
        p.time = border_nodes[i].x * elementSize.time;
        p.scale = border_nodes[i].y * elementSize.scale;
        v[i] = p;
    }

    spline_model.updateFilter();
}


void PeakModel::
        findBorder()
{
    // Find range of classified pixels
    BOOST_ASSERT(!classifictions.empty());

    Heightmap::Reference ref = classifictions.begin()->first;
    unsigned
            w = ref.samplesPerBlock(),
            h = ref.scalesPerBlock();

    uint2 start_point;
    if (!anyBorderPixel(start_point, w, h))
        return;

    bool val = classifiedVal(start_point.x, start_point.y, w, h);

    border_nodes.clear();
    std::vector<uint2> border_pts;

    unsigned firstdir = 0;
    start_point = nextBorderPixel(start_point, w, h, firstdir);
    uint2 pos = start_point;
    border_nodes.push_back( start_point );
    do
    {
        pos = nextBorderPixel(pos, w, h, firstdir);

        if (1<border_pts.size())
        {
            // Define a line from 'lastnode' to 'pos' and check if all points in
            // 'border_pts' is less than or equal to 1 unit away from the line
            uint2& lastnode = border_nodes.back();
            float2 d = make_float2(pos.x - lastnode.x,
                                   pos.y - lastnode.y);
            float r = 1.f/sqrtf(d.x*d.x + d.y*d.y);
            d.x *= r;
            d.y *= r;

            unsigned i;
            for (i=0; i<border_pts.size(); ++i)
            {
                float2 q = make_float2( border_pts[i].x - lastnode.x,
                                        border_pts[i].y - lastnode.y );

                float dot = q.x*d.y + q.y*d.x;
                if (dot*dot > 2)
                    break;
            }

            if (i<border_pts.size()) // nope not ok,
            {
                border_nodes.push_back( border_pts.back() );
                printf("%d, %d;\n", border_pts.back().x, border_pts.back().y);
                fflush(stdout);
                border_pts.clear();
            }
        }

        border_pts.push_back( pos );
    } while(pos.x!=start_point.x || pos.y!=start_point.y);
}


bool PeakModel::
        anyBorderPixel( uint2& pos, unsigned w, unsigned h )
{
    BOOST_FOREACH(PeakAreas::value_type v, classifictions)
    {
        bool *b = v.second->getCpuMemory();

        for (unsigned y=0; y<h; ++y)
        {
            for (unsigned x=0; x<w; ++x)
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
        nextBorderPixel( uint2 v, unsigned w, unsigned h, unsigned& firstdir )
{
    int2 p[] =
    { // walk clockwise
        {+1, +0},
        {+1, +1},
        {+0, +1},
        {-1, +1},
        {-1, +0},
        {-1, -1},
        {+0, -1},
        {+1, -1}
    };

    unsigned N = sizeof(p)/sizeof(p[0]);

    bool prev = true;
    for (unsigned i=firstdir; i<firstdir+N+1; ++i)
    {
        unsigned j=i%N;
        uint2 r = make_uint2(v.x + p[j].x, v.y + p[j].y);
        bool b = classifiedVal(r.x, r.y, w, h);
        if (b && !prev)
        {
            firstdir = (i + 4)%N;
            return r;
        }
        prev = b;
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

    bool wasOut = classification[x+y*w] == 0;
    if (!wasOut)
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
        pixel_count++;
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
            if (0==y) {
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
