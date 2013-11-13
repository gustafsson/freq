#include "peakmodel.h"

// tools
#include "tools/renderview.h"
#include "tools/support/brushpaintkernel.h"
#include "tools/support/operation-composite.h"

// Sonic AWE
#include "tfr/cwt.h"
#include "heightmap/collection.h"
#include "heightmap/block.h"
#include "heightmap/glblock.h"
#include "heightmap/blocklayout.h"

// gpumisc
#include "tvector.h"
#ifdef USE_CUDA
#include "cudaglobalstorage.h"
#endif

// std
#include <queue>

namespace Tools { namespace Selections
{

PeakModel::PeakModel( RenderModel* rendermodel )
    :   spline_model( rendermodel )
{
}


PeakModel::PeakAreaP PeakModel::
        getPeakArea(Heightmap::Reference ref)
{
    PeakAreaP& area = classifictions[ ref ];

    if (!area)
    {
        Heightmap::BlockLayout block_size = c->block_layout();
        area.reset( new DataStorage<bool>(block_size.texels_per_row (), block_size.texels_per_column (), 1));
        EXCEPTION_ASSERT( area->numberOfBytes() == area->numberOfElements());
        memset( area->getCpuMemory(), 0, area->numberOfBytes() );
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


float PeakModel::
        heightVal(Heightmap::Reference ref, unsigned x, unsigned y)
{
    Heightmap::pBlock block = c->getBlock( ref );
    if (!block)
        return 0;
    DataStorage<float>::Ptr blockData = block->glblock->height()->data;
    float* data = blockData->getCpuMemory();

    Heightmap::BlockLayout block_size = c->block_layout();
    return data[x+y*block_size.texels_per_row ()];
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
        findAddPeak( Heightmap::Collection::Ptr cptr, Heightmap::Reference ref, Heightmap::Position pos )
{
    Heightmap::Collection::WritePtr write_ptr(cptr);
    c = &*write_ptr;

    Heightmap::BlockLayout block_size = c->block_layout();
    Heightmap::Region r = Heightmap::RegionFactory(block_size)(ref);
    unsigned h = block_size.texels_per_column ();
    unsigned w = block_size.texels_per_row ();
    unsigned y0 = (pos.scale-r.a.scale)/r.scale()*(h-1) + .5f;
    unsigned x0 = (pos.time-r.a.time)/r.time()*(w-1) + .5f;

    classifictions.clear();

    pixel_count = 0;
    use_min_limit = false;
    min_limit = 0;
    found_max = 0;
    found_min = FLT_MAX;
    pixel_limit = 30000; // 10*pixel_count;

    loopClassify(ref, x0, y0);

    use_min_limit = true;
    classifictions.clear();
    min_limit = found_min + (found_max - found_min)*0.1f;
    middle_limit = heightVal(ref, x0, y0);

    loopClassify(ref, x0, y0);

#ifdef USE_CUDA
    // Discard image data from CPU
    foreach( PeakAreas::value_type const& v, classifictions )
    {
        Heightmap::pBlock block = c->getBlock( v.first );
        if (block)
        {
            DataStorage<float>::Ptr blockData = block->glblock->height()->data;
            blockData->OnlyKeepOneStorage<CudaGlobalStorage>();
        }
    }
#endif

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
        p.time = (border_nodes[i].x + .5f) * elementSize.time;
        p.scale = (border_nodes[i].y + .5f) * elementSize.scale;
        v[i] = p;
    }

    c = 0;
}


void PeakModel::
        findBorder()
{
    // Find range of classified pixels
    if (classifictions.empty())
        return;

    Heightmap::BlockLayout block_size = c->block_layout ();
    unsigned
            w = block_size.texels_per_row (),
            h = block_size.texels_per_column ();

    BorderCoordinates start_point;
    if (!anyBorderPixel(start_point, w, h))
        return;

    border_nodes.clear();
    std::vector<BorderCoordinates> border_pts;

    unsigned firstdir = 0;
    start_point = nextBorderPixel(start_point, w, h, firstdir);
    BorderCoordinates pos = start_point;
    border_nodes.push_back( start_point );
    do
    {
        pos = nextBorderPixel(pos, w, h, firstdir);

        if (1<border_pts.size())
        {
            // Define a line from 'lastnode' to 'pos' and check if all points in
            // 'border_pts' is less than or equal to 1 unit away from the line
            BorderCoordinates& lastnode = border_nodes.back();
            tvector<2, float> d(pos.x - (float)lastnode.x,
                                pos.y - (float)lastnode.y);

            d = d.Normalized();

            unsigned i;
            for (i=0; i<border_pts.size(); ++i)
            {
                tvector<2, float> q(border_pts[i].x - (float)lastnode.x,
                                    border_pts[i].y - (float)lastnode.y );

                //float dot = q.x*d.y + q.y*d.x; what?
                float dot = q[0]*d[1] + q[1]*d[0]; // todo explain
                if (dot*dot > 2)
                    break;
            }

            if (i<border_pts.size()) // nope not ok,
            {
                border_nodes.push_back( border_pts.back() );
                border_pts.clear();
            }
        }

        border_pts.push_back( pos );
    } while(pos.x!=start_point.x || pos.y!=start_point.y);
}


bool PeakModel::
        anyBorderPixel( BorderCoordinates& pos, unsigned w, unsigned h )
{
    foreach(PeakAreas::value_type v, classifictions)
    {
        bool *b = v.second->getCpuMemory();

        for (unsigned y=0; y<h; ++y)
        {
            for (unsigned x=0; x<w; ++x)
            {
                if (b[ x + y*w ])
                {
                    pos = BorderCoordinates(
                            x + v.first.block_index[0]*w,
                            y + v.first.block_index[1]*h);
                    return true;
                }
            }
        }
    }

    return false;
}


PeakModel::BorderCoordinates PeakModel::
        nextBorderPixel( BorderCoordinates v, unsigned w, unsigned h, unsigned& firstdir )
{
    int p[][2] =
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
        BorderCoordinates r(v.x + p[j][0], v.y + p[j][1]);
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

    foreach(PeakAreas::value_type v, gaussed_classifictions)
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


    EXCEPTION_ASSERT(!classifictions.empty());
    Heightmap::Reference ref = classifictions.begin()->first;

    Heightmap::Reference
            min_ref = ref,
            max_ref = ref;


    foreach(PeakAreas::value_type v, classifictions)
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
    Heightmap::Region r = Heightmap::RegionFactory(c->block_layout ())(ref);
    if (r.b.scale > 1 || r.a.scale >= 1)
        return;

    Heightmap::pBlock block = c->getBlock( ref );
    if (!block)
        return;
    DataStorage<float>::Ptr blockData = block->glblock->height()->data;
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

    if (val > found_max) found_max = val;
    if (val < found_min) found_min = val;

    if (val > prevVal)
        state = PS_Increasing;
    else if (val == prevVal)
        state = prevState;
    else
        state = PS_Decreasing;

    if (prevState > state)
        state = PS_Out;

    if (use_min_limit && state == PS_Out)
        if (val >= min_limit)
            state = PS_Increasing;

    bool is_in = state != PS_Out;
    classification[x+y*w] = is_in;

    if (is_in)
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

struct Pt
{
    Pt(Heightmap::Reference ref,
       unsigned x, unsigned y,
       PropagationState prevState,
       float prevVal)
           :
           ref(ref),
           x(x), y(y),
           prevState(prevState),
           prevVal(prevVal)
    {}

    Heightmap::Reference ref;
    unsigned x, y;

    PropagationState prevState;
    float prevVal;
};

void PeakModel::
        loopClassify( Heightmap::Reference ref0,
                      unsigned x0, unsigned y0
                      )
{
    std::vector<Pt> pts;

    pts.push_back( Pt(ref0, x0, y0, PS_Increasing, -FLT_MAX) );

    Heightmap::BlockLayout block_size = c->block_layout();
    unsigned w = block_size.texels_per_row (),
             h = block_size.texels_per_column ();

    pixel_count = 0;

    float* data=0;
    bool* classification=0;
    Heightmap::Reference prevRef = ref0.parent();
    prevRef.block_index[0] = -1;
    prevRef.block_index[1] = -1;

    while (!pts.empty())
    {
        if (use_min_limit && pixel_count > pixel_limit )
        {
            this->classifictions.clear();
            return;
        }

        Pt inf = pts.back();
        Heightmap::Reference& ref = inf.ref;
        unsigned& x = inf.x;
        unsigned& y = inf.y;
        PropagationState& prevState = inf.prevState;
        float& prevVal = inf.prevVal;

        pts.pop_back();

        if (x>=w)
        {
            ref = ref.sibblingRight();
            x -= w;
        }
        if (y>=h)
        {
            ref = ref.sibblingTop();
            y -= h;

            Heightmap::Region r = Heightmap::RegionFactory(c->block_layout ())(ref);
            if (r.a.scale >= 1 || r.b.scale > 1 )
            {
                this->classifictions.clear();
                return;
            }
        }

        if (!(ref == prevRef))
        {
            Heightmap::pBlock block = c->getBlock( ref );
            if (!block)
            {
                // Would have needed unavailable blocks to compute this,
                // abort instead
                this->classifictions.clear();
                return;
            }

            DataStorage<float>::Ptr blockData = block->glblock->height()->data;
            data = blockData->getCpuMemory();

            PeakAreaP area = getPeakArea(ref);
            classification = area->getCpuMemory();

            prevRef = ref;
        }

        bool wasOut = classification[x + y*w] == 0;
        if (!wasOut)
            continue;


        float val = data[x + y*w];

        PropagationState state;

        if (val > found_max) found_max = val;
        if (val < found_min) found_min = val;


        if (val > prevVal)
            state = PS_Increasing;
        else if (val == prevVal)
            state = prevState;
        else
            state = PS_Decreasing;

        if (prevState > state)
            state = PS_Out;

        bool is_in, go_further;

        if (!use_min_limit)
        {
            is_in = state != PS_Out;
            go_further = is_in;
        }
        else
        {
            is_in = (val >= min_limit);
            if (val >= middle_limit)
                go_further = true;
            else
                go_further = is_in && state == PS_Decreasing;
        }

        classification[x + y*w] = is_in;

        if (go_further)
        {
            pixel_count++;

            pts.push_back( Pt(ref,x+1,y,state,val));
            pts.push_back( Pt(ref,x,y+1,state,val));

            if (val != prevVal)
            {
                if (0==x) {
                    if (0<ref.block_index[0])
                        pts.push_back( Pt(ref.sibblingLeft(),w-1,y,state,val));

                } else {
                    pts.push_back( Pt(ref,x-1,y,state,val));
                }
                if (0==y) {
                    if (0<ref.block_index[1])
                        pts.push_back( Pt(ref.sibblingBottom(),x,h-1,state,val));
                } else {
                    pts.push_back( Pt(ref,x,y-1,state,val));
                }
            }
        }
    }
}


}} // Tools::Selections
