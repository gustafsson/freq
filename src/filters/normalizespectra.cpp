#include "normalizespectra.h"
#include "tfr/stft.h"

#include "exceptionassert.h"
#include "cpumemorystorage.h"
#include "neat_math.h"
#include "string.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace Tfr;
using namespace std;

namespace Filters {

NormalizeSpectra::NormalizeSpectra(float means)
    :
      meansHz_(means)
{
}


bool NormalizeSpectra::
        operator()( Chunk& chunk )
{
    //removeSlidingMean( chunk );
    removeSlidingMedian( chunk );

    return false;
}


//    for (int w=0; w<stftChunk->nSamples (); ++w)
//    {
//        int W = stftChunk->nActualScales ();

//        // One octave around frequency bin 'i' goes from 2^(log2(i)-0.5) to 2^(log2(i)+0.5)
//        for (uint32_t i=0; i<W; ++i)
//        {
//            float v = p[ i + w*W ];
//            ... log2f and powf are slow
//        }
//    }


void NormalizeSpectra::
        removeSlidingMean( Chunk& chunk )
{
    StftChunk* stftChunk = dynamic_cast<StftChunk*>(&chunk);
    EXCEPTION_ASSERT( stftChunk );

    ChunkElement* p = CpuMemoryStorage::ReadOnly<1>( stftChunk->transform_data ).ptr();

    int nSamples = stftChunk->nSamples ();
    int W = stftChunk->nActualScales ();

#pragma omp parallel for
    for (int w=0; w<nSamples; ++w)
    {
        int R = ceil (meansHz_ * W / 2);
        float m = 0.f;
        ChunkElement * q = p + w*W;
        vector<float> original(W);
        for (int i=0; i<W; ++i)
            original[i] = norm(q[i]);

        int R2 = 0;
        for (int i=0; i<W && i<R; ++i)
        {
            m += original[i];
            R2++;
        }
        for (int i=0; i<W; ++i)
        {
            // q[i].real ( sqrtf(std::max( 0.f, norm(q[i]) - m/R2 ) ) );
            // q[i].real ( sqrtf( std::abs (norm(q[i]) - m/R2 ) ));
            q[i].real ( original[i] - m/R2 );
            q[i].imag ( 0.f );

            if (i>R)
            {
                m -= original[i-R];
                R2--;
            }
            if (i+R<W)
            {
                m += original[i+R];
                R2++;
            }
        }
    }
}


template<class I, class J>
void SlidingMedian( I inIter, I inEnd, J output, int window )
{
    typedef typename std::iterator_traits<I>::value_type T;

    std::vector<T> cq( window );
    std::vector<T> ordered( window );

    int index = 0;
    bool full = false;
    for( ; inIter != inEnd; ++inIter )
    {
        T in = *inIter;
        T old = cq[index];
        cq[index] = in;
        index = ( index+1 ) % window;
        if( index == 0 ) full = true;
        ordered.erase( std::lower_bound( ordered.begin(), ordered.end(), old ) );
        ordered.insert( std::upper_bound( ordered.begin(), ordered.end(), in ), in );
        if( full )
        {
            if( window % 2 ) *output++ = ordered[ window/2 ];
            else *output++ = ( ordered[ window/2-1 ] + ordered[ window/2 ] ) / T(2);
        }
    }
}


template<class I, class J>
void SlidingMedian( I inIter, I inEnd, J output, float fraction )
{
    typedef typename std::iterator_traits<I>::value_type T;

    int N = inEnd-inIter;
    std::vector<T> cq((N+2)*fraction + 1.5f);
    std::vector<T> ordered;

    int index = 0;
    int readindex = 0;
    for( int j = 0; j<N; ++j )
    {
        unsigned window = std::max(0.f, j*fraction + 0.5f);
        if (window < 2) window = 2;

        // when window increases we can't be sure that ordered is still full
        if (ordered.size () > 0 && (ordered.size () >= window || readindex-window <= j-window/2))
        {
            T old = cq[index];
            ordered.erase( std::lower_bound( ordered.begin(), ordered.end(), old ) );
        }

        while (ordered.size () < window)
        {
            T in = *inIter;
            cq[index] = in;
            index = ( index+1 ) % window;

            ordered.insert( std::upper_bound( ordered.begin(), ordered.end(), in ), in );

            if (inIter != inEnd && ordered.size () > window/2)
            {
                ++inIter;
                ++readindex;
            }
        }

        if( window % 2 ) *output++ = ordered[ window/2 ];
        else *output++ = ( ordered[ window/2-1 ] + ordered[ window/2 ] ) / T(2);
    }
}


void NormalizeSpectra::
        removeSlidingMedian( Chunk& chunk )
{
    StftChunk* stftChunk = dynamic_cast<StftChunk*>(&chunk);
    EXCEPTION_ASSERT( stftChunk );

    ChunkElement* p = CpuMemoryStorage::ReadWrite<1>( stftChunk->transform_data ).ptr();

    int nSamples = stftChunk->nSamples ();
    int W = stftChunk->nActualScales ();
    int R = computeR( chunk );

    if (meansHz_<0)
        R = 0;

#pragma omp parallel for
    for (int w=0; w<nSamples; ++w)
    {
        vector<float> original(W+2*R);
        ChunkElement * q = p + w*W;
        for (int i=0; i<R; ++i)
            original[i] = 0.f;
        for (int i=0; i<W; ++i)
            original[i+R] = abs(q[i]);
        for (int i=W+R; i<W+2*R; ++i)
            original[i] = 0.f;

        vector<float> median;
        //TaskInfo("meansHz_ = %f", meansHz_);
        if (meansHz_<0)
            SlidingMedian(
                        original.begin (),
                        original.end (),
                        std::insert_iterator< std::vector<float> >( median, median.begin() ),
                        -meansHz_);
        else
            SlidingMedian(
                        original.begin (),
                        original.end (),
                        std::insert_iterator< std::vector<float> >( median, median.begin() ),
                        2*R);
        //TaskInfo("meansHz_ = %f, median.size() = %d", meansHz_, median.size());

//        int s = median.size ();
//        float* mv = &median[0];
//        float* mo = &original[0];

        for (int i=0; i<W; ++i)
        {
            float v = original[i+R] / median[i];
            q[i].real ( v );
            //q[i].real ( sqrtf(fabsf(v)) * (v > 0 ? 1.f : -1.f) );
            q[i].imag ( 0.f );
        }
    }
}


int NormalizeSpectra::
        computeR( const Tfr::Chunk& chunk )
{
    const StftChunk* stftChunk = dynamic_cast<const StftChunk*>(&chunk);
    int processingWindow = stftChunk->window_size ();

    float fs = chunk.original_sample_rate;
    float hz_per_bin = fs/processingWindow;
    int R = ceil ( meansHz_/hz_per_bin / 2);
    return R;
}




} // namespace Filters
