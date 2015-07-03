#ifndef BLOCKKERNELDEF_H
#define BLOCKKERNELDEF_H

#include "resample.h"
#include "operate.h"

#ifdef FREQAXIS_CALL
#undef FREQAXIS_CALL
#endif
#define FREQAXIS_CALL RESAMPLE_ANYCALL
#include "tfr/freqaxis.h"
#include "heightmap/freqaxis.h"

// order of included headers matters because of FREQAXIS_CALL and RESAMPLE_CALL
#include "blockkernel.h"

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define M_PIf ((float)M_PI)

#ifdef __CUDACC__
#ifndef USE_CUDA
#define USE_CUDA
#endif
#endif

#ifdef USE_CUDA
typedef float2 BlockElemType;
#else
typedef Tfr::ChunkElement BlockElemType;
#endif

class ConverterPhase
{
public:
    RESAMPLE_CALL float operator()( BlockElemType v, DataPos const& /*dataPosition*/ ) const
    {
#ifdef USE_CUDA
        return atan2(v.x, v.y);
#else
        return atan2(v.imag(), v.real());
#endif
    }
};



template<Heightmap::AmplitudeAxis Axis>
class ConverterAmplitudeAxis
{
public:
    ConverterAmplitudeAxis(float normalization_factor)
        :
          normalization_factor(normalization_factor)
    {}

    RESAMPLE_CALL float operator()( BlockElemType v, DataPos const& dataPosition ) const
    {
        return Heightmap::AmplitudeValue<Axis>()( normalization_factor*ConverterAmplitude()( v, dataPosition ) );
    }

private:
    float normalization_factor;
};



template<>
class ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Real>
{
public:
    ConverterAmplitudeAxis(float normalization_factor)
        :
          normalization_factor(normalization_factor)
    {}

    RESAMPLE_CALL float operator()( BlockElemType v, DataPos const& /*dataPosition*/ ) const
    {
#ifdef USE_CUDA
        float q = v.x;
#else
        float q = v.real ();
#endif
        return normalization_factor * q;
    }

private:
    float normalization_factor;
};



template<typename DefaultConverter>
class AxisFetcher
{
public:
    typedef float T;

    AxisFetcher(const DefaultConverter& default_converter)
        :   defaultConverter(default_converter)
    {

    }


    template<typename OtherConverter>
    AxisFetcher& operator=(const AxisFetcher<OtherConverter>& b)
    {
        inputAxis = b.inputAxis;
        outputAxis = b.outputAxis;
        scale = b.scale;
        offs = b.offs;
        xstep = b.xstep;
        xmax = b.xmax;
        return *this;
    }


    template<typename Reader>
    RESAMPLE_CALL float operator()( ResamplePos const& p, Reader& reader )
    {
        return this->operator ()(p, reader, defaultConverter);
    }


    RESAMPLE_CALL bool isnear(float a, float b)
    {
        return a > b*(1.f-1e-2f) && a < b*(1.f+1e-2f);
    }


    template<typename Reader, typename Converter>
    RESAMPLE_CALL float operator()( ResamplePos const& p, Reader& reader, const Converter& c = Converter() )
    {
        ResamplePos q2;
        // exp2f (called in 'getFrequency') is only 4 multiplies for arch 1.x
        // so these are fairly cheap operations. One reciprocal called in
        // 'getFrequencyScalar' is just as fast.
        q2.x = p.x;
        q2.y = p.y*scale+offs;
        float hz = outputAxis.getFrequency( q2.y );
        //float hz2 = outputAxis.getFrequency( q2.y + ystep );
        q2.y = inputAxis.getFrequencyScalar( hz );
        //float next_q2y = inputAxis.getFrequencyScalar( hz2 );

        DataPos q(floor(q2.x), floor(q2.y));
        ResamplePos k( q2.x - floor(q2.x), q2.y - floor(q2.y) );

        float v = 0.f;

        //float ystep = next_q2y - q2.y;

//        if (ystep <= 1.f)
//        {
            if (xstep <= 1.f)
            {
                // bilinear interpolation
                v = interpolate(
                        interpolate(
                                get( q, reader, c ),
                                get( DataPos(q.x+1.f, q.y), reader, c ),
                                k.x),
                        interpolate(
                                get( DataPos(q.x, q.y+1.f), reader, c ),
                                get( DataPos(q.x+1.f, q.y+1.f), reader, c ),
                                k.x),
                        k.y );
            }
            else
            {
                float v2 = 0;
                float xstop = min(floor(q2.x+xstep), xmax);

                for (float x=q2.x; x<xstop; ++x)
                    v = max(v, get( DataPos(x, q.y), reader, c ));

                for (float x=q2.x; x<xstop; ++x)
                    v2 = max(v2, get( DataPos(x, q.y+1.f), reader, c ));

                v = interpolate( v, v2, k.y);
            }
//        }
//        else
//        {
//            if (xstep <= 1.f)
//            {
//                for (float y=q2.y; y<floor(q2.y+ystep); ++y)
//                {
//                    v = max(v, interpolate(
//                            get( DataPos(q.x, y), reader, c ),
//                            get( DataPos(q.x+1.f, y), reader, c ),
//                            k.x));
//                }
//            }
//            else
//            {
//                for (float x=q2.x; floor(x<q2.x+xstep); ++x)
//                {
//                    for (float y=q2.y; floor(y<q2.y+ystep); ++y)
//                    {
//                        v = max(v, get( DataPos(x, y), reader, c ));
//                    }
//                }
//            }
//        }

        return v;
    }

    template<typename Reader, typename Converter>
    RESAMPLE_CALL float get( DataPos const& q, Reader& reader, Converter& c )
    {
        return c( reader( q ), q );
    }

    Tfr::FreqAxis inputAxis;
    Heightmap::FreqAxis outputAxis;
    DefaultConverter defaultConverter;

    float scale;
    float offs;

    float xstep;
    float xmax;
//    float ystep;
};


template<typename DefaultConverter>
class AxisFetcherTranspose
{
public:
    typedef float T;

    AxisFetcherTranspose(const DefaultConverter& default_converter)
        :   defaultConverter(default_converter)
    {

    }

//    template<typename Reader>
//    RESAMPLE_CALL float operator()( ResamplePos const& p, Reader& reader )
//    {
//        ResamplePos q;
//        // exp2f (called in 'getFrequency') takes time equal to 4 multiplies for cuda arch 1.x
//        // so these are fairly cheap operations. One reciprocal called in
//        // 'getFrequencyScalar' is just as fast.
//        // Tests have shown that this doen't affect the total execution time,
//        // when outputAxis and inputAxis are affine transformations of eachother.
//        // Hence the kernel is memory bound.
//        float hz = outputAxis.getFrequency( p.x );
//        q.x = inputAxis.getFrequencyScalar( hz );
//        q.y = p.y;

//        float r = InterpolateFetcher<float, DefaultConverter>(defaultConverter)( q, reader );
//        return r;
//    }

    template<typename Reader>
    RESAMPLE_CALL float operator()( ResamplePos const& p, Reader& reader )
    {
        ResamplePos q2;
        // exp2f (called in 'getFrequency') takes time equal to 4 multiplies for cuda arch 1.x
        // so these are fairly cheap operations. One reciprocal called in
        // 'getFrequencyScalar' is just as fast.
        // Tests have shown that this doen't affect the total execution time,
        // when outputAxis and inputAxis are affine transformations of eachother.
        // Hence the kernel is memory bound.
        float hz = outputAxis.getFrequency( p.x );
        float hz1 = outputAxis.getFrequency( p.x - 0.5f*xstep );
        float hz2 = outputAxis.getFrequency( p.x + 0.5f*xstep );
        q2.x = inputAxis.getFrequencyScalar( hz );
        float qx1 = inputAxis.getFrequencyScalar( hz1 );
        float qx2 = inputAxis.getFrequencyScalar( hz2 );
        q2.y = p.y;

        //float r = InterpolateFetcher<float, DefaultConverter>(defaultConverter)( q, reader );
        //return r;

        DataPos q(floor(q2.x), floor(q2.y));
        ResamplePos k( q2.x - floor(q2.x), q2.y - floor(q2.y) );

        float v = 0.f;
        float xdiff = fabsf(qx2 - qx1);

        if (xdiff <= 1.f)
        {
            if (ystep <= 1)
            {
                // bilinear interpolation
                v = interpolate(
                        interpolate(
                                get( q, reader ),
                                get( DataPos(q.x+1, q.y), reader ),
                                k.x),
                        interpolate(
                                get( DataPos(q.x, q.y+1), reader ),
                                get( DataPos(q.x+1, q.y+1), reader ),
                                k.x),
                        k.y );
            }
            else
            {
                int stopy = min(q.y+ystep, ymax);

                for (DataPos getp(q.x, q.y); getp.y<stopy; ++getp.y)
                    v = max(v, get( getp, reader));

                float v2 = 0.f;
                for (DataPos getp(q.x + 1, q.y); getp.y<stopy; ++getp.y)
                    v2 = max(v2, get( getp, reader));

                v = interpolate( v, v2, k.x);
            }
        }
        else
        {
            qx1 = floor(min(qx2, qx1));
            int stopx = qx1+xdiff;

            if (ystep <= 1)
            {
                float v2=0;

                for (DataPos getp(qx1, q.y); getp.x<stopx; ++getp.x)
                    v = max(v, get( getp, reader));

                for (DataPos getp(qx1, q.y+1); getp.x<stopx; ++getp.x)
                    v2 = max(v2, get( getp, reader));

                v = interpolate( v, v2, k.y);
            }
            else
            {
                DataPos getp(0, 0);
                int stopy = min(q.y+ystep, ymax);

                for (getp.y = q.y; getp.y<stopy; ++getp.y)
                {
                    for (getp.x = qx1; getp.x<stopx; ++getp.x)
                    {
                        v = max(v, get( getp, reader));
                    }
                }
            }
        }
        return v;
    }

    template<typename Reader>
    RESAMPLE_CALL float get( DataPos const& q, Reader& reader )
    {
        return defaultConverter( reader( q ), q );
    }

    DefaultConverter defaultConverter;
    Tfr::FreqAxis inputAxis;
    Heightmap::FreqAxis outputAxis;

    float xstep;
    int ystep;
    int ymax;
};


template<typename DefaultConverter>
class WeightFetcher
{
public:
    typedef float T;

    WeightFetcher(const DefaultConverter& b)
        : axes(b)
    {

    }

    template<typename Reader>
    RESAMPLE_CALL float operator()( ResamplePos const& p, Reader& reader )
    {
        float v = axes( p, reader );
        float phase1 = axes( p, reader, ConverterPhase() );
        float phase2 = axes( ResamplePos(p.x, p.y+1), reader, ConverterPhase() );
        float phasediff = phase2 - phase1;
        if (phasediff < -M_PIf ) phasediff += 2*M_PIf;
        if (phasediff > M_PIf ) phasediff -= 2*M_PIf;

        float k = exp2f(-fabsf(phasediff));

        return v * k;
    }

    AxisFetcher<DefaultConverter> axes;
};


class SpecialPhaseFetcher
{
public:
    template<typename Reader>
    RESAMPLE_CALL float operator()( ResamplePos const& p, Reader& reader )
    {
        // Plot "how wrong" the phase is
        ResamplePos q1 = p;
        ResamplePos p2 = p;
        p2.x++;
        ResamplePos q2 = p2;
        q1.x /= validInputs4.width()-1;
        q1.y /= validInputs4.height()-1;
        q1.x *= inputRegion.width();
        q1.y *= inputRegion.height();
        q1.x += inputRegion.left;
        q1.y += inputRegion.top;
        q2.x /= validInputs4.width()-1;
        q2.y /= validInputs4.height()-1;
        q2.x *= inputRegion.width();
        q2.y *= inputRegion.height();
        q2.x += inputRegion.left;
        q2.y += inputRegion.top;

        float phase = InterpolateFetcher<float, ConverterPhase>()( p, reader );
        float phase2 = InterpolateFetcher<float, ConverterPhase>()( p2, reader );
        float v = InterpolateFetcher<float, ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic> >()( p, reader );
        float phasediff = phase2 - phase;
        if (phasediff < -M_PIf ) phasediff += 2*M_PIf;
        if (phasediff > M_PIf ) phasediff -= 2*M_PIf;

        float f = freqAxis.getFrequency( q1.y );
        f *= (q2.x - q1.x)*2*M_PIf;
        return f*v;

        /*float expected_phase = f * q.x * 2*M_PIf;
        float phasediff = fmodf( expected_phase+2*M_PIf-phase, 2*M_PIf );

        return expected_phase;*/
        //return freqAxis.getFrequency( q.y )/22050.0f;
        //return /*v **/ phasediff / (2*M_PIf);
    }

    Tfr::FreqAxis freqAxis;
    ResampleArea inputRegion;
    ValidSamples validInputs4;
};


//    cuda-memcheck complains even on this testkernel when using global memory
//    from OpenGL but not on cudaMalloc'd memory. See MappedVbo test.
//__global__ void testkernel(
//        float* output, elemSize3_t sz)
//{
//    elemSize3_t  writePos;
//    writePos.x = blockIdx.x * 128 + threadIdx.x;
//    writePos.y = blockIdx.y * 1 + threadIdx.y;
//    if (writePos.x<sz.x && writePos.y < sz.y)
//    {
//        int o = writePos.x  +  writePos.y * sz.x;
//        o = o % 32;
//        output[o] = 0;
//    }
//}

#include <iostream>
using namespace std;


inline
template<typename AxisConverter>
void blockResampleChunkAxis( Tfr::ChunkData::ptr inputp,
                 DataStorage<float>::ptr output,
                 ValidInterval validInputs,
                 ResampleArea inputRegion,
                 ResampleArea outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Heightmap::FreqAxis outputAxis,
                 AxisConverter amplitudeAxis,
                 bool enable_subtexel_aggregation
                 )
{
    // translate type to be read as a cuda texture
#ifdef USE_CUDA
    DataStorage<float2>::Ptr input =
            CudaGlobalStorage::BorrowPitchedPtr<float2>(
                    inputp->size(),
                    CudaGlobalStorage::ReadOnly<2>( inputp ).getCudaPitchedPtr()
                    );
#else
    Tfr::ChunkData::ptr input = inputp;
#endif

    DataStorageSize sz_output = output->size();

//    cuda-memcheck complains even on this testkernel when using global memory
//    from OpenGL but not on cudaMalloc'd memory. See MappedVbo test.
//    dim3 block( 128 );
//    dim3 grid( int_div_ceil( sz_output.x, block.x ), sz_output.y );
//    testkernel<<< grid, block>>>(output.ptr(), sz_output);
//    float*devptr;
//    cudaMalloc(&devptr, sizeof(float)*32);
//    testkernel<<< grid, block>>>(devptr, sz_output);
//    cudaFree(devptr);
//    return;

    // We wan't to do our own translation from coordinate position to matrix
    // position in the input matrix. By giving bottom=0 and top=2 we tell
    // 'resample2d_fetcher' to only translate to a normalized reading position
    // [0, height-1) along the input y-axis.
//    uint4 validInputs4 = make_uint4( validInputs.x, 0, validInputs.y, sz_input.y );
    ValidSamples validInputs4( validInputs.first, 0, validInputs.last, 2 );
    ValidSamples validOutputs( 0, 0, sz_output.width, sz_output.height );

    AxisFetcher<AxisConverter> axes = amplitudeAxis;
    axes.inputAxis = inputAxis;
    axes.outputAxis = outputAxis;
    axes.offs = inputRegion.top;
    axes.scale = inputRegion.height();
    axes.xmax = validInputs.last;

    axes.xstep = (validInputs.last - validInputs.first - 1) / (float)sz_output.width * outputRegion.width()/inputRegion.width();
    // axes.ystep = 1; // because of varying frequency density ystep should be computed in the kernel together with the FreqAxes
    // axes.ystep = input->size().height / (float)sz_output.height * outputRegion.height()/(float)inputRegion.height();

    if (!enable_subtexel_aggregation)
    {
        axes.xstep = 0;
        // axes.ystep = 1;
    }

    switch (transformMethod)
    {
    case Heightmap::ComplexInfo_Amplitude_Weighted:
    {
        WeightFetcher<AxisConverter> fetcher = WeightFetcher<AxisConverter>(amplitudeAxis);
        fetcher.axes = axes;

        resample2d_fetcher<WeightFetcher<AxisConverter>, AssignOperator<float> >(
                input,
                output,
                validInputs4,
                validOutputs,
                inputRegion,
                outputRegion,
                false,
                fetcher
        );
        break;
    }
    case Heightmap::ComplexInfo_Amplitude_Non_Weighted:
        resample2d_fetcher<AxisFetcher<AxisConverter>, AssignOperator<float> >(
                input,
                output,
                validInputs4,
                validOutputs,
                inputRegion,
                outputRegion,
                false,
                axes
        );
        break;
    case Heightmap::ComplexInfo_Phase:
    {
        AxisFetcher<ConverterPhase> axesPhase = ConverterPhase();
        axesPhase = axes;

        resample2d_fetcher<AxisFetcher<ConverterPhase>, AssignOperator<float> >(
                    input,
                    output,
                    validInputs4,
                    validOutputs,
                    inputRegion,
                    outputRegion,
                    false,
                    axesPhase
            );
    }
    /*case Heightmap::ComplexInfo_Phase:
            SpecialPhaseFetcher specialPhase;
            specialPhase.freqAxis = freqAxis;
            specialPhase.inputRegion = inputRegion;
            specialPhase.validInputs4 = validInputs4;
            resample2d_fetcher<float, float2, float, SpecialPhaseFetcher, AssignOperator<float> >(
                    input,
                    output,
                    validInputs4,
                    validOutputs,
                    inputRegion,
                    outputRegion,
                    false,
                    specialPhase
            );*/
    }
}


inline
void blockResampleChunk( Tfr::ChunkData::ptr input,
                  BlockData::ptr output,
                 ValidInterval validInputs, // validInputs is the first and last-1 valid samples in x
                 ResampleArea inputRegion,
                 ResampleArea outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Heightmap::FreqAxis outputAxis,
                 Heightmap::AmplitudeAxis amplitudeAxis,
                 float normalization_factor,
                 bool enable_subtexel_aggregation
                 )
{
    switch(amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:
        blockResampleChunkAxis(
                input, output, validInputs, inputRegion,
                outputRegion, transformMethod, inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Linear>(normalization_factor),
                enable_subtexel_aggregation);
        break;
    case Heightmap::AmplitudeAxis_Logarithmic:
        blockResampleChunkAxis(
                input, output, validInputs, inputRegion,
                outputRegion, transformMethod, inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>(normalization_factor),
                enable_subtexel_aggregation);
        break;
    case Heightmap::AmplitudeAxis_5thRoot:
        blockResampleChunkAxis(
                input, output, validInputs, inputRegion,
                outputRegion, transformMethod, inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_5thRoot>(normalization_factor),
                enable_subtexel_aggregation);
        break;
    case Heightmap::AmplitudeAxis_Real:
        break;
    default:
        break;
    }
}


inline
template<typename AxisConverter>
void resampleStftAxis( Tfr::ChunkData::ptr inputp,
                   size_t nScales, size_t nSamples,
                   BlockData::ptr output,
                   ValidInterval outputInterval,
                   ResampleArea inputRegion,
                   ResampleArea outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Heightmap::FreqAxis outputAxis,
                   AxisConverter axisConverter,
                   bool enable_subtexel_aggregation
                   )
{
#ifdef USE_CUDA
    cudaPitchedPtr cpp = CudaGlobalStorage::ReadOnly<2>( inputp ).getCudaPitchedPtr();
    cpp.xsize = cpp.pitch = nScales * sizeof(float2);
    cpp.ysize = nSamples;
    DataStorage<float2>::Ptr input =
            CudaGlobalStorage::BorrowPitchedPtr<float2>( DataStorageSize( nScales, nSamples ), cpp );
#else
    Tfr::ChunkElement* p = CpuMemoryStorage::ReadOnly<2>( inputp ).ptr();
    Tfr::ChunkData::ptr input =
            CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>(
                    DataStorageSize( nScales, nSamples ), p );
#endif

    DataStorageSize sz_input = input->size(); // != inputp->size()
    DataStorageSize sz_output = output->size();


    // We wan't to do our own translation from coordinate position to matrix
    // position in the input matrix. By giving left=0 and right=2 we tell
    // 'resample2d_fetcher' to only translate to a normalized reading position
    // [0, width-1) along the input x-axis. 'resample2d_fetcher' does the
    // transpose for us.
    ValidSamples validInputs( 0, 0, 2, sz_input.height );
    //ValidOutputs validOutputs( sz_output.width, sz_output.height ); // TODO need validOutputInterval to use fetcher.ystep
    ValidSamples validOutputs( outputInterval.first, 0, outputInterval.last, sz_output.height );

    AxisFetcherTranspose<AxisConverter> fetcher = axisConverter;
    fetcher.inputAxis = inputAxis;
    fetcher.outputAxis = outputAxis;

    // xstep and ystep are used after transposing to input coordinates
    fetcher.ystep = (sz_input.height-1) / (float)(sz_output.width-1) * outputRegion.width()/inputRegion.width();
    fetcher.xstep = 1.f/(sz_output.height-1) * outputRegion.height()/inputRegion.height();
    fetcher.ymax = sz_input.height;

    if (!enable_subtexel_aggregation)
    {
        fetcher.ystep = 0;
        fetcher.xstep = 0;
    }

    resample2d_fetcher(
                input,
                output,
                validInputs,
                validOutputs,
                inputRegion,
                outputRegion,
                true,
                fetcher,
                AssignOperator<float>()
        );
}


inline
void resampleStft( Tfr::ChunkData::ptr input,
                   size_t nScales, size_t nSamples,
                   BlockData::ptr output,
                   ValidInterval outputInterval,
                   ResampleArea inputRegion,
                   ResampleArea outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Heightmap::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis,
                   float normalization_factor,
                   bool enable_subtexel_aggregation)
{
    // fetcher.factor makes it roughly equal height to Cwt
    switch(amplitudeAxis)
    {
    case Heightmap::AmplitudeAxis_Linear:
        resampleStftAxis(
                input, nScales, nSamples, output, outputInterval,
                inputRegion, outputRegion,
                inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Linear>(normalization_factor),
                enable_subtexel_aggregation);
        break;
    case Heightmap::AmplitudeAxis_Logarithmic:
        resampleStftAxis(
                input, nScales, nSamples, output, outputInterval,
                inputRegion, outputRegion,
                inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_Logarithmic>(normalization_factor),
                enable_subtexel_aggregation);
        break;
    case Heightmap::AmplitudeAxis_5thRoot:
        resampleStftAxis(
                input, nScales, nSamples, output, outputInterval,
                inputRegion, outputRegion,
                inputAxis, outputAxis,
                ConverterAmplitudeAxis<Heightmap::AmplitudeAxis_5thRoot>(normalization_factor),
                enable_subtexel_aggregation);
        break;
    case Heightmap::AmplitudeAxis_Real:
        break;
    default:
        break;
    }
}

#endif // BLOCKKERNELDEF_H
