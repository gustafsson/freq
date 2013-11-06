/// see resample2d_elem

#ifndef RESAMPLE_CU_H
#define RESAMPLE_CU_H

#include "resamplehelpers.h"
#include "unused.h"

template<typename T>
inline RESAMPLE_CALL T zero()    { return 0; }

template<>
inline RESAMPLE_CALL ResamplePos zero() { return ResamplePos(0,0); }


template<typename OutputT, typename Fetcher, typename Reader>
inline RESAMPLE_CALL OutputT fetch( int x, int y, Fetcher& fetcher, Reader& reader )
{
    return fetcher( DataPos(x, y), reader );
}


template<typename OutputT, typename Fetcher, typename Reader>
inline RESAMPLE_CALL OutputT fetch( int x, float y, Fetcher& fetcher, Reader& reader )
{
    float yb = floor(y);
    float yk = y-yb;
    int yu = (int)yb;
    OutputT a = (yk==1.f) ? zero<OutputT>() : fetch<OutputT>( x, yu, fetcher, reader );
    OutputT b = (yk==0.f) ? zero<OutputT>() : fetch<OutputT>( x, yu+1, fetcher, reader );

    return interpolate( a, b, yk );
}


template<typename OutputT, typename Fetcher, typename Reader, typename YType>
inline RESAMPLE_CALL OutputT fetch( float x, YType y, Fetcher& fetcher, Reader& reader )
{
    float xb = floor(x);
    float xk = x-xb;
    int xu = (int)xb;
    OutputT a = (xk==1.f) ? zero<OutputT>() : fetch<OutputT>( xu, y, fetcher, reader );
    OutputT b = (xk==0.f) ? zero<OutputT>() : fetch<OutputT>( xu+1, y, fetcher, reader );

    return interpolate( a, b, xk );
}

// #define MULTISAMPLE

template<typename OutputT, typename Fetcher, typename Reader, typename YType>
#ifdef MULTISAMPLE
inline RESAMPLE_CALL OutputT getrow( float x, float x1, float x2, YType y, Fetcher& fetcher, Reader& reader )
#else
inline RESAMPLE_CALL OutputT getrow( float x, UNUSED(float x1), UNUSED(float x2), YType y, Fetcher& fetcher, Reader& reader )
#endif
{
    OutputT c;

#ifdef MULTISAMPLE
    if (floor(x2)-ceil(x1) <= 3)
#endif
        // Very few samples in this interval, interpolate and take middle
        c = fetch<OutputT>( x, y, fetcher, reader );
#ifdef MULTISAMPLE
    else
    {
        // Not very few samples in this interval, fetch max value

        // if (floor(x1) < x1)
        c = fetch<OutputT>( x1, y, fetcher, reader );

        for (int x=ceil(x1); x<=floor(x2); ++x)
            maxassign( c, fetch<OutputT>( x, y, fetcher, reader ));

        if (floor(x2) < x2)
            maxassign( c, fetch<OutputT>( x2, y, fetcher, reader ));
    }
#endif
    return c;
}


template<typename A, typename B>
inline RESAMPLE_CALL bool isless_test( A const& a, B const& b)
{
    return a < b;
}

template<>
inline RESAMPLE_CALL bool isless_test( ResamplePos const& a, ResamplePos const& b)
{
    return a.x*a.x + a.y * a.y < b.x*b.x + b.y*b.y;
}

template<typename T>
inline RESAMPLE_CALL void maxassign(T& a, T const& b )
{
    if ( isless_test(a, b) )
        a = b;
}


/**
  resample2d_elem resamples an image to another size, and optionally applies
  a conversion (template argument 'Converter') to each element. Upsampling is
  performed by linear interpolation. Downsampling uses max values, with
  linearly interpolated edges.

  @param input
    Input image

  @param output
    Output image

  @param inputRegion
    translation( inputRegion ) = 'affine transformation of entire input image
    region'. inputRegion must be given in the same unit as outputRegion. The
    region is inclusive which means that if the input contains 5 samples and
    covers the region [0,1] samples are defined at points 0, 0.25, 0.5, 0.75
    and 1. So a signal with discrete sample rate 'fs' over the region [0,4]
    seconds should contain exactly '4*fs+1' number of samples.

  @param outputRegion
    outputRegion is an affine transformation of the entire ouput image region.
    Given in the same unit as inputRegion. The intersection of inputRegion and
    outputRegion will be translated to an area of the input image that is
    resampled, converted, and written to output.

    If the intersection is not covered by outputRegion, then some samples in
    output will not be written to. Samples that are crossed by the intersection
    border will also not be written to. It is up to the caller to handle this
    situation.


  @param validInputs
    Valid inputs may be smaller than the given input image. validInputs are
    given in number of samples. inputRegion still refers to the total input
    image size though, validInputs effectively makes the intersection smaller.

  @param validOutputs
    Valid outputs may be smaller than the given output image. validOutputs are
    given in number of samples. outputRegion still refers to the total output
    image size though, validOutputs effectively makes the intersection smaller.

  @param converter
    Converts OutputT to InputT. This conversion may optionally use the position
    where data is read from. See NoConverter for an example.

  @param translation
    Translates intersection position to input read coordinates.
  */
template<
        typename Fetcher,
        typename Transform,
        typename Reader,
        typename Writer>
inline RESAMPLE_CALL void resample2d_elem (
        DataPos writePos,
        ResampleArea validInputs,
        Fetcher fetcher,
        ValidSamples outputSize,
        Transform coordinateTransform,
        Reader reader,
        Writer writer
        )
{
    if (writePos.x>=outputSize.right || writePos.x<outputSize.left)
        return;
    if (writePos.y>=outputSize.bottom || writePos.y<outputSize.top)
        return;

#ifndef MULTISAMPLE
    ResamplePos p(writePos.x, writePos.y);
    p = coordinateTransform(p);
    if (p.x < validInputs.left) return;
    if (p.y < validInputs.top) return;
    if (p.x > validInputs.right-0.9999f) return;
    if (p.y > validInputs.bottom-0.9999f) return;

    typename Fetcher::T c = fetcher( p, reader );
#else
    ResamplePos p1(writePos.x, writePos.y);
    ResamplePos p2 = p1;
    p2.x += 0.5f;
    p2.y += 0.5f;
    p1.x -= 0.5f;
    p1.y -= 0.5f;

    p1 = coordinateTransform(p1);
    p2 = coordinateTransform(p2);

    ResamplePos p(
            (p1.x+p2.x)*.5f,
            (p1.y+p2.y)*.5f );

    if (p1.x < validInputs.left) p1.x = validInputs.left;
    if (p1.y < validInputs.top) p1.y = validInputs.top;
    if (p2.x <= validInputs.left) return;
    if (p2.y <= validInputs.top) return;
    if (p1.x >= validInputs.right-1) return;
    if (p1.y >= validInputs.bottom-1) return;
    if (p2.x > validInputs.right-1) p2.x = validInputs.right-1;
    if (p2.y > validInputs.bottom-1) p2.y = validInputs.bottom-1;

    // TODO make fetcher work for MULTISAMPLE
    typename Fetcher::T c = fetcher( p1, p2, reader );

    if (floor(p2.y)-ceil(p1.y) <= 3.f)
        // Very few samples in this interval, interpolate and take middle position
        c = getrow<FetchT>( p.x, p1.x, p2.x, p.y, fetcher, reader);
    else
    {
        // Not very few samples in this interval, fetch max value
        //if (floor(p1.y) < p1.y)
        c = getrow<FetchT>( p.x, p1.x, p2.x, p1.y, fetcher, reader);

        for (int y=ceil(p1.y); y<=floor(p2.y); ++y)
            maxassign( c, getrow<FetchT>( p.x, p1.x, p2.x, y, fetcher, reader) );

        if (floor(p2.y) < p2.y)
            maxassign( c, getrow<FetchT>( p.x, p1.x, p2.x, p2.y, fetcher, reader) );
    }
#endif

    writer( c, writePos );
}



template<
        typename Reader,
        typename Fetcher,
        typename Writer>
static void resample2d_transform(
        Reader reader,
        Writer writer,
        ValidSamples validInputs,
        ValidSamples validOutputs,
        DataPos outputSize,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose = false,
        Fetcher fetcher = Fetcher()
        )
{
    // If regions are disjoint, don't do anything
    if (inputRegion.right < outputRegion.left ||
        inputRegion.left > outputRegion.right ||
        inputRegion.bottom < outputRegion.top ||
        inputRegion.top > outputRegion.bottom)
    {
        return;
    }

    if (!transpose)
    {
        AffineTransform transform(
                inputRegion,
                outputRegion,
                validInputs,
                outputSize
                );

        resample2d_storage(
                validInputs,
                fetcher,
                validOutputs,

                transform,
                reader,
                writer
        );
    }
    else
    {
        AffineTransformFlip transform(
                inputRegion,
                outputRegion,
                validInputs,
                outputSize
                );

        resample2d_storage(
                validInputs,
                fetcher,
                validOutputs,

                transform,
                reader,
                writer
        );
    }
}


template<
        typename Fetcher,
        typename Assignment,
        typename InputT,
        typename OutputT
        >
static void resample2d_fetcher(
        boost::shared_ptr<DataStorage<InputT> > input,
        boost::shared_ptr<DataStorage<OutputT> > output,
        ValidSamples validInputs,
        ValidSamples validOutputs,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose = false,
        Fetcher fetcher = Fetcher(),
        Assignment assignment = Assignment()
        )
{
    // make sure validInputs is smaller than input size
    validInputs.right = min(validInputs.right, input->size().width);
    validInputs.bottom = min(validInputs.bottom, input->size().height);
    validInputs.left = min(validInputs.right, validInputs.left);
    validInputs.top = min(validInputs.bottom, validInputs.top);

    // make sure validOutputs is smaller than output size
    validOutputs.right = min(validOutputs.right, output->size().width);
    validOutputs.bottom = min(validOutputs.bottom, output->size().height);
    validOutputs.left = min(validOutputs.right, validOutputs.left);
    validOutputs.top = min(validOutputs.bottom, validOutputs.top);

    resample2d_reader(
            input,
            DefaultWriterStorage<typename Fetcher::T>(
                    output, DataPos(validOutputs.right, validOutputs.bottom), assignment),
            validInputs, validOutputs, DataPos(output->size().width, output->size().height),
            inputRegion, outputRegion, transpose,
            fetcher );
}


template<
        typename Converter,
        typename Assignment,
        typename InputT,
        typename OutputT>
static void resample2d(
        boost::shared_ptr<DataStorage<InputT> > input,
        boost::shared_ptr<DataStorage<OutputT> > output,
        ValidSamples validInputs,
        ValidSamples validOutputs,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose = false,
        Converter converter = Converter(),
        Assignment assignment = Assignment()
        )
{
    resample2d_fetcher( input, output, validInputs, validOutputs,
                inputRegion, outputRegion, transpose,
                InterpolateFetcher<typename Converter::T, Converter>( converter ),
                assignment );
}


template<
        typename Converter,
        typename Assignment,
        typename InputT,
        typename OutputT
        >
static void resample2d_plain(
        boost::shared_ptr<DataStorage<InputT> > input,
        boost::shared_ptr<DataStorage<OutputT> > output,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose=false,
        Converter converter = Converter(),
        Assignment assignment = Assignment()
        )
{
    EXCEPTION_ASSERT( input.get() );
    EXCEPTION_ASSERT( output.get() );

    DataStorageSize sz_input = input->size();
    DataStorageSize sz_output = output->size();

    ValidSamples validInputs( 0, 0, sz_input.width, sz_input.height );
    ValidSamples validOutputs( 0, 0, sz_output.width, sz_output.height );

    resample2d<Converter, Assignment>(
            input,
            output,
            validInputs,
            validOutputs,
            inputRegion,
            outputRegion,
            transpose,
            converter,
            assignment);
}


template<
        typename Converter,
        typename InputT,
        typename OutputT>
static void resample2d_plain(
        boost::shared_ptr<DataStorage<InputT> > input,
        boost::shared_ptr<DataStorage<OutputT> > output,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose=false,
        Converter converter = Converter()
        )
{
    resample2d_plain(input,
                     output,
                     inputRegion,
                     outputRegion,
                     transpose,
                     converter,
                     AssignOperator<OutputT>() );
}


/*template<
        typename InputT,
        typename OutputT>
static void resample2d_overlap(
        InputT input,
        OutputT output,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1)
        )
{
    resample2d<InputT, OutputT, NoConverter<InputT, OutputT>, AssignOperator >(
            input,
            output,
            inputRegion,
            outputRegion );
}
*/

#endif // RESAMPLE_CU_H
