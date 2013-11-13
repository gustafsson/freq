#ifndef RESAMPLEHELPERS_H
#define RESAMPLEHELPERS_H

#include "resampletypes.h"
#include "datastorage.h"

template<typename T>
inline RESAMPLE_CALL T interpolate( T const& a, T const& b, float k )
{
    return (1.f-k)*a + k*b;
}

template<typename InputT, typename OutputT=InputT>
class NoConverter
{
public:
    typedef OutputT T;

    RESAMPLE_CALL OutputT operator()( InputT v, DataPos const& /*dataPosition*/ )
    {
        return v;
    }
};


class ConverterAmplitude
{
public:
    typedef float T;

    template<typename InputT>
    RESAMPLE_CALL float operator()( InputT v, DataPos const& dataPosition );
};


template<typename OutputT, typename Converter>
class InterpolateFetcher
{
public:
    typedef OutputT T;

    InterpolateFetcher() {}
    InterpolateFetcher(const Converter& converter):converter(converter) {}

    template<typename Reader>
    RESAMPLE_CALL OutputT operator()( ResamplePos const& p, Reader& reader )
    {
        DataPos q(floor(p.x), floor(p.y));
        ResamplePos k( p.x - floor(p.x), p.y - floor(p.y) );

        return interpolate(
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

    template<typename Reader>
    RESAMPLE_CALL OutputT get( DataPos const& q, Reader& reader )
    {
        return converter( reader( q ), q );
    }

private:
    Converter converter;
};

template<typename OutputT, typename Converter>
class NearestFetcher
{
public:
    NearestFetcher() {}
    NearestFetcher(Converter converter):converter(converter) {}

    template<typename Reader>
    RESAMPLE_CALL OutputT operator()( ResamplePos const& p, Reader& reader )
    {
        DataPos q(floor(p.x+.5f), floor(p.y+.5f));
        return converter( reader( q ), q );
    }

private:
    Converter converter;
};

class AffineTransform
{
public:
    AffineTransform(
            ResampleArea inputRegion,
            ResampleArea outputRegion,
            ValidSamples validInputs,
            DataPos outputSize
            )
                :
                scale(0,0),
                translation(0,0)
    {
//      x = writePos.x;
//      x = x / (outputSize.x-1);
//      x = x * width(outputRegion) + left(outputRegion);
//      % Here x is in 'global position space'
//      x = x - left(inputRegion);
//      x = x / width(inputRegion);
//      x = x * (width(validInputs)-1);
//      x = x + left(validInputs);
//      readPos.x = x;
//      Express this as one affine transform by:
//      readPos.x = x * scale.x + translation.x;
        // The if clauses takes care of the special case when one of the
        // dimensions is just one element wide
        translation.x = validInputs.left;
        translation.y = validInputs.top;
        scale.x = (validInputs.width()-1) / inputRegion.width();
        scale.y = (validInputs.height()-1) / inputRegion.height();
        translation.x += (outputRegion.left - inputRegion.left)*scale.x;
        translation.y += (outputRegion.top - inputRegion.top)*scale.y;
        if (outputSize.x==1) ++outputSize.x;
        if (outputSize.y==1) ++outputSize.y;
        scale.x *= outputRegion.width()/(outputSize.x-1);
        scale.y *= outputRegion.height()/(outputSize.y-1);
    }

    template<typename Vec2>
    RESAMPLE_ANYCALL Vec2 operator()( Vec2 const& p )
    {
        Vec2 q;
        q.x = translation.x + p.x*scale.x;
        q.y = translation.y + p.y*scale.y;
        return q;
    }

private:
    ResamplePos scale;
    ResamplePos translation;
};


class AffineTransformFlip
{
public:
    AffineTransformFlip(
            ResampleArea inputRegion,
            ResampleArea outputRegion,
            ValidSamples validInputs,
            DataPos outputSize
            )
                :
                scale(0,0),
                translation(0,0)
    {
        //      x = writePos.x;
        //      x = x / (outputSize.x-1);
        //      x = x * width(outputRegion) + left(outputRegion);
        //      y = x;
        //      y = y - top(inputRegion);
        //      y = y / height(inputRegion);
        //      y = y * (height(validInputs)-1);
        //      y = y + top(validInputs);
        //      readPos.y = y;
        //      Express this as one affine transform by:
        //      readPos.y = writePos.x * scale.y + translation.y;
        //      readPos.x = writePos.y * scale.x + translation.x;
        translation.x = validInputs.left;
        translation.y = validInputs.top;
        scale.x = (validInputs.width()-1) / inputRegion.height();
        scale.y = (validInputs.height()-1) / inputRegion.width();
        translation.x += (outputRegion.top - inputRegion.top)*scale.x;
        translation.y += (outputRegion.left - inputRegion.left)*scale.y;
        if (outputSize.x==1) ++outputSize.x;
        if (outputSize.y==1) ++outputSize.y;
        scale.x *= outputRegion.height()/(outputSize.y-1);
        scale.y *= outputRegion.width()/(outputSize.x-1);
    }


    RESAMPLE_ANYCALL ResamplePos operator()( ResamplePos const& p )
    {
        return ResamplePos(
                translation.x + p.y*scale.x,
                translation.y + p.x*scale.y );
    }

private:
    ResamplePos scale;
    ResamplePos translation;
};


template<typename OutputT>
class AssignOperator
{
public:
    RESAMPLE_CALL void operator()( OutputT &e, OutputT const& v )
    {
        e = v;
    }
};


template<typename OutputT, typename Assignment=AssignOperator<OutputT>, typename FetchT=OutputT >
class DefaultWriter
{
public:
    DefaultWriter(OutputT* output, int outputPitch, int rowCount, Assignment assignment=Assignment())
        :   output(output),
            outputPitch(outputPitch),
            rowCount(rowCount),
            assignment(assignment)
    {}


    RESAMPLE_CALL void operator()( FetchT const& value, DataPos writePos )
    {
        if (writePos.x >= outputPitch )
            writePos.x = outputPitch - 1;
        if (writePos.y >= rowCount )
            writePos.y = rowCount - 1;
        int o = outputPitch*writePos.y + writePos.x;
        assignment(output[o], value);
    }


    OutputT* output;
    int outputPitch; // pitch in number of elements
    int rowCount;
    Assignment assignment;
};


template<typename FetchT, typename OutputT, typename Assignment>
static DefaultWriter<OutputT, Assignment, FetchT> DefaultWriterStorage(
        boost::shared_ptr<DataStorage<OutputT> > outputp,
        DataPos outputSize,
        Assignment assignment);



#endif // RESAMPLEHELPERS_H
