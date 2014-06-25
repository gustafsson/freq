#ifndef HEIGHTMAPBLOCK_CU_H
#define HEIGHTMAPBLOCK_CU_H

#include "tfr/freqaxis.h"
#include "tfr/chunkdata.h"
#include "resampletypes.h"
#include "heightmap/amplitudeaxis.h"

typedef DataStorage<float> BlockData;

struct ValidInterval
{
    ValidInterval(unsigned first, unsigned last)
        :
        first(first),
        last(last)
    {}

    unsigned first, last;
};

/**
  The namespace Tfr does not know about the namespace Heightmap
  */
namespace Heightmap {

    template<AmplitudeAxis>
    class AmplitudeValue;


    template<>
    class AmplitudeValue<AmplitudeAxis_Linear>
    {
    public:
        RESAMPLE_CALL float operator()( float v )
        {
            return 25.f * v;
        }
    };

    template<>
    class AmplitudeValue<AmplitudeAxis_Logarithmic>
    {
    public:
        RESAMPLE_CALL float operator()( float v )
        {
            // Sonic AWE until 2013-07-01
            // return 0.02f * (log2f(0.001f + v) - log2f(0.001f));

            // Define range by
            // "1/3 = f * (log(1) - log(0.2))"
            // this is claimed to be in use by lofar.as in the lofargram player
            //
            // it's unclear how this should be interpreted, one way is:
            //     f(v) = a*log2(v) + b
            //     f(v) < 0   |  v < 0.2
            //     f(v) = 1/3 |  v = 1
            // a=0.14356, b=1/3
            //
            // however these values are used:
            // a=0.058918, b=0.16831
            //
            // which doesn't work in Sonic AWE (too dark)
            // fiddling around to get something similar ended up as
            float f = 0.019f * log2f(v) + 0.3333f;
            return f<0.f ? 0.f : f;
        }
    };

    template<>
    class AmplitudeValue<AmplitudeAxis_5thRoot>
    {
    public:
        RESAMPLE_CALL float operator()( float v )
        {
            return 0.4f*powf(v, 0.2);
        }
    };


    class AmplitudeValueRuntime
    {
    public:
        AmplitudeValueRuntime(AmplitudeAxis x):x(x) {}
        RESAMPLE_CALL float operator()( float v )
        {
            switch(x) {
            case AmplitudeAxis_Linear:
                return AmplitudeValue<AmplitudeAxis_Linear>()( v );
            case AmplitudeAxis_Logarithmic:
                return AmplitudeValue<AmplitudeAxis_Logarithmic>()( v );
            case AmplitudeAxis_5thRoot:
                return AmplitudeValue<AmplitudeAxis_5thRoot>()( v );
            case AmplitudeAxis_Real:
                break;
            }
            return 0.f;
        }

    private:
        AmplitudeAxis x;
    };
}

extern "C"
        void blockResampleChunk(
                Tfr::ChunkData::ptr input,
                BlockData::ptr output,
                 ValidInterval validInputs,
                 ResampleArea inputRegion,
                 ResampleArea outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Tfr::FreqAxis outputAxis,
                 Heightmap::AmplitudeAxis amplitudeAxis,
                 float normalization_factor,
                 bool enable_subtexel_aggregation
                 );

/*
extern "C"
void expandStft( cudaPitchedPtrType<float2> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float min_hz,
                 float max_hz,
                 float out_offset,
                 float out_length,
                 unsigned cuda_stream);

extern "C"
void expandCompleteStft( cudaPitchedPtrType<float2> inStft,
                 cudaPitchedPtrType<float> outBlock,
                 float out_min_hz,
                 float out_max_hz,
                 float out_stft_size,
                 float in_offset,
                 float in_min_hz,
                 float in_max_hz,
                 unsigned in_stft_size,
                 unsigned cuda_stream);
*/
extern "C"
void resampleStft( Tfr::ChunkData::ptr input,
                   size_t nScales, size_t nSamples,
                   BlockData::ptr output,
                   ValidInterval outputInterval,
                   ResampleArea inputRegion,
                   ResampleArea outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis,
                   float normalization_factor,
                   bool enable_subtexel_aggregation);

#endif // HEIGHTMAPBLOCK_CU_H
