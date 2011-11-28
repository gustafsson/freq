#ifndef HEIGHTMAPBLOCK_CU_H
#define HEIGHTMAPBLOCK_CU_H

#include "tfr/freqaxis.h"
#include "tfr/chunkdata.h"
#include "resampletypes.h"
#include "amplitudeaxis.h"

typedef DataStorage<float> BlockData;

struct ValidInputInterval
{
    ValidInputInterval(unsigned first, unsigned last)
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
            return 0.02f * (log2f(0.0001f + v) - log2f(0.0001f));
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
            }
            return 0.f;
        }

    private:
        AmplitudeAxis x;
    };
};

extern "C"
        void blockResampleChunk(
                Tfr::ChunkData::Ptr input,
                BlockData::Ptr output,
                 ValidInputInterval validInputs,
                 ResampleArea inputRegion,
                 ResampleArea outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Tfr::FreqAxis outputAxis,
                 Heightmap::AmplitudeAxis amplitudeAxis,
                 float normalization_factor,
                 bool full_resolution
                 );

extern "C"
void blockMerge( BlockData::Ptr inBlock,
                 BlockData::Ptr outBlock,
                 ResampleArea in_area,
                 ResampleArea out_area );

extern "C"
void blockClearPart( BlockData::Ptr block,
                 unsigned start_t );

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
void resampleStft( Tfr::ChunkData::Ptr input,
                   size_t nScales, size_t nSamples,
                   BlockData::Ptr output,
                   ResampleArea inputRegion,
                   ResampleArea outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis,
                   float normalization_factor);

#endif // HEIGHTMAPBLOCK_CU_H
