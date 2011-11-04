#ifndef HEIGHTMAPBLOCK_CU_H
#define HEIGHTMAPBLOCK_CU_H

#include "tfr/freqaxis.h"
#include "tfr/chunkdata.h"

typedef DataStorage<float> BlockData;

struct BlockArea
{
    BlockArea(float x1, float y1, float x2, float y2)
        : x1(x1), y1(y1), x2(x2), y2(y2)
    {}

    float x1, y1, x2, y2;
};

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

    // TODO find a better name
    enum ComplexInfo {
        ComplexInfo_Amplitude_Weighted,
        ComplexInfo_Amplitude_Non_Weighted,
        ComplexInfo_Phase
    };

    enum AmplitudeAxis {
        AmplitudeAxis_Linear,
        AmplitudeAxis_Logarithmic,
        AmplitudeAxis_5thRoot
    };
};

extern "C"
        void blockResampleChunk(
                Tfr::ChunkData::Ptr input,
                BlockData::Ptr output,
                 ValidInputInterval validInputs,
                 BlockArea inputRegion,
                 BlockArea outputRegion,
                 Heightmap::ComplexInfo transformMethod,
                 Tfr::FreqAxis inputAxis,
                 Tfr::FreqAxis outputAxis,
                 Heightmap::AmplitudeAxis amplitudeAxis
                 );

extern "C"
void blockMerge( BlockData::Ptr inBlock,
                 BlockData::Ptr outBlock,
                 BlockArea in_area,
                 BlockArea out_area );
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
                   BlockArea inputRegion,
                   BlockArea outputRegion,
                   Tfr::FreqAxis inputAxis,
                   Tfr::FreqAxis outputAxis,
                   Heightmap::AmplitudeAxis amplitudeAxis );

#endif // HEIGHTMAPBLOCK_CU_H
