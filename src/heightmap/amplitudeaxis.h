#ifndef AMPLITUDEAXIS_H
#define AMPLITUDEAXIS_H

namespace Heightmap {

enum AmplitudeAxis {
    AmplitudeAxis_Linear,
    AmplitudeAxis_Logarithmic,
    AmplitudeAxis_5thRoot,
    AmplitudeAxis_Real
};


enum ComplexInfo {
    ComplexInfo_Amplitude_Weighted,
    ComplexInfo_Amplitude_Non_Weighted,
    ComplexInfo_Phase
};

} // namespace Heightmap

#endif // AMPLITUDEAXIS_H
