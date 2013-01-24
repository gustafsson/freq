#ifndef TFR_STFTSETTINGS_H
#define TFR_STFTSETTINGS_H

#include "transform.h"

namespace Tfr {

class SaweDll StftDesc: public TransformDesc
{
public:
    enum WindowType
    {
        WindowType_Rectangular,
        WindowType_Hann,
        WindowType_Hamming,
        WindowType_Tukey,
        WindowType_Cosine,
        WindowType_Lanczos,
        WindowType_Triangular,
        WindowType_Gaussian,
        WindowType_BarlettHann,
        WindowType_Blackman,
        WindowType_Nuttail,
        WindowType_BlackmanHarris,
        WindowType_BlackmanNuttail,
        WindowType_FlatTop,
        WindowType_NumberOfWindowTypes
    };

    StftDesc();

    virtual pTransform createTransform() const;
    virtual float displayedTimeResolution( float FS, float hz ) const;
    virtual FreqAxis freqAxis( float FS ) const;
    virtual unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    virtual unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    virtual std::string toString() const;
    virtual bool operator==(const TransformDesc& b) const;


    int increment() const;
    int chunk_size() const;
    int set_approximate_chunk_size( unsigned preferred_size );

    /// @ Try to use set_approximate_chunk_size(unsigned) unless you need an explicit stft size
    void set_exact_chunk_size( unsigned chunk_size );

    /**
        If false (default), operator() will do a real-to-complex transform
        instead of a full complex-to-complex.

        (also known as R2C and C2R transforms are being used instead of C2C
        forward and C2C backward)
    */
    bool compute_redundant() const;
    void compute_redundant(bool);

    int averaging() const;
    void averaging(int);

    float overlap() const;
    WindowType windowType() const;
    std::string windowTypeName() const { return windowTypeName(windowType()); }
    void setWindow(WindowType type, float overlap);

    /**
      Different windows are more sutiable for applying the window on the inverse as well.
      */
    static bool applyWindowOnInverse(WindowType);
    static std::string windowTypeName(WindowType);


    /**
      'affected_samples' describes where it is possible that
        'source()->readFixedLength( I ) != readFixedLength( I )'
      '~affected_samples' describes where it is guaranteed that
        'source()->readFixedLength( I ) == readFixedLength( I )'

      A filter is allowed to be passive in some parts and nonpassive in others.
      'affected_samples' describes where. 'affected_samples' is allowed to
      change over time as well.

      As default all samples are possibly affected by an Operation.
      */
    virtual Signal::Intervals affected_samples() const;

private:
    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    int _window_size;
    bool _compute_redundant;
    int _averaging;
    float _overlap;
    WindowType _window_type;
};

} // namespace Tfr

#endif // TFR_STFTSETTINGS_H
