#ifndef TFR_STFTSETTINGS_H
#define TFR_STFTSETTINGS_H

#include "transform.h"

namespace Tfr {

/**
 * @brief The StftDesc class should create stft transforms.
 *
 * It should describe how an stft transform behaves.
 *
 * It should be copyable and stringable.
 */
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

    // overloaded from TransformDesc
    TransformDesc::Ptr copy() const;
    pTransform createTransform() const;
    float displayedTimeResolution( float FS, float hz ) const;
    FreqAxis freqAxis( float FS ) const;
    unsigned next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    unsigned prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate ) const;
    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    std::string toString() const;
    bool operator==(const TransformDesc& b) const;


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

    /**
     * @brief enable_inverse describes if the inverse transform is enabled.
     * This affects requiredInterval and affectedInterval which only compute
     * what's needed to produce a time-frequency representation of an interval
     * but not what's needed to produce a stable inverse of that representation.
     *
     * An exception is thrown if Transform::inverse is called while
     * enabled_inverse is false.
     *
     * Default: true
     */
    bool enable_inverse() const;
    void enable_inverse(bool);

    float overlap() const;
    WindowType windowType() const;
    std::string windowTypeName() const { return windowTypeName(windowType()); }
    void setWindow(WindowType type, float overlap);

    /**
      Different windows are more sutiable for applying the window on the inverse as well.
      */
    static bool applyWindowOnInverse(WindowType);
    static std::string windowTypeName(WindowType);


private:
    /**
        Default window size for the windowed fourier transform, or short-time fourier transform, stft
        Default value: chunk_size=1<<11
    */
    int _window_size;
    bool _compute_redundant;
    int _averaging;
    bool _enable_inverse;
    float _overlap;
    WindowType _window_type;

public:
    static void test();
};

} // namespace Tfr

#endif // TFR_STFTSETTINGS_H
