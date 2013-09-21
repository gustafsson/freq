#include "stft.h"

#include "computationkernel.h"
#include "demangle.h"
#include "neat_math.h"

namespace Tfr {

StftDesc::
        StftDesc()
    :
      _window_size( 1<<11 ),
      _compute_redundant(false),
      _averaging(1),
      _overlap(0.f),
      _window_type(WindowType_Rectangular)
{
    compute_redundant( _compute_redundant );
}


TransformDesc::Ptr StftDesc::
        copy() const
{
    return TransformDesc::Ptr(new StftDesc(*this));
}


pTransform StftDesc::
        createTransform() const
{
    return pTransform(new Stft(*this));
}


FreqAxis StftDesc::
        freqAxis( float FS ) const
{
    FreqAxis fa;

    if (compute_redundant())
        fa.setLinear( FS, chunk_size()-1 );
    else
        fa.setLinear( FS, chunk_size()/2 );

    return fa;
}


float StftDesc::
        displayedTimeResolution( float FS, float /*hz*/ ) const
{
    return 0.125f*chunk_size() / FS;
}


unsigned StftDesc::
        next_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ ) const
{
    int window_size = chunk_size();
    int averaging = this->averaging();

    if ((int)current_valid_samples_per_chunk<window_size)
        return window_size;

    size_t maxsize = std::min( (size_t)(64<<20), (size_t)availableMemoryForSingleAllocation() );
    maxsize = std::max((size_t)window_size, maxsize/(3*sizeof(ChunkElement)));
    unsigned alignment = window_size*averaging;
    return std::min((unsigned)maxsize, spo2g(align_up(current_valid_samples_per_chunk, alignment)/alignment)*alignment);
}


unsigned StftDesc::
        prev_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ ) const
{
    int window_size = chunk_size();
    int averaging = this->averaging();

    if ((int)current_valid_samples_per_chunk<2*window_size)
        return window_size;

    size_t maxsize = std::min( (size_t)(64<<20), (size_t)availableMemoryForSingleAllocation() );
    maxsize = std::max((size_t)window_size, maxsize/(3*sizeof(ChunkElement)));
    unsigned alignment = window_size*averaging;
    return std::min((unsigned)maxsize, lpo2s(align_up(current_valid_samples_per_chunk, alignment)/alignment)*alignment);
}


Signal::Interval StftDesc::
        requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const
{
    // _averaging shouldn't be a property of the transform but of the visualization
    Signal::IntervalType increment = _averaging*this->increment();
    int chunk_size = _averaging*this->chunk_size ();

    // align down to multiple of increment
    int half1 = chunk_size/2;
    int half2 = chunk_size - half1;
    int first = align_down(I.first, increment) - half1,
        last = align_up(I.last, increment) + half2;

    if (expectedOutput)
        *expectedOutput = Signal::Interval(
                first + chunk_size-increment,
                last - chunk_size+increment);

    return Signal::Interval(first, last);
}


Signal::Interval StftDesc::
        affectedInterval( const Signal::Interval& I ) const
{
    return requiredInterval(I, 0);
}


unsigned oksz(unsigned x)
{
    if (0 == x)
        x = 1;

    Fft fft;
    unsigned ls = fft.lChunkSizeS(x+1, 4);
    unsigned sg = fft.sChunkSizeG(x-1, 4);
    if (x-ls < sg-x)
        return ls;
    else
        return sg;
}

int StftDesc::
        set_approximate_chunk_size( unsigned preferred_size )
{
    //_window_size = 1 << (unsigned)floor(log2f(preferred_size)+0.5);
    _window_size = oksz( preferred_size );

    size_t free = availableMemoryForSingleAllocation();

    unsigned multiple = 0;
    multiple++; // input
    multiple++; // output
    multiple++; // overhead during computaion

    unsigned slices = 1;
    if (slices * _window_size*multiple*sizeof(Tfr::ChunkElement) > free)
    {
        unsigned max_size = free / (slices*multiple*sizeof(Tfr::ChunkElement));
        _window_size = Fft().lChunkSizeS(max_size+1, 4);
    }

    _window_size = std::max(4, _window_size);

    return _window_size;

//    if (_ok_chunk_sizes.empty())
//        build_performance_statistics(true);

//    std::vector<unsigned>::iterator itr =
//            std::lower_bound( _ok_chunk_sizes.begin(), _ok_chunk_sizes.end(), preferred_size );

//    unsigned N1 = *itr, N2;
//    if (itr == _ok_chunk_sizes.end())
//    {
//        N2 = spo2g( preferred_size - 1 );
//        N1 = lpo2s( preferred_size + 1 );
//    }
//    else if (itr == _ok_chunk_sizes.begin())
//        N2 = N1;
//    else
//        N2 = *--itr;

//    _chunk_size = absdiff(N1, preferred_size) < absdiff(N2, preferred_size) ? N1 : N2;
//    return _chunk_size;
}


void StftDesc::
        set_exact_chunk_size( unsigned chunk_size )
{
    _window_size = chunk_size;
}


bool StftDesc::
        compute_redundant() const
{
    return _compute_redundant;
}


void StftDesc::
        compute_redundant(bool value)
{
    _compute_redundant = value;
    if (_compute_redundant)
    {
        // free unused memory
        //_handle_ctx_c2r(0,0);
        //_handle_ctx_r2c(0,0);
    }
    else
    {
        // free unused memory
        //_handle_ctx_c2c(0,0);
    }
}


int StftDesc::
        averaging() const
{
    return _averaging;
}


void StftDesc::
        averaging(int value)
{
    if (1 > value)
        value = 1;
    if (10 < value)
        value = 10;

    _averaging = value;
}


float StftDesc::
        overlap() const
{
    return _overlap;
}


StftDesc::WindowType StftDesc::
        windowType() const
{
    return _window_type;
}


std::string StftDesc::
        windowTypeName(WindowType type)
{
    switch(type)
    {
    case WindowType_Rectangular: return "Rectangular";
    case WindowType_Hann: return "Hann";
    case WindowType_Hamming: return "Hamming";
    case WindowType_Tukey: return "Tukey";
    case WindowType_Cosine: return "Cosine";
    case WindowType_Lanczos: return "Lanczos";
    case WindowType_Triangular: return "Triangular";
    case WindowType_Gaussian: return "Gaussian";
    case WindowType_BarlettHann: return "Barlett-Hann";
    case WindowType_Blackman: return "Blackman";
    case WindowType_Nuttail: return "Nuttail";
    case WindowType_BlackmanHarris: return "Blackman-Harris";
    case WindowType_BlackmanNuttail: return "Blackman-Nuttail";
    case WindowType_FlatTop: return "Flat top";
    default: return "Unknown window type";
    }
}


bool StftDesc::
        applyWindowOnInverse(WindowType type)
{
    switch (type)
    {
    case WindowType_Rectangular: return false;
    case WindowType_Hann: return true;
    case WindowType_Hamming: return true;
    case WindowType_Tukey: return false;
    case WindowType_Cosine: return true;
    case WindowType_Lanczos: return true;
    case WindowType_Triangular: return false;
    case WindowType_Gaussian: return false;
    case WindowType_BarlettHann: return false;
    case WindowType_Blackman: return false;
    case WindowType_Nuttail: return false;
    case WindowType_BlackmanHarris: return false;
    case WindowType_BlackmanNuttail: return false;
    case WindowType_FlatTop: return false;
    default: return false;
    }
}


void StftDesc::
        setWindow(WindowType type, float overlap)
{
    _window_type = type;
    _overlap = std::max(0.f, std::min(0.98f, overlap));
}


int StftDesc::
        increment() const
{
    int window_size = chunk_size();
    float overlap = this->overlap();
    float wanted_increment = window_size*(1.f-overlap);

    // _window_size must be a multiple of increment for inverse to be correct
    int divs = std::max(1.f, std::floor(window_size/wanted_increment));
    while (window_size/divs*divs != window_size && divs < window_size)
    {
        int s = window_size/divs;
        divs = (window_size + s - 1)/s;
    }
    divs = std::min( window_size, std::max( 1, divs ));

    return window_size/divs;
}


int StftDesc::
        chunk_size() const
{
    return _window_size;
}


std::string StftDesc::
        toString() const
{
    std::stringstream ss;
    ss << vartype(*this) << ", "
       << "window_size=" << chunk_size()
       << ", redundant=" << (compute_redundant()?"C2C":"R2C")
       << ", overlap=" << overlap()
       << ", window_type=" << windowTypeName();
    return ss.str();
}


bool StftDesc::
        operator==(const TransformDesc& b) const
{
    // Works for CepstrumParams as well.
    if (typeid(b)!=typeid(*this))
        return false;

    const StftDesc* p = dynamic_cast<const StftDesc*>(&b);

    return _window_size == p->_window_size &&
            _compute_redundant == p->_compute_redundant &&
            _averaging == p->_averaging &&
            _overlap == p->_overlap &&
            _window_type == p->_window_type;
}

} // namespace Tfr

#include "exceptionassert.h"

using namespace Signal;

namespace Tfr {

std::ostream& operator<< (std::ostream& o, const AxisScale& a) {
    switch(a) {
    case AxisScale_Linear: return o << "Linear";
    case AxisScale_Logarithmic: return o << "Logarithmic";
    case AxisScale_Quefrency: return o << "Quefrency";
    case AxisScale_Unknown: return o << "Unknown";
    default: return o << "Invalid";
    }
}


std::ostream& operator<< (std::ostream& o, const FreqAxis& i) {
    return o << "axis = " << i.axis_scale
      << ", step = " << i.f_step
      << ", max = " << i.max_frequency_scalar;
}


std::ostream& operator<< (std::ostream& o, const TransformDesc& d) {
    return o << d.toString ();
}


void StftDesc::
        test()
{
    // It should create stft transforms.
    {
        StftDesc d;
        EXCEPTION_ASSERT_EQUALS( vartype(*d.createTransform()), vartype(Stft()) );
    }


    // It should describe how an stft transform behaves.
    {
        StftDesc d;
        EXCEPTION_ASSERT_EQUALS( vartype(*d.copy()), vartype(StftDesc()) );
        EXCEPTION_ASSERT_EQUALS( d.displayedTimeResolution(4,1), 0.125f*d.chunk_size() / 4 );
        EXCEPTION_ASSERT_EQUALS( d.freqAxis(4), [](){ FreqAxis f; f.setLinear(4, 1024); return f; }());
        EXCEPTION_ASSERT_EQUALS( d.next_good_size (1,4), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.prev_good_size (1,4), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.next_good_size (2047,4), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.next_good_size (2048,4), 4096 );
        EXCEPTION_ASSERT_EQUALS( d.next_good_size (2049,4), 8192 ); // <-- unexpected behaviour
        EXCEPTION_ASSERT_EQUALS( d.prev_good_size (4097,4), 4096 );
        EXCEPTION_ASSERT_EQUALS( d.prev_good_size (4096,4), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.prev_good_size (4095,4), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.prev_good_size (2049,4), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.requiredInterval (Interval(1,2), 0 ), Interval(-1024,3*1024) );
        EXCEPTION_ASSERT_EQUALS( d.requiredInterval (Interval(-1,2), 0 ), Interval(-3*1024,3*1024) );
        EXCEPTION_ASSERT_EQUALS( d.requiredInterval (Interval(-1,0), 0 ), Interval(-3*1024,1024) );
        EXCEPTION_ASSERT_EQUALS( d.affectedInterval (Interval(1,2) ), Interval(-1024,3*1024) );
        EXCEPTION_ASSERT( !d.toString().empty () );
        EXCEPTION_ASSERT_EQUALS( d, *d.copy () );

        EXCEPTION_ASSERT_EQUALS( d.increment(), 2048 );
        EXCEPTION_ASSERT_EQUALS( d.chunk_size(), 2048 );
        d.set_approximate_chunk_size(2);
        EXCEPTION_ASSERT_EQUALS( d.chunk_size(), 4 );
        d.set_approximate_chunk_size(7);
        EXCEPTION_ASSERT_EQUALS( d.chunk_size(), 8 );
        d.set_approximate_chunk_size(9);
        EXCEPTION_ASSERT_EQUALS( d.chunk_size(), 8 );
        d.set_exact_chunk_size (7);
        EXCEPTION_ASSERT_EQUALS( d.chunk_size(), 7 );
        EXCEPTION_ASSERT( !d.compute_redundant () );
        d.compute_redundant (true);
        EXCEPTION_ASSERT( d.compute_redundant () );
        EXCEPTION_ASSERT( d.averaging () );
        EXCEPTION_ASSERT_EQUALS( d.averaging (), 1 );
        d.averaging (2);
        EXCEPTION_ASSERT_EQUALS( d.averaging (), 2 );
        EXCEPTION_ASSERT_EQUALS( d.overlap (), 0.f );
        EXCEPTION_ASSERT_EQUALS( d.windowType (), WindowType_Rectangular );
        EXCEPTION_ASSERT_EQUALS( d.windowTypeName (), "Rectangular" );
        d.setWindow (WindowType_Nuttail, 0.75);
        EXCEPTION_ASSERT_EQUALS( d.windowTypeName (), "Nuttail" );
        EXCEPTION_ASSERT_EQUALS( d.overlap (), 0.75 );
    }
}



} // namespace Tfr
