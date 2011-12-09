#include "cepstrum.h"
#include "stft.h"
#include "stftkernel.h"

#include "TaskTimer.h"
#include "neat_math.h"

using namespace Signal;
namespace Tfr {

Cepstrum::
        Cepstrum()
{
    stft_ = Stft::SingletonP();
}


pChunk Cepstrum::
        operator()( pBuffer b )
{
    TaskTimer tt("Cepstrum");
    Stft ft = *stft();
    ft.compute_redundant( true );
    pChunk cepstra = ft(b);

    ::cepstrumPrepareCepstra( cepstra->transform_data, 1.f/ft.chunk_size() );

    ft.compute( cepstra->transform_data, cepstra->transform_data, FftDirection_Forward );
    cepstra->freqAxis = freqAxis( cepstra->original_sample_rate );

    TaskInfo("Cepstrum debug. Was %s , returned %s ",
        b->getInterval().toString().c_str(),
        cepstra->getInterval().toString().c_str());

    return cepstra;

}


FreqAxis Cepstrum::
        freqAxis( float FS )
{
    FreqAxis fa;
    fa.setQuefrency( FS, chunk_size());
    return fa;
}


float Cepstrum::
        displayedTimeResolution( float FS, float hz )
{
    return Stft::Singleton().displayedTimeResolution( FS, hz );
}


unsigned Cepstrum::
        next_good_size( unsigned current_valid_samples_per_chunk, float sample_rate )
{
    return stft()->next_good_size(current_valid_samples_per_chunk, sample_rate);
}


unsigned Cepstrum::
        prev_good_size( unsigned current_valid_samples_per_chunk, float sample_rate )
{
    return stft()->prev_good_size(current_valid_samples_per_chunk, sample_rate);
}


std::string Cepstrum::
        toString()
{
    std::stringstream ss;
    ss << "Tfr::Cepstrum (" << stft()->toString() << ")";
    return ss.str();
}


unsigned Cepstrum::
        chunk_size()
{
    return stft()->chunk_size();
}


Signal::pBuffer Cepstrum::
        inverse( pChunk )
{
    throw std::logic_error("Not implemented");
}


Stft* Cepstrum::stft()
{
    return dynamic_cast<Stft*>(stft_.get());
}


} // namespace Tfr
