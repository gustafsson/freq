#include "spectrogram.h"

template<typename T>
class Toptrlist {
    list<T*> ptrList;
public:
    operator list<T*>() { return ptrList; }
    void operator()(const T& v ) { ptrList.push_back(&v); }
};

class SortSlots {
public:
    float t,f;
    bool operator<(const SpectogramBlock& a, const SpectogramBlock& b) {

    }
};

Spectogram::Spectogram( boost::shared_ptr<Waveform> waveform, unsigned waveform_channel=0 )
:   _original_waveform(waveform),
    _original_waveform_channel(waveform_channel)
{}

SpectogramBlock* Spectogram::getBlock( float t, float f, float dt, float df) {
    // Grab a temprary list to fiddle with
    list<SpectogramSlot*> pslots = Toptrlist(_slots);

    // Search among slots for the closest one

    boost::shared_ptr _slots
    Spectogram
}
