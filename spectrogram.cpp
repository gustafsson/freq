#include "spectrogram.h"

using namespace std;

template<typename T>
class Toptrlist {
    list<T*> ptrList;
public:
    Toptrlist(list<T*> ptrList) :ptrList(ptrList) {}

    operator list<T*>() { return ptrList; }
    void operator()(const T& v ) { ptrList.push_back(&v); }
};

bool operator<(const Spectrogram_chunk& a, const Spectrogram_chunk& b) {
    return a.fzoom<b.fzoom; // dummy
}

class SortSlots {
public:
    float t,f;
};

Spectrogram::Spectrogram( boost::shared_ptr<Waveform> waveform, unsigned waveform_channel )
:   _original_waveform(waveform),
    _original_waveform_channel(waveform_channel)
{}

Spectrogram_chunk* Spectrogram::getBlock( float t, float f, float dt, float df) {
    // Grab a temprary list to fiddle with
    //list<Spectrogram_slot*> pslots = Toptrlist<Spectrogram_slot>(_slots);

    // Search among slots for the closest one

//    boost::shared_ptr _slots
//    Spectrogram
    t=f+dt+df;
    return 0;
}
