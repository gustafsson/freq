#ifndef WAVEFORMRECORDER_H
#define WAVEFORMRECORDER_H

namespace Waveform {

class Recorder
{
public:
    Recorder(pWaveform waveform);
    ~Recorder();

    bool isRecording();
};

} // namespace Waveform

#endif // WAVEFORMRECORDER_H
