#ifndef SIGNALAUDIOFILE_H
#define SIGNALAUDIOFILE_H

/*
    input signal, 1D, can be sound, matlab vector or data-file
        --- Different sources ---
        a sound can be read from disc
        a sound can be recorded
        a matlab vector can be supplied by mex calls
        a data file can be binary data or csv
        all signals in Sonic AWE have a sample rate, data files need to supply this
        Two, or more, sources can be overloaded, the sample rate of the total is equal
        to the highest sample rate. Samples from sources of lower sample rates need to
        be resampled.

        The input signal could always be interpreted as a stream, from which a number
        of elements, floats, can be read. But the access pattern only needs a simple
        API providing a read method, which returns a signal buffer (eq to streambuf):

        Signal::pBuffer b = Signal::Source::read( firstSample, number_of_samples )

        A Signal doesn't have any pBuffers, it's up to each implementation to provide them
        when requested for. A Signal::MicrophoneRecorder might for instance have a member
        std::list<Signal::pBuffer> to store recordings in, internally. These would be
        assembled to another Buffer when requested by read.

        --- Different sinks ---
        A partial Tfd is then computed from the Buffer. The inverse Cwt can be put into
        a Sink for playback.

        void Signal::Sink::put( Signal::pBuffer )
        unsigned Signal::Sink::expected_samples_left=0;
        void Signal::Playback::put( Signal::pBuffer )

        A Sink is used for audio playback, and possibly saving data as well. A Sink can be
        registered as the callback for an inverse Cwt. Signal::Playback implements
        Signal::Sink and waits for enough data before beginning playback. To do proper
        estimations, Signal::Sink::expected_samples_left can be used to tell how many
        more samples that can be expected to be put into the chunk. If left to its default
        value of 0 playback will start playing immediately when each new chunk is received.
        Signal::Save opens a .wav-file for writing and writes the result when
        expected_samples_left is matched, or is 0. If Buffer has an offset, WavWriter will
        write with the same offset.

        void Signal::WavWriter::put( Signal::pBuffer )

        WavWriter can also have an additional global offset so that two WavWriters can work
        at the same time. One WavWriter writes 'selection.wav', this WavWriter have a global
        offset such that the start of the wav file equals the start of the selection.
        Another WavWrtier writes to $cache as computed by the platform independent equivalent of
        "$wavfile = `echo $INPUT_FILE_NAME | sed 's/\.[^\.]*$//`.wav~'
        "$cache = $wavfile~
        Note the tilde at the end. If the user chooses to export her/his work at the end of
        a session, $cache is renamed to $wavfile. If "echo $INPUT_FILE_NAME | grep -i \.wav$"
        returns 0, the original file ends with .wav and $cache is renamed to $INPUT_FILE_NAME
        The user should probably be asked about overwriting the original sound.

        --- Rendering Waveforms ---
        The waveform should be rendered like the TFD, in blocks - WaveformBlocks - in
        of different scales. Those blocks are created by doing reads from a
        Signal::Source and building up Tfd::WaveformBlock, which are rendered by
        Tfd::Waveform. "Waveform" refers to "the shape and form of a signal". That is,
        a waveform is not the signal itself, but a Waveform describes the looks of a signal.

        As for the Tfd, a WaveformBlock isn't created until it is requested for.

        Signals are never stored on the GPU, they are transfered when requested for, used
        for one calculation/kernel execution and then released.

        --- Channels? ---
        Signals are treated as 1D vectors. If a signal contains 'n' channels they must be
        'n' sources and 'n' different Tfrs. It is possible to render two Tfrs on top of
        eachother, in different colors. But each computation step is independent of other
        channels. Thus Signal::Buffer and Tfr::Chunk is unaware of the term channel.

        Playback can be set up such that incoming data are streamed to different channels,
        Signal::Sink

        --- Operations ---
        A Signal::Operation is a Signal::Source that reads from another Signal::Source.

        A quite complex Operation is a Signal::Cache. A cache remembers the output of another Signal::Source.
        The user has no influence on Signal::Cache, instead Sonic AWE makes descisions based on heuristics where
        to insert Caches.

        A trivial operation is to move a Source in time. This simply changes the offset value for the Buffer.

        A simple operation is to alter the sample rate of a Source, it requires some kind of resampling.

        A non-so-trivial related operation is to scale a Source in time. This could be achieved by implementing
        the local cosine transform, or by some other means in transform.

        A quite trivial operation is to superimpose a second Signal::Source onto the first one. This requires
        a resampling operation if they don't have the same sample rate. These kind of operations are called
        Layers.

        Some non-so-trivial operations require Filters on the Cwt domain and are called FilterOperations. The
        filter is always a FilterChain and might in turn invoke several filters.

        Buffer = Tfr::InverseCwt( _filter_chain ( Tfr::Cwt ( buffer ) ) )
        (If the filterchain is empty, the FilterOperation doesn't do anything and returns the Buffer immediately.)

        Filters suited for the filterchain is for instance to move a Buffer in frequency, or to scale a signal in
        frequency.

        FilterOperations are the only part of Signal:: that is aware of Tfr.
        Tfr is well aware of Signal, but not FilterOperations.

        This kind of structure is called streams or image streams.
        Operations are connected to form a chain of operations, and there are an OperationIterator which can travel
        down through the child operations. However there are no explicit structure to keep all Operations "under the
        same roof". If an operation is deleted or altered, all operations that are depending on that operation will
        have to be recomputed. So the cache will have to be invalidated.

        For this purpose there is an InvalidSamplesDescriptor class

        protected: InvalidSamplesDescriptor Operation::updateIsd()
        {
            foreach ( Operation* child, ChildrenList ( this ) ) {
                _myIsd |= position_transform( child->updateIsd() );
            }

            return _myIsd;
        }

        protected: Operation::sample InvalidSamplesDescriptor

        The InvalidSamplesDescriptor defines which cache regions that are valid. It is real quick to process the
        tree once each frame.

        If an operation has been altered, but in a way such that most parts of the Source remain unaffected (for instance
        a small circle-remove filter is inserted in a FilterOperation). The InvalidSamplesDescriptor for that
        FilterOperation is updated by _myIsd |= oldFilter->updateIsd(). Each filter is responsible for maintaining
        its own InvalidSamplesDescriptor for data that has not been requested for.

    Time Frequency Representation/Distribution transform,
    Tfr is mostly a bunch of functionals. Some functionals doesn't keep much meaningful states and are thus transient.
        Tfr::Stft takes a Signal::Buffer and produces a Tfr::StftChunk. transient
        Tfr::Cwt takes a Signal::Buffer and produces a Tfr::Chunk.
        Tfr::Filter applies filter operations to a Tfr::Chunk.
        Tfr::InverseCwt takes a Tfr::Chunk and produces a Signal::Buffer. Has a selectionfilter.
        Tfr::CwtCompleteCallback abstract callback.
        Tfr::CwtFilterChangedCallback abstract callback.
        Tfr::CwtComputer computes a Cwt chunk, applies a pFilter (ChainFilter) and calls the callback.
        Tfr::CwtComputer is also called whenever a filter has changed. CwtQueue passes this information on to callbacks.
        Tfr does not known about Heightmap.
        Hightmap::MergeChunk and TfrHeightmap known about Tfr.
        The rest of Hightmap MergeChunk does not known about Tfr.
        Heightmap::MergeChunk updates a Heightmap::HeightBlock with a Tfr::Chunk. Transient.
        Heightmap::MergeChunk can also update a Heightmap::HeightBlock with a Tfr::StftChunk. Transient.
        Heightmap::HeightBlock, vbos, computing slope and rendering with shaders.
        Heightmap::HeightCollection collection of HeightBlocks. == old spectrogram
        Heightmap::PlotHeightmap takes a pHeightmap as argument == old spectrogram-renderer

        Heightmap::TfrHeightmap
            implements abstract HeightCollection
            implements abstract ChunkCompleteCallback
            takes a Signal::pSource as constructing argument, source can be Signal::MicrophoneRecorder,
                multiple sources is implemented after layer have been implemented
            also takes a Tfr::ChunkQueue

            When a brand new HeightBlock is requested for, TfrHeightmap fetches a Buffer over those samples,
            processes it with Tfr::Stft and sends it to Heightmap::MergeChunk. It may happen that the entire buffer would
            be to large, in which case it is divided in smaller sections until it fits on the GPU, the process is
            repeated until the entire block is filled. This is a blocking operation.

            However, though filled this still renders the block incomplete. And the MergeChunk is kept in the list
            of MergeChunks to complete. MergeChunk keeps track of which
            It could optionally be filled from other HeightBlocks which are on resolution levels +-1, if they don't have
            a mergeblock associated... And if they do copy all but keep track of what is complete and what registered
            is not.

            When an imcomplete HeightBlock is requested for, TfrHeightmap determines a good chunk size
            (based on the time it took to compute the last chunk, or based on command line arguments),
            taking wavelet_std into account, and puts this into CwtQueue if the CwtQueue is empty.

            When CwtQueue has finished and calls CwtCompleteCallback the incoming (in the callback) Tfr::Chunk
            is then sent to Heightmap::MergeChunk. And the block is updated.

       CwtQueue
            is a queue of ranges, might be worked off by one or multiple GPUs

            1. ask pSource for a Buffer over those samples
            2. The retrieved Signal::Buffer is sent to Tfr::Cwt,
            3. The Tfr::Chunk is sent to this->operation (which is likely is an operationlist).
            4. The Tfr::Chunk is sent to the callback, there might be multiple callbacks on each queue.

            CwtCompleteCallback takes a CwtQueue as constructing argument and can thus access the CwtQueue mutex
            and be more threadsafe. CwtQueue aquires the mutex when calling CwtCompleteCallback, and CwtCompleteCallback
            aquires the mutex upon destruction, where it also removes itself from CwtQueue.

       Playback and Tfr::InverseCwt

            When playback is requested by the user, the CwtQueue is cleared and rebuilt with sample sections that
            are supposed to be played. The PlaybackCallback is registered and Signal::Playback is created with the total
            number of samples to play. The selectionfilter in InverseCwt is used to compute the range. At each timestep,
            one item is worked off the CwtQueue. The chunk that comes out is passed on to InverserCwt and the Buffer is
            sent to Signal::Playback which will start playing if it estimates that there wouldn't be any chops.

            unsigned Signal::Playback::getCurrentPosition() can be used to query the playback marker and get a value
            for something to render.

    When the filter in CwtQueue is changed, blocks need to be invalidated. The Playback is ok, it is invalidated when
    started. And if the inverse has already been computed we can't afford to discard that.
    How is the connection made between CwtQueue and creating new MergeChunks in HightCollection for affected blocks?
    A copy of filters could be kept, that would be simple. But a filter could be potentially really complex, and stores
    references to external data, of unknown kind. A copy of such a filter might still refer to the same source,
    and if the source would change... naa... left open... A filter has no notion of parent filters neither.


    Filters are applied to Tfr::Chunks

    A filter should
 Invalidate children...
    It is possible to keep Signals can be combined
*/

#include "signal-source.h"

namespace audiere
{
    class SampleSource;
}

namespace Signal
{

class Audiofile: public Source
{
public:

    Audiofile();
    Audiofile(const char* filename);

    virtual pBuffer read( unsigned firstSample, unsigned numberOfSamples );

    pBuffer getChunkBehind() { return _waveform; }
    void setChunk( pBuffer chunk ) { _waveform = chunk; }
    void play();
    pBuffer getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Buffer::Interleaved interleaved );
private:
    void appendChunk( pBuffer chunk );

    /**
      Writes wave audio with 16 bits per sample
      */
    void writeFile( const char* filename );
    pSource crop();

    unsigned channel_count() {        return _waveform->waveform_data->getNumberOfElements().height; }
    virtual unsigned sample_rate();
    virtual unsigned number_of_samples();

private:
    audiere::SampleSource* _source;

    pBuffer _waveform;

    std::string _last_filename;
};

} // namespace Signal

#endif // SIGNALAUDIOFILE_H
