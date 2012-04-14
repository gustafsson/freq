#include "matlaboperation.h"
#include "hdf5.h"
#include "recorder.h"
#include "tools/support/plotlines.h"

#include "tfr/chunk.h"

// gpumisc
#include "cpumemorystorage.h"

#if defined(__GNUC__)
    #include <unistd.h>
    #include <sys/time.h>
#endif

using namespace std;
using namespace Signal;
using namespace boost;
using namespace boost::posix_time;


namespace Adapters {


MatlabOperation::
        MatlabOperation( Signal::pOperation source, MatlabFunctionSettings* s )
:   OperationCache(source),
    _settings(0)
{
    settings(s);
}


MatlabOperation::
        MatlabOperation()
:   OperationCache(Signal::pOperation()),
    _settings(0)
{
    settings(0);
}



MatlabOperation::
        ~MatlabOperation()
{
    TaskInfo("~MatlabOperation");
    TaskInfo(".");

    settings(0);
}


std::string MatlabOperation::
        name()
{
    if (!_matlab)
        return Operation::name();
    return _matlab->matlabFunctionFilename();
}


std::string MatlabOperation::
        functionName()
{
    if (!_matlab)
        return Operation::name();
    return _matlab->matlabFunction();
}


void MatlabOperation::
        invalidate_samples(const Intervals& I)
{
    // If computing in order and invalidating something that has already been
    // computed
    TaskInfo("MatlabOperation invalidate_samples(%s)", I.toString().c_str());

    Intervals previously_computed = cached_samples() & ~invalid_returns();
    bool start_over = _settings && _settings->computeInOrder() && (I & previously_computed);

    if (start_over)
    {
        // Start over and recompute all blocks again
        restart();
    }
    else
    {
        OperationCache::invalidate_samples( I );

        if (plotlines && source())
            plotlines->clear( I, sample_rate() );
    }
}


bool MatlabOperation::
        dataAvailable()
{
    if (ready_data)
        return true;

    std::string file = _matlab->isReady();
    if (!file.empty())
    {
        TaskTimer tt("Reading data from Matlab/Octave");
        double redundancy=0;
        pBuffer plot_pts;

        try
        {
            ready_data = Hdf5Buffer::loadBuffer( file, &redundancy, &plot_pts );
        }
        catch (const Hdf5Error& e)
        {
            if (Hdf5Error::Type_OpenFailed == e.type() && e.data() == file)
            {
                // Couldn't open it for reading yet, wait
                return false;
            }

            throw e;
        }

        ::remove( file.c_str());

        if (_settings->chunksize() < 0)
            redundancy = 0;

        IntervalType support = (IntervalType)std::floor(redundancy + 0.5);
        _settings->overlap(support);

        if (!ready_data)
        {
            TaskInfo("Couldn't read data from Matlab/Octave");
            return false;
        }

        if (this->plotlines){ // Update plot
            Tools::Support::PlotLines& plotlines = *this->plotlines.get();

            if (plot_pts)
            {
                float start = ready_data->start();
                float length = ready_data->length();

                DataStorageSize N = plot_pts->waveform_data()->size();
                for (unsigned id=0; id<N.depth; ++id)
                {
                    float* p = CpuMemoryStorage::ReadOnly<1>( plot_pts->waveform_data() ).ptr() + id*N.width*N.height;

                    if (3 <= N.height)
                        for (unsigned x=0; x<N.width; ++x)
                            plotlines.set( id, p[ x ], p[ x + N.width ], p[ x + 2*N.width ] );
                    else if (2 == N.height)
                        for (unsigned x=0; x<N.width; ++x)
                            plotlines.set( id, p[ x ], p[ x + N.width ] );
                    else if (1 == N.height)
                        for (unsigned x=0; x<N.width; ++x)
                            plotlines.set( id, start + (x+0.5)*length/N.width, p[ x ] );

                    TaskInfo("Line plot %u now has %u points", id, plotlines.line( id ).data.size());
                }
            }
        }

        Interval oldI = sent_data->getInterval();
        Interval newI = ready_data->getInterval();

        float *oldP = sent_data->waveform_data()->getCpuMemory();
        float *newP = ready_data->waveform_data()->getCpuMemory();

        Intervals J;

        for (unsigned c=0; c<ready_data->channels() && c<sent_data->channels(); c++)
        {
            Interval equal = oldI & newI;
            oldP += equal.first - oldI.first;
            newP += equal.first - newI.first;
            oldP += oldI.count() * c;
            newP += newI.count() * c;
            for (unsigned i=0; i<equal.count();i++)
                if (*oldP != *newP)
                {
                    equal.last = equal.first;
                    break;
                }

            if (equal.count())
                _invalid_returns[c] -= equal;

            J |= newI - equal;
        }

        Signal::Intervals samples_to_invalidate = invalid_returns() & J;
        TaskInfo("invalid_returns = %s, J = %s, invalid_returns & J = %s",
                 invalid_returns().toString().c_str(),
                 J.toString().c_str(),
                 samples_to_invalidate.toString().c_str());

        if (J.empty())
        {
            TaskInfo("Matlab script didn't change anything");
        }
        else
        {
            TaskInfo("Matlab script made some changes");
        }

        if (samples_to_invalidate)
            OperationCache::invalidate_samples( samples_to_invalidate );

        Recorder* recorder = dynamic_cast<Recorder*>(root());
        bool isrecording = 0!=recorder;
        if (isrecording)
        {
            // Leave the process running so that we can continue a recording or change the list of operations
        }
        else
        {
            if (((invalid_samples() | invalid_returns()) - J).empty())
                _matlab->endProcess(); // Finished with matlab
        }

        return true;
    }

    return false;
}


bool MatlabOperation::
        isWaiting()
{
    return _matlab->isWaiting();
}


Interval MatlabOperation::
        intervalToCompute( const Interval& I )
{
    if (0 == I.count())
        return I;

    Signal::Interval J = I;

    if (_settings->chunksize() < 0)
        J = Interval(0, number_of_samples());
    else
    {
        if (_settings->computeInOrder() )
            J = (invalid_samples() | invalid_returns()).fetchInterval( I.count() );
        else
            J = (invalid_samples() | invalid_returns()).fetchInterval( I.count(), I.first );

        if (0<_settings->chunksize())
            J.last = J.first + _settings->chunksize();
    }

    IntervalType support = _settings->overlap();
    Interval signal = getInterval();
    J &= signal;
    Interval K = Intervals(J).enlarge( support ).spannedInterval();

    bool need_data_after_end = K.last > signal.last;
    if (0<_settings->chunksize() && (int)J.count() != _settings->chunksize())
        need_data_after_end = true;

    if (need_data_after_end)
    {
        Recorder* recorder = dynamic_cast<Recorder*>(root());
        bool isrecording = 0!=recorder;
        if (isrecording)
        {
            bool need_a_specific_chunk_size = 0<_settings->chunksize();
            if (need_a_specific_chunk_size)
            {
                if (recorder->isStopped() && !_settings->computeInOrder())
                {
                    // Ok, go on
                }
                else
                {
                    return Interval(0,0);
                }
            }
            else
            {
                if (recorder->isStopped())
                {
                    // Ok, go on
                }
                else
                {
                    return Interval(0,0);
                    // Don't use any samples after the end while recording
                    K &= signal;

                    if (Intervals(K).shrink(support).empty())
                        return Interval(0,0);
                }
            }
        }
    }

    return K;
}


pBuffer MatlabOperation::
        readRaw( const Interval& I )
{
    if (!_matlab)
        return pBuffer();

    try
    {
        if (dataAvailable())
        {
            Signal::pBuffer b = ready_data;
            ready_data.reset();
            TaskInfo("MatlabOperation::read(%s) Returning ready data %s, %u channels",
                     I.toString().c_str(),
                     b->getInterval().toString().c_str(),
                     b->waveform_data()->size().height );
            return b;
        }

        if (_matlab->hasProcessEnded())
        {
            if (_matlab->hasProcessCrashed())
            {
                TaskInfo("MatlabOperation::read(%s) process ended", I.toString().c_str() );

                return source()->readFixedLength( I );
            }
            else
            {
                restart();
            }
        }

        if (!isWaiting())
        {
            TaskTimer tt("MatlabOperation::read(%s)", I.toString().c_str() );
            Interval K = intervalToCompute(I);

            if (0 == K.count())
                return pBuffer();

            // just 'read()' might return the entire signal, which would be way to
            // slow to export in an interactive manner
            sent_data = source()->readFixedLengthAllChannels( K );

            string file = _matlab->getTempName();

            IntervalType overlap = _settings->overlap();
            Hdf5Buffer::saveBuffer( file, *sent_data, overlap );

            TaskInfo("Sending %s to Matlab/Octave", sent_data->getInterval().toString().c_str() );
            _matlab->invoke( file );
        }
        else
        {
            TaskInfo("MatlabOperation::read(%s) Is waiting for Matlab/Octave to finish", I.toString().c_str() );
        }
    }
    catch (const std::runtime_error& e)
    {
        TaskInfo("MatlabOperation caught %s", e.what());
        _matlab->endProcess();
        throw std::invalid_argument( e.what() ); // invalid_argument doesn't crash the application
    }

    return pBuffer();
}


void MatlabOperation::
        restart()
{
    _cache.clear();
    _matlab.reset();

    if (_settings)
    {
        _matlab.reset( new MatlabFunction( _settings->scriptname(), 4, _settings ));

        OperationCache::invalidate_samples( Signal::Intervals::Intervals_ALL );
    }

    if (plotlines)
        plotlines->clear();
}


void MatlabOperation::
        settings(MatlabFunctionSettings* settings)
{
    if (_settings && _settings->operation)
    {
        _settings->operation = 0;
        delete _settings;
    }

    _settings = settings;

    restart();

    if (_settings)
    {
        _settings->operation = this;
    }
}

} // namespace Adapters
