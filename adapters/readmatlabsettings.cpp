#include "readmatlabsettings.h"

#include "hdf5.h"

// gpumisc
#include "demangle.h"

// boost
#include <boost/algorithm/string.hpp>

// qt
#include <QTimer>
#include <QFileInfo>

using namespace std;
using namespace boost;

namespace Adapters {

void ReadMatlabSettings::
        readSettingsAsync(QString filename, QObject *receiver, const char *member, const char* failedmember)
{
    if (!QFileInfo(filename).exists())
        return;

    ReadMatlabSettings* settings = new ReadMatlabSettings( filename, MetaData_Settings );
    settings->connect( settings, SIGNAL(settingsRead(Adapters::DefaultMatlabFunctionSettings)), receiver, member );
    if (failedmember)
        settings->connect( settings, SIGNAL(failed(QString, QString)), receiver, failedmember );
    settings->setParent( receiver );

    settings->readAsyncAndDeleteSelfWhenDone();
}


ReadMatlabSettings::
        ReadMatlabSettings( QString filename, MetaData type )
            :
            type_( type ),
            deletethis_( true ) // ReadMatlabSettings deletes itself when finished
{
    settings.scriptname( filename.toStdString() );
}


void ReadMatlabSettings::
        readAsyncAndDeleteSelfWhenDone()
{
    if (!QFileInfo(settings.scriptname().c_str()).exists())
    {
        emit failed(settings.scriptname().c_str(), "File doesn't exist");

        if (deletethis_)
            delete this;

        return;
    }

    function_.reset( new MatlabFunction( settings.scriptname().c_str(), type_ == MetaData_Settings ? "settings" : "source", 4, type_ == MetaData_Settings ? 0 : &settings ) );
    QTimer::singleShot(100, this, SLOT(checkIfReady()));
}


Signal::pBuffer ReadMatlabSettings::
        sourceBuffer()
{
    return source_buffer_;
}


string ReadMatlabSettings::
        iconpath()
{
    return iconpath_;
}


void ReadMatlabSettings::
        checkIfReady()
{
    bool finished = function_->hasProcessCrashed() || function_->hasProcessEnded();

    string file = function_->isReady();
    if (file.empty() && !finished)
    {
        QTimer::singleShot(100, this, SLOT(checkIfReady()));
        return;
    }

    QByteArray ba = function_->getProcess()->readAllStandardOutput();
    QString s( ba );
    s = s.trimmed();
    if (s.isEmpty())
        TaskInfo("ReadMatlabSettings %s: no output", function_->matlabFunction().c_str());
    else
        TaskInfo("ReadMatlabSettings %s: output: %s", function_->matlabFunction().c_str(), s.toLatin1().data());

    bool success = false;

    if (!file.empty()) try
    {
        switch (type_)
        {
        case MetaData_Settings:
            readSettings(file);
            break;
        case MetaData_Source:
            readSource(file);
            break;
        default:
            BOOST_ASSERT( false );
            break;
        }
        success = true;
    }
    catch (const runtime_error& x)
    {
        TaskInfo("ReadMatlabSettings::%s %s", vartype(x).c_str(), x.what());
        s += "\n";
        s += vartype(x).c_str();
        s += ": ";
        s += x.what();
    }

    if (!success)
        emit failed( settings.scriptname().c_str(), s );

    if (deletethis_)
        delete this;
}


void ReadMatlabSettings::
        readSettings(string file)
{
    Hdf5Input h5(file);

    settings.arguments( h5.tryread<string>("arguments", settings.arguments()) );
    settings.chunksize( h5.tryread<double>("chunk_size", settings.chunksize() ));
    settings.computeInOrder( h5.tryread<double>("compute_chunks_in_order", settings.computeInOrder()));
    settings.operation = 0;
    settings.overlap( h5.tryread<double>("overlapping", settings.overlap()));
    iconpath_ = h5.tryread<string>("icon", "");
    settings.argument_description( h5.tryread<string>("argument_description", settings.argument_description()));
    bool is_source = 0.0 != h5.tryread<double>("is_source", 0.0);
    if (is_source)
        settings.setAsSource();

    settings.print("ReadMatlabSettings settings");

    emit settingsRead( settings );
}


void ReadMatlabSettings::
        readSource(string file)
{
    Hdf5Input h5(file);

    source_buffer_ = h5.tryread<Signal::pBuffer>("samples", Signal::pBuffer());
    if (source_buffer_)
        source_buffer_->sample_rate = h5.tryread<double>("samplerate", 1);
    else
        settings.argument_description( h5.tryread<string>("argument_description", settings.argument_description()));

    emit sourceRead();
}


} // namespace Adapters
