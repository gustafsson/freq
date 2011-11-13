#include "readmatlabsettings.h"

#include "hdf5.h"

#include "demangle.h"

#include <QTimer>
#include <QFileInfo>

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
    settings.scriptname_ = filename.toStdString();
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


std::string ReadMatlabSettings::
        iconpath()
{
    return iconpath_;
}


void ReadMatlabSettings::
        checkIfReady()
{
    bool finished = function_->hasProcessCrashed() || function_->hasProcessEnded();

    std::string file = function_->isReady();
    if (file.empty() && !finished)
    {
        QTimer::singleShot(100, this, SLOT(checkIfReady()));
        return;
    }

    QByteArray ba = function_->getProcess()->readAllStandardOutput();
    QString s( ba );
    TaskInfo("ReadMatlabSettings: output: %s", s.toLatin1().data());

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
    catch (const std::runtime_error& x)
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
        readSettings(std::string file)
{
    Hdf5Input h5(file);

    settings.arguments_ = h5.tryread<std::string>("arguments", settings.arguments_);
    settings.chunksize_ = h5.tryread<double>("chunk_size", settings.chunksize_);
    settings.computeInOrder_ = h5.tryread<double>("compute_chunks_in_order", settings.computeInOrder_);
    settings.operation = 0;
    settings.pid_ = 0;
    settings.redundant_ = h5.tryread<double>("overlapping", settings.computeInOrder_);
    iconpath_ = h5.tryread<std::string>("icon", "");

    TaskInfo ti("ReadMatlabSettings: settings");
    TaskInfo("arguments = %s", settings.arguments_.c_str());
    TaskInfo("chunksize = %d", settings.chunksize_);
    TaskInfo("computeInOrder = %d", settings.computeInOrder_);
    TaskInfo("redundant = %d", settings.redundant_);
    TaskInfo("scriptname = %s", settings.scriptname_.c_str());

    emit settingsRead( settings );
}


void ReadMatlabSettings::
        readSource(std::string file)
{
    Hdf5Input h5(file);

    source_buffer_ = h5.read<Signal::pBuffer>("data");
    if (source_buffer_)
        source_buffer_->sample_rate = h5.tryread<double>("samplerate", 1);

    emit sourceRead();
}


} // namespace Adapters
