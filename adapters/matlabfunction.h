#ifndef MATLABFUNCTION_H
#define MATLABFUNCTION_H

// std
#include <string>

// boost
#include <boost/noncopyable.hpp>

// qt
#include <QProcess>

namespace Adapters {

class MatlabOperation;

class MatlabFunctionSettings
{
public:
    MatlabFunctionSettings():operation(0) {}
    virtual ~MatlabFunctionSettings() { operation = 0; }

    virtual int chunksize() = 0;
    virtual bool computeInOrder() = 0;
    virtual int redundant() = 0;
    virtual void redundant(int) = 0;
    virtual void setProcess(QProcess*) = 0;
    virtual std::string scriptname() = 0;
    virtual std::string arguments() = 0;

    MatlabOperation* operation;
};

class DefaultMatlabFunctionSettings: public MatlabFunctionSettings
{
public:
    DefaultMatlabFunctionSettings() : chunksize_(0),computeInOrder_(0),redundant_(0),pid_(0) {}

    int chunksize() { return chunksize_; }
    bool computeInOrder() { return computeInOrder_; }
    int redundant() { return redundant_; }
    void redundant(int v) { redundant_ = v; }
    void setProcess(QProcess* pid_) { pid_ = pid_; }
    std::string scriptname() { return scriptname_; }
    std::string arguments() { return arguments_; }

    int chunksize_;
    bool computeInOrder_;
    int redundant_;
    QProcess* pid_;
    std::string scriptname_;
    std::string arguments_;
};

/**
  Several files are used to cooperate with matlab functions:

  filewatcher.m is used to call the client specified 'matlabFunction'.
  dataFile will have the form of 'matlabFunction'.f68e7b8a.h5
           it is written by MatlabFunction and read by filewatcher.
           When a client calls invokeAndWait(source) the file 'source'
           is renamed to 'dataFile'.
  resultFile will have the form of 'matlabFunction'.f68e7b8a.h5.result.h5
           it is read by MatlabFunction and written by filewatcher.
           When a client calls invokeAndWait(source) the returned filename
           is 'resultFile'.
  source should preferably be unique, getTempName can be called if the
           client want a suggestion for a temp name. getTempName returns
           'dataFile'~.

  One instance of octave or matlab will be created for each instance of
  MatlabFunction. Each instance is then killed in each destructor.
  */
class MatlabFunction: public QObject, private boost::noncopyable
{
    Q_OBJECT
public:
    /**
      Name of a matlab function and timeout measuerd in seconds.
      */
    MatlabFunction( std::string matlabFunction, float timeout, MatlabFunctionSettings* settings );
    MatlabFunction( QString f, QString subname, float timeout, MatlabFunctionSettings* settings );
    ~MatlabFunction();

    std::string getTempName();

    /**
      'source' should be the filename of a data file containing data that
      octave and matlab can read with the command a=load('source');
      */
    void invoke( std::string source );
    bool isWaiting();

    /**
      Returns an empty string if not ready.
      */
    std::string isReady();

    /**
      the return string contains the filename of a data file with the
      result of the function call.
      */
    std::string waitForReady();

    bool hasProcessEnded();
    bool hasProcessCrashed();
    void endProcess();
    QProcess* getProcess();

    std::string matlabFunction();
    std::string matlabFunctionFilename();
    float timeout();

private slots:
    void finished ( int exitCode, QProcess::ExitStatus exitStatus );

private:
    void init(std::string path, MatlabFunctionSettings* settings);
    //void kill();
    void abort();

    QProcess* _pid;
    std::string _dataFile;
    std::string _resultFile;
    std::string _matlab_function;
    std::string _matlab_filename;
    bool _hasCrashed;

    float _timeout;
};

} // namespace Adapters

#endif // MATLABFUNCTION_H
