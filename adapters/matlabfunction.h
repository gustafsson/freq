#ifndef MATLABFUNCTION_H
#define MATLABFUNCTION_H

// std
#include <string>

// boost
#include <boost/noncopyable.hpp>

// qt
#include <QProcess>
#include <QScopedPointer>

namespace Adapters {

class MatlabOperation;

class MatlabFunctionSettings
{
public:
    MatlabFunctionSettings():operation(0) {}
    MatlabFunctionSettings& operator=(const MatlabFunctionSettings& b);
    virtual ~MatlabFunctionSettings() { operation = 0; }

    virtual int chunksize() const = 0;
    virtual void chunksize(int) = 0;
    virtual bool computeInOrder() const = 0;
    virtual void computeInOrder(bool) = 0;
    virtual int overlap() const = 0;
    virtual void overlap(int) = 0;
    virtual std::string scriptname() const = 0;
    virtual void scriptname(const std::string&) = 0;
    virtual std::string arguments() const = 0;
    virtual void arguments(const std::string&) = 0;
    virtual std::string argument_description() const = 0;
    virtual void argument_description(const std::string&) = 0;

    virtual void setProcess(QProcess*) = 0;
    MatlabOperation* operation;

    bool isTerminal();
    bool isSource();
    void setAsSource();
    void print(const char*str);
};

class DefaultMatlabFunctionSettings: public MatlabFunctionSettings
{
public:
    DefaultMatlabFunctionSettings();
    DefaultMatlabFunctionSettings(const MatlabFunctionSettings& b);
    DefaultMatlabFunctionSettings& operator=(const MatlabFunctionSettings& b);

    int chunksize() const { return chunksize_; }
    void chunksize(int v) { chunksize_ = v; }
    bool computeInOrder() const { return computeInOrder_; }
    void computeInOrder(bool v) { computeInOrder_ = v; }
    int overlap() const { return redundant_; }
    void overlap(int v) { redundant_ = v; }
    std::string scriptname() const { return scriptname_; }
    void scriptname(const std::string& v) { scriptname_ = v; }
    std::string arguments() const { return arguments_; }
    void arguments(const std::string& v) { arguments_ = v; }
    std::string argument_description() const { return argument_description_; }
    void argument_description(const std::string& v) { argument_description_ = v; }

    void setProcess(QProcess*);

private:
    friend class MatlabOperation;

    int chunksize_;
    bool computeInOrder_;
    int redundant_;
    std::string scriptname_;
    std::string arguments_;
    std::string argument_description_;
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
    MatlabFunction( QString f, QString subname, float timeout, MatlabFunctionSettings* settings, bool justtest );
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
    void init(std::string path, MatlabFunctionSettings* settings, bool justtest = false, bool sendoutput = true);
    //void kill();
    void abort();

    QProcess* _pid;
    std::string _dataFile;
    std::string _resultFile;
    std::string _matlab_function;
    std::string _matlab_filename;
    bool _hasCrashed;

    float _timeout;
    QString _interopName;
};

} // namespace Adapters

#endif // MATLABFUNCTION_H
