#ifndef ADAPTERS_MATLABOPERATION_H
#define ADAPTERS_MATLABOPERATION_H

#include "signal/operationcache.h"
#include "tools/support/plotlines.h"

// boost
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/split_member.hpp>

class QProcess;

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
class MatlabFunction: boost::noncopyable
{
public:
    /**
      Name of a matlab function and timeout measuerd in seconds.
      */
    MatlabFunction( std::string matlabFunction, float timeout, MatlabFunctionSettings* settings );
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
    void endProcess();

    std::string matlabFunction();
    std::string matlabFunctionFilename();
    float timeout();

private:
    // Not copyable
    MatlabFunction( const MatlabFunction& );
    MatlabFunction& operator=(const MatlabFunction&);

    //void kill();
	void abort();

    QProcess* _pid;
    std::string _dataFile;
    std::string _resultFile;
    std::string _matlab_function;
    std::string _matlab_filename;
    float _timeout;
};


class MatlabOperation: public Signal::OperationCache
{
public:
    MatlabOperation( Signal::pOperation source, MatlabFunctionSettings* settings );
    ~MatlabOperation();

    // Does only support mono, use first channel
    virtual unsigned num_channels() { return std::min(1u, Signal::OperationCache::num_channels()); }

    virtual std::string name();
    virtual Signal::pBuffer readRaw( const Signal::Interval& I );
    virtual void invalidate_samples(const Signal::Intervals& I);

    void restart();
    void settings(MatlabFunctionSettings*);
    MatlabFunctionSettings* settings() { return _settings; }

    /// Will call invalidate_samples if new data is available
    bool dataAvailable();

    bool isWaiting();

    boost::scoped_ptr<Tools::Support::PlotLines> plotlines;
protected:
    boost::scoped_ptr<MatlabFunction> _matlab;
    MatlabFunctionSettings* _settings;
    Signal::pBuffer ready_data;

private:
    friend class boost::serialization::access;
    MatlabOperation();
    template<class Archive> void save(Archive& ar, const unsigned int /*version*/) const {
        using boost::serialization::make_nvp;

        DefaultMatlabFunctionSettings settings;
        settings.scriptname_ =  _settings->scriptname();
        settings.redundant_ = _settings->redundant();
        settings.computeInOrder_ = _settings->computeInOrder();
        settings.chunksize_ = _settings->chunksize();
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(settings.scriptname_);
        ar & BOOST_SERIALIZATION_NVP(settings.chunksize_);
        ar & BOOST_SERIALIZATION_NVP(settings.computeInOrder_);
        ar & BOOST_SERIALIZATION_NVP(settings.redundant_);
    }
    template<class Archive> void load(Archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        DefaultMatlabFunctionSettings* settingsp = new DefaultMatlabFunctionSettings();
        DefaultMatlabFunctionSettings& settings = *settingsp;
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
        ar & BOOST_SERIALIZATION_NVP(settings.scriptname_);
        ar & BOOST_SERIALIZATION_NVP(settings.chunksize_);
        ar & BOOST_SERIALIZATION_NVP(settings.computeInOrder_);
        ar & BOOST_SERIALIZATION_NVP(settings.redundant_);
        settings.operation = this;

        this->settings(settingsp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

} // namespace Adapters

#endif // ADAPTERS_MATLABOPERATION_H
