#ifndef ADAPTERS_MATLABOPERATION_H
#define ADAPTERS_MATLABOPERATION_H

#include "signal/operationcache.h"

namespace Adapters {

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
class MatlabFunction
{
public:
    /**
      Name of a matlab function and timeout measuerd in seconds.
      */
    MatlabFunction( std::string matlabFunction, float timeout );
    ~MatlabFunction();

    /**
      'source' should be the filename of a data file containing data that
      octave and matlab can read with the command a=load('source');

      the return string contains the filename of a data file with the
      result of the function call.
      */
    std::string invokeAndWait( std::string source );
    std::string getTempName();

    std::string matlabFunction();
    float timeout();

private:
    // Not copyable
    MatlabFunction( const MatlabFunction& );
    MatlabFunction& operator=(const MatlabFunction&);

	void kill();
	void abort();

    void* _pid;
    std::string _dataFile;
    std::string _resultFile;
    std::string _matlab_function;
    float _timeout;
};

class MatlabOperation: public Signal::OperationCache
{
public:
    MatlabOperation( Signal::pOperation source, std::string matlabFunction );

    virtual Signal::pBuffer readRaw( const Signal::Interval& I );

    void restart();

protected:
    boost::scoped_ptr<MatlabFunction> _matlab;
};

} // namespace Adapters

#endif // ADAPTERS_MATLABOPERATION_H
