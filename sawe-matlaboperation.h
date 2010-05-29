#ifndef SAWEMATLABOPERATION_H
#define SAWEMATLABOPERATION_H

#include "signal-operation.h"

namespace Sawe {

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
    MatlabFunction( std::string matlabFunction, float timeout=10 );
    ~MatlabFunction();

    /**
      'source' should be the filename of a data file containing data that
      octave and matlab can read with the command a=load('source');

      the return string contains the filename of a data file with the
      result of the function call.
      */
    std::string invokeAndWait( std::string source );
    std::string getTempName();
private:
    // Not copyable
    MatlabFunction( const MatlabFunction& );
    MatlabFunction& operator=(const MatlabFunction&);

    int _pid;
    std::string _dataFile;
    std::string _resultFile;
    float _timeout;
};

class MatlabOperation: public Signal::Operation
{
public:
    MatlabOperation( Signal::pSource source, std::string matlabFunction );

    virtual Signal::pBuffer read( unsigned firstSample, unsigned numberOfSamples );
protected:
    MatlabFunction _matlab;
};

} // namespace Sawe

#endif // SAWEMATLABOPERATION_H
