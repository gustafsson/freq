#ifndef ADAPTERS_HDF5_H
#define ADAPTERS_HDF5_H

// Define _HDF5USEDLL_ to tell HDF5 to use dynamic library linking
#define _HDF5USEDLL_

#include "signal/sink.h"
#include "tfr/cwtfilter.h"
#include <H5Ipublic.h>
#include <H5Tpublic.h>
#include <TaskTimer.h>

//typedef int hid_t; // from H5Ipublic

//#define VERBOSE_HDF5_HEADER
#define VERBOSE_HDF5_HEADER if(0)

namespace Adapters {

class Hdf5Error: public std::runtime_error
{
public:
    enum Type
    {
        Type_OpenFailed,
        Type_CreateFailed,
        Type_MissingDataset,
        Type_HdfFailure
    };

    Hdf5Error(Type t, const std::string& message, const std::string& data="");
    virtual
    ~Hdf5Error() throw();

    Type type() const { return t_; }
    const std::string& data() const { return data_; }
private:
    Type t_;
    std::string data_;
};


/**
  Throws Hdf5Error on errors.
  */
class Hdf5Output {
public:
    Hdf5Output(std::string filename);
    ~Hdf5Output();

    hid_t file_id() const { return _file_id; }

    template<typename T> void add( std::string name, const T& );

private:
    hid_t _file_id;
    boost::scoped_ptr<TaskTimer> _timer;
};


template<> void Hdf5Output::add( std::string name, const Signal::Buffer&);
template<> void Hdf5Output::add( std::string name, const Tfr::Chunk&);
template<> void Hdf5Output::add( std::string name, const double&);
template<> void Hdf5Output::add( std::string name, const std::string&);

/**
  Throws Hdf5Error on errors.
  */
class Hdf5Input {
public:
    Hdf5Input(std::string filename);
    ~Hdf5Input();

    hid_t file_id() const { return _file_id; }

    /**
      Will first try reading a dataset named as octave would name it: '/datasetname/value'
      And then try to read a dataset as Hdf5Output would name it: 'datasetname'

      read_exact is supposed to have template specializations for each supported type 'T'.
      It is a link-time error to read a type that doesn't have a template specialization.
      */
    template<typename T> T read( std::string datasetname )
    {
        std::string err;
        static bool looks_like_matlab = false;

        bool looked_like_matlab = looks_like_matlab;

        for (int i=0; i<2; i++) try
        {
            switch(i ^ (int)looked_like_matlab) {
                // Octave style
                case 0: return read_exact<T>( "/" + datasetname + "/value" );

                // Matlab style
                case 1: return read_exact<T>( datasetname );
            }
        } catch (const std::runtime_error& x) {
            looks_like_matlab = !looks_like_matlab;
            if (err.empty() || strcmp(x.what(), err.substr(0, err.size()-1).c_str()) != 0)
                err = err + x.what() + "\n";
        }
        throw std::runtime_error(err.c_str());
    }

    template<typename T> T tryread( std::string datasetname, T defaultValue )
    {
        try {
            return read<T>(datasetname);
        } catch (const std::runtime_error& x) {
            VERBOSE_HDF5_HEADER TaskInfo ti("Hdf5Input couldn't read '%s' from '%s'\n%s", datasetname.c_str(), _filename.c_str(), x.what());
            VERBOSE_HDF5_HEADER TaskInfo("Tip: Use tools like \"h5ls -fr\" to investigate the dataset.");
            return defaultValue;
        }
    }

private:
    hid_t _file_id;
    boost::scoped_ptr<TaskTimer> _timer;
    std::string _filename;

    /**
      Reads a dataset named 'datasetname'. To be called through 'read'.
      @see read
      */
    template<typename T> T read_exact( std::string datasetname );

    void findDataset(const std::string& name);
    std::vector<hsize_t> getInfo(const std::string& name, H5T_class_t* class_id=0);
};

template<> Signal::pBuffer  Hdf5Input::read_exact<Signal::pBuffer> ( std::string datasetname );
template<> Tfr::pChunk      Hdf5Input::read_exact<Tfr::pChunk>     ( std::string datasetname );
template<> double           Hdf5Input::read_exact<double>          ( std::string datasetname );
template<> std::string      Hdf5Input::read_exact<std::string>     ( std::string datasetname );

/**
  Transforms a pBuffer into a pChunk with Cwt and saves the chunk in a file called
  sonicawe-x.csv, where x is a number between 1 and 9, or 0 if all the other 9 files already
  exists. The file is saved with the csv-format comma separated values, but values are
  actually separated by spaces. One row of the csv-file corresponds to one row of the chunk.
*/
class Hdf5Chunk: public Tfr::CwtFilter
{
public:
    Hdf5Chunk(std::string filename="sawe_chunk.h5");

    virtual void operator()( Tfr::Chunk& );

    static void             saveChunk( std::string filename, const Tfr::Chunk& );
    static Tfr::pChunk      loadChunk( std::string filename );

private:
    std::string _filename;
};

class Hdf5Buffer: public Signal::Sink
{
public:
    Hdf5Buffer(std::string filename="sawe_buffer.h5");

    virtual void put(Signal::pBuffer);

    static void             saveBuffer( std::string filename, const Signal::Buffer&, double overlapping );
    static Signal::pBuffer  loadBuffer( std::string filename, double* overlapping, Signal::pBuffer* plot );

private:
    std::string _filename;
};

} // namespace Adapters

#endif // ADAPTERS_HDF5_H
