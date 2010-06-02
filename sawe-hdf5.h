#ifndef SAWEHDF5_H
#define SAWEHDF5_H

#include "tfr-chunksink.h"
#include <H5Ipublic.h>
#include <H5Tpublic.h>
#include <TaskTimer.h>

//typedef int hid_t; // from H5Ipublic

namespace Sawe {

/**
  Throws std::runtime_error on errors.
  */
class Hdf5Output {
public:
    Hdf5Output(std::string filename);
    ~Hdf5Output();

    hid_t file_id() const { return _file_id; }

    void add( std::string name, const Signal::Buffer&);
    void add( std::string name, const Tfr::Chunk&);
    void add( std::string name, const double&);
    void add( std::string name, const std::string&);

private:
    hid_t _file_id;
    boost::scoped_ptr<TaskTimer> _timer;
};

/**
  Throws std::runtime_error on errors.
  */
class Hdf5Input {
public:
    Hdf5Input(std::string filename);
    ~Hdf5Input();

    hid_t file_id() const { return _file_id; }

    template<typename T> T read( std::string name )
    {
        std::string err;

        for (int i=0; i<2; i++)
            try
        {
            switch(i) {
                case 0: {T a = read_exact<T>( "/" + name + "/value" ); return a;}
                case 1: {T a = read_exact<T>( name ); return a;}
            }
        } catch (const std::runtime_error& x) {
            err = err + x.what() + "\n";
        }
        throw std::runtime_error(err.c_str());
    }

private:
    hid_t _file_id;
    boost::scoped_ptr<TaskTimer> _timer;

    template<typename T> T read_exact( std::string name );
    void findDataset(const std::string& name);
    std::vector<hsize_t> getInfo(const std::string& name, H5T_class_t* class_id=0);
};

template<> Signal::pBuffer  Hdf5Input::read_exact<Signal::pBuffer> ( std::string name );
template<> Tfr::pChunk      Hdf5Input::read_exact<Tfr::pChunk>     ( std::string name );
template<> double           Hdf5Input::read_exact<double>          ( std::string name );
template<> std::string      Hdf5Input::read_exact<std::string>     ( std::string name );

/**
  Transforms a pBuffer into a pChunk with CwtSingleton and saves the chunk in a file called
  sonicawe-x.csv, where x is a number between 1 and 9, or 0 if all the other 9 files already
  exists. The file is saved with the csv-format comma separated values, but values are
  actually separated by spaces. One row of the csv-file corresponds to one row of the chunk.
*/
class Hdf5Sink: public Tfr::ChunkSink
{
public:
    enum DataType {
        DataType_CHUNK,
        DataType_BUFFER
    };

    Hdf5Sink(std::string filename="sawe_chunk.h5", bool saveChunk=true);

    void    put( Signal::pBuffer , Signal::pSource );

    static void             saveBuffer( std::string filename, const Signal::Buffer& );
    static void             saveChunk( std::string filename, const Tfr::Chunk& );

    static Signal::pBuffer  loadBuffer( std::string filename );
    static Tfr::pChunk      loadChunk( std::string filename );

private:

    bool _saveChunk;
    std::string _filename;
};

} // namespace Sawe

#endif // SAWEHDF5_H
