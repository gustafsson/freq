#include "hdf5.h"

#include "tfr/cwt.h"
#include "tfr/cwtchunk.h"

#include <sstream>
#include <fstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <H5Fpublic.h>

#include "hdf5_hl.h"

//#define TIME_HDF5
#define TIME_HDF5 if(0)

//#define VERBOSE_HDF5
#define VERBOSE_HDF5 if(0)

using namespace std;

namespace Adapters
{


Hdf5Error::
        Hdf5Error(Type t, const std::string& message, const std::string& data)
            :
            std::runtime_error(message),
            t_(t),
            data_(data)
{
}


Hdf5Input::
        Hdf5Input(std::string filename)
{
    TIME_HDF5 _timer.reset(new TaskTimer("Reading HDF5-file '%s'", filename.c_str()));

    _file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (0>_file_id) throw Hdf5Error(Hdf5Error::Type_OpenFailed, "Could not open HDF5 file named '" + filename + "'", filename);
}


Hdf5Output::
        Hdf5Output(std::string filename)
{
    TIME_HDF5 _timer.reset(new TaskTimer("Writing HDF5-file '%s'", filename.c_str()));

    _file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (0>_file_id) throw Hdf5Error(Hdf5Error::Type_CreateFailed, "Could not create HDF5 file named '" + filename + "'", filename);
}


Hdf5Input::
        ~Hdf5Input()
{
    herr_t status = H5Fclose (_file_id);
    if (0>status)
        TaskTimer("Could not close HDF5 file (%d), got %d", _file_id, status).suppressTiming();;
}


Hdf5Output::
        ~Hdf5Output()
{
    herr_t status = H5Fclose (_file_id);
    if (0>status)
        TaskTimer("Could not close HDF5 file (%d), got %d", _file_id, status).suppressTiming();;
}


void Hdf5Input::
        findDataset(const std::string& name)
{
    std::vector<std::string> strs;
    boost::split(strs, name, boost::is_any_of("/"));
    BOOST_ASSERT( !strs.empty() );

    if (strs.front().empty())
        strs.erase( strs.begin() );
    BOOST_ASSERT( !strs.empty() );

    herr_t status = H5LTfind_dataset ( _file_id, strs[0].c_str() );
    if (1!=status) throw Hdf5Error(Hdf5Error::Type_MissingDataset, "Hdf5 file does not contain a dataset named '" + strs[0] + "'", strs[0]);
}


vector<hsize_t> Hdf5Input::
        getInfo(const std::string& name, H5T_class_t* class_id)
{
    findDataset(name);

    int RANK=0;
    herr_t status = H5LTget_dataset_ndims ( _file_id, name.c_str(), &RANK );
    if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "get_dataset_ndims failed");

    vector<hsize_t> dims(RANK);
    status = H5LTget_dataset_info ( _file_id, name.c_str(), &dims[0], class_id, 0 );
    if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "get_dataset_info failed");

    return dims;
}


template<>
void Hdf5Output::
        add( std::string name, const Signal::Buffer& cb)
{
    VERBOSE_HDF5 TaskTimer tt("Adding buffer '%s'", name.c_str());

    float* p = cb.waveform_data()->getCpuMemory();

    const unsigned RANK=2;
    hsize_t     dims[RANK]={cb.channels(), cb.number_of_samples()};

    herr_t      status = H5LTmake_dataset(_file_id,name.c_str(),RANK,dims,H5T_NATIVE_FLOAT,p);
    if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "Could not create and write a H5T_NATIVE_FLOAT type dataset named '" + name + "'", name);
}


template<>
Signal::pBuffer Hdf5Input::
        read_exact<Signal::pBuffer>( std::string name )
{
    VERBOSE_HDF5 TaskTimer tt("Reading buffer '%s'", name.c_str());

    herr_t      status;
    stringstream ss;

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);
    if (dims.size()==0)
        dims.push_back( 1 );
    if (dims.size()==1)
    {
        dims.push_back( dims[0] );
        dims[0] = 1;
    }
    if (dims.size()==2)
    {
        dims.push_back( dims[1] );
        dims[1] = dims[0];
        dims[0] = 1;
    }

    if (3!=dims.size()) throw Hdf5Error(Hdf5Error::Type_MissingDataset, ((stringstream&)(ss << (const char*)"Rank of '" << name << "' is '" << dims.size() << "' instead of 0, 1, 2 or 3.")).str(), name);

    if (0==class_id)
        return Signal::pBuffer();

    if (H5T_FLOAT!=class_id) throw Hdf5Error(Hdf5Error::Type_MissingDataset, ((stringstream&)(ss << "Class id for '" << name << "' is '" << class_id << "' instead of H5T_FLOAT.")).str(), name);

    Signal::pBuffer buffer( new Signal::Buffer(0, dims[2], 44100, dims[1], dims[0] ) );
    float* p = buffer->waveform_data()->getCpuMemory();

    status = H5LTread_dataset(_file_id, name.c_str(), H5T_NATIVE_FLOAT, p);
    if (0>status) throw Hdf5Error(Hdf5Error::Type_MissingDataset, "Could not read a H5T_NATIVE_FLOAT type dataset named '" + name + "'", name);

    return buffer;
}


template<>
void Hdf5Output::
        add( std::string name, const Tfr::Chunk& chunk)
{
    VERBOSE_HDF5 TaskTimer tt("Adding chunk '%s'", name.c_str());

    float2* p = chunk.transform_data->getCpuMemory();
    cudaExtent s = chunk.transform_data->getNumberOfElements();

    const unsigned RANK=2;
    hsize_t     dims[RANK]={s.height,s.width};

    herr_t      status;

    // By converting from float to double beforehand, execution time in octave dropped from 6 to 2 seconds.
    {
        GpuCpuData<double2> dbl(0, chunk.transform_data->getNumberOfElements());
        double2* dp = dbl.getCpuMemory();
        int datatype = -1;
        datatype = H5Tcreate( H5T_COMPOUND, 16 );
        H5Tinsert( datatype, "real", 0, H5T_NATIVE_DOUBLE );
        H5Tinsert( datatype, "imag", 8, H5T_NATIVE_DOUBLE );

        size_t N = s.height*s.width;

        for (unsigned n=0; n<N; n++)
        {
            dp[n].x = p[n].x;
            dp[n].y = p[n].y;
        }

        status = H5LTmake_dataset(_file_id,name.c_str(),RANK,dims,datatype,dp);
        if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "Could not create and write a H5T_COMPOUND type dataset named 'chunk'");

        status = H5Tclose(datatype);
        if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "Could not close HDF5 datatype");
    }
}


template<>
Tfr::pChunk Hdf5Input::
        read_exact<Tfr::pChunk>( std::string name )
{
    VERBOSE_HDF5 TaskTimer tt("Reading chunk '%s'", name.c_str());

    herr_t      status;
    stringstream ss;

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);

    if (2!=dims.size()) throw Hdf5Error(Hdf5Error::Type_MissingDataset, ((stringstream&)(ss << "Rank of '" << name << "' is '" << dims.size() << "' instead of 3.")).str(), name);

    Tfr::pChunk chunk( new Tfr::CwtChunk );
    chunk->chunk_offset = 0;
    chunk->sample_rate = 44100;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = dims[1];
    chunk->transform_data.reset( new GpuCpuData<float2>(0, make_cudaExtent( dims[1], dims[0], 1 )));
    chunk->freqAxis.setLogarithmic( 20, 22050, chunk->nScales() - 1 );

    float2* p = chunk->transform_data->getCpuMemory();

    if (H5T_COMPOUND==class_id)
    {
        GpuCpuData<double2> dbl(0, make_cudaExtent( dims[1], dims[0], 1 ));
        double2* dp = dbl.getCpuMemory();
        int datatype = -1;
        datatype = H5Tcreate( H5T_COMPOUND, 16 );
        H5Tinsert( datatype, "real", 0, H5T_NATIVE_DOUBLE );
        H5Tinsert( datatype, "imag", 8, H5T_NATIVE_DOUBLE );

        status = H5LTread_dataset(_file_id,name.c_str(),datatype,dp);
        if (0>status) throw Hdf5Error(Hdf5Error::Type_MissingDataset, "Could not read a H5T_COMPOUND type dataset named '" +name + "'", name);

        size_t N = dims[0]*dims[1];

        for (unsigned n=0; n<N; n++)
        {
            p[n].x = dp[n].x;
            p[n].y = dp[n].y;
        }

        status = H5Tclose(datatype);
        if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "Could not close HDF5 datatype");
    } else if (H5T_FLOAT==class_id){
        GpuCpuData<float1> dbl(0, make_cudaExtent( dims[1], dims[0], 1 ));
        float1* dp = dbl.getCpuMemory();

        status = H5LTread_dataset(_file_id,name.c_str(),H5T_NATIVE_FLOAT,dp);
        if (0>status) throw Hdf5Error(Hdf5Error::Type_MissingDataset, "Could not read a H5T_NATIVE_FLOAT type dataset named '" +name + "'", name);

        size_t N = dims[0]*dims[1];
        for (unsigned n=0; n<N; n++)
        {
            p[n].x = dp[n].x;
            p[n].y = 0;
        }
    } else {
        throw Hdf5Error(Hdf5Error::Type_MissingDataset, ((stringstream&)(ss << "Class id for '" << name << "' is '" << class_id << "' instead of H5T_COMPOUND.")).str(), name);
    }

    return chunk;
}


template<>
void Hdf5Output::
        add( std::string name, const double& v)
{
    VERBOSE_HDF5 TaskTimer tt("Adding double '%s'", name.c_str());

    hsize_t one[]={1};
    herr_t status = H5LTmake_dataset(_file_id,name.c_str(),1,one,H5T_NATIVE_DOUBLE,&v);
    if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "Could not create and write a double type dataset named '" + name + "'", name);
}


template<>
double Hdf5Input::
        read_exact<double>( std::string name )
{
    VERBOSE_HDF5 TaskTimer tt("Reading double '%s'", name.c_str());

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);

    BOOST_FOREACH( const hsize_t& t, dims)
            BOOST_ASSERT( t == 1 );

    double v;
    herr_t status = H5LTread_dataset(_file_id,name.c_str(),H5T_NATIVE_DOUBLE,&v);
    if (0>status) throw Hdf5Error(Hdf5Error::Type_MissingDataset, "Could not read a H5T_NATIVE_DOUBLE type dataset named '" + name + "'", name);

    return v;
}


template<>
void Hdf5Output::
        add( std::string name, const std::string& s)
{
    VERBOSE_HDF5 TaskTimer tt("Adding string '%s'", name.c_str());

    const char* p = s.c_str();

    const unsigned RANK=1;
    hsize_t     dims[RANK]={s.size()};

    herr_t status = H5LTmake_dataset(_file_id,name.c_str(),RANK,dims,H5T_C_S1,p);
    if (0>status) throw Hdf5Error(Hdf5Error::Type_HdfFailure, "Could not create and write a H5T_C_S1 type dataset named '" + name + "'", name);
}


template<>
std::string Hdf5Input::
        read_exact<std::string>( std::string name)
{
    VERBOSE_HDF5 TaskTimer tt("Reading string: %s", name.c_str());

    findDataset(name);

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);
    std::string v; v.reserve( dims[0]+1 );

    herr_t status = H5LTread_dataset(_file_id,name.c_str(),H5T_C_S1,&v[0]);
    if (0>status) throw Hdf5Error(Hdf5Error::Type_MissingDataset, "Could not read a H5T_C_S1 type dataset named '" + name + "'", name);

    return v;
}


static const char* dsetBuffer="buffer";
static const char* dsetChunk="chunk";
static const char* dsetOffset="offset";
static const char* dsetSamplerate="samplerate";
static const char* dsetRedundancy="redundancy";
static const char* dsetPlot="plot";

Hdf5Chunk::Hdf5Chunk( std::string filename)
:   _filename(filename) {}

Hdf5Buffer::Hdf5Buffer( std::string filename)
    :   _filename(filename) {}


void Hdf5Chunk::
        operator()( Tfr::Chunk& c )
{
    Tfr::Chunk* chunk;
    Tfr::pChunk pchunk;
    Tfr::CwtChunkPart* cwt = dynamic_cast<Tfr::CwtChunkPart*>(&c);

    if (cwt)
    {
        pchunk = cwt->cleanChunk();
        chunk = pchunk.get();
    }
    else
        chunk = &c;

    Hdf5Chunk::saveChunk(_filename, *chunk);
}


void Hdf5Buffer::
    put( Signal::pBuffer b)
{
    Hdf5Buffer::saveBuffer(_filename, *b, 0);
}


// TODO save and load all properties of chunks and buffers, not only raw data.
// The Hdf5 file is well suited for storing such data as well.
void Hdf5Buffer::
        saveBuffer( string filename, const Signal::Buffer& cb, double redundancy)
{
    Hdf5Output h5(filename);

    h5.add<Signal::Buffer>( dsetBuffer, cb );
    h5.add<double>( dsetOffset, cb.sample_offset );
    h5.add<double>( dsetSamplerate, cb.sample_rate );
    h5.add<double>( dsetRedundancy, redundancy );
}


void Hdf5Chunk::
        saveChunk( string filename, const Tfr::Chunk &chunk )
{
    Hdf5Output h5(filename);
    h5.add<Tfr::Chunk>( dsetChunk, chunk );
    h5.add<double>( dsetOffset, chunk.chunk_offset );
    h5.add<double>( dsetSamplerate, chunk.sample_rate );
}


Signal::pBuffer Hdf5Buffer::
        loadBuffer( string filename, double* redundancy, Signal::pBuffer* plot )
{
    Hdf5Input h5(filename);

    Signal::pBuffer b = h5.read<Signal::pBuffer>( dsetBuffer );
    if (b)
    {
        b->sample_offset = h5.read<double>( dsetOffset );
        b->sample_rate = h5.read<double>( dsetSamplerate );
    }
    try {
    *plot = h5.read<Signal::pBuffer>( dsetPlot );
    } catch (const std::runtime_error& ) {} // ok, never mind then
    *redundancy = h5.read<double>( dsetRedundancy );

    return b;
}


Tfr::pChunk Hdf5Chunk::
        loadChunk( string filename )
{
    Hdf5Input h5(filename);

    Tfr::pChunk c = h5.read<Tfr::pChunk>( dsetChunk );
    c->chunk_offset = h5.read<double>( dsetOffset );
    c->sample_rate = h5.read<double>( dsetSamplerate );

    return c;
}

} // namespace Adapters
