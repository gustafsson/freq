#include "sawe-hdf5.h"
#include <sstream>
#include <fstream>
#include "tfr-cwt.h"
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std;

namespace Sawe
{


Hdf5Input::
        Hdf5Input(std::string filename)
{
    _timer.reset(new TaskTimer("Reading HDF5-file '%s'", filename.c_str()));

    _file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (0>_file_id) throw runtime_error("Could not open HDF5 file named '" + filename + "'");
}

Hdf5Output::
        Hdf5Output(std::string filename)
{
    _timer.reset(new TaskTimer("Writing HDF5-file '%s'", filename.c_str()));

    _file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (0>_file_id) throw runtime_error("Could not create HDF5 file named '" + filename + "'");
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
    if (1!=status) throw runtime_error("Hdf5 file does not contain a dataset named '" + strs[0] + "'");
}

vector<hsize_t> Hdf5Input::
        getInfo(const std::string& name, H5T_class_t* class_id)
{
    findDataset(name);

    int RANK=0;
    herr_t status = H5LTget_dataset_ndims ( _file_id, name.c_str(), &RANK );
    if (0>status) throw runtime_error("get_dataset_ndims failed");

    vector<hsize_t> dims(RANK);
    status = H5LTget_dataset_info ( _file_id, name.c_str(), dims.data(), class_id, 0 );
    if (0>status) throw runtime_error("get_dataset_info failed");

    return dims;
}

template<>
void Hdf5Output::
        add( std::string name, const Signal::Buffer& cb)
{
    TaskTimer tt("Adding buffer '%s'", name.c_str());

    Signal::pBuffer data;
    const Signal::Buffer* b = &cb;
    if (b->interleaved()==Signal::Buffer::Interleaved_Complex) {
        data = b->getInterleaved(Signal::Buffer::Only_Real);
        b = &*data;
    }

    float* p = b->waveform_data->getCpuMemory();
    cudaExtent s = b->waveform_data->getNumberOfElements();

    const unsigned RANK=1;
    hsize_t     dims[RANK]={s.width};

    herr_t      status = H5LTmake_dataset(_file_id,name.c_str(),RANK,dims,H5T_NATIVE_FLOAT,p);
    if (0>status) throw runtime_error("Could not create and write a H5T_NATIVE_FLOAT type dataset named '" + name + "'");
}

template<>
Signal::pBuffer Hdf5Input::
        read_exact<Signal::pBuffer>( std::string name )
{
    TaskTimer tt("Reading buffer '%s'", name.c_str());

    herr_t      status;
    stringstream ss;

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);

    if (1!=dims.size() && 2!=dims.size()) throw runtime_error(((stringstream&)(ss << (const char*)"Rank of '" << name << "' is '" << dims.size() << "' instead of 1 or 2.")).str());

    if (H5T_FLOAT!=class_id) throw runtime_error(((stringstream&)(ss << "Class id for '" << name << "' is '" << class_id << "' instead of H5T_FLOAT.")).str());

    Signal::pBuffer buffer( new Signal::Buffer(0, dims[0], 44100 ) );
    float* p = buffer->waveform_data->getCpuMemory();

    status = H5LTread_dataset(_file_id, name.c_str(), H5T_NATIVE_FLOAT, p);
    if (0>status) throw runtime_error("Could not read a H5T_NATIVE_FLOAT type dataset named '" + name + "'");

    return buffer;
}

template<>
void Hdf5Output::
        add( std::string name, const Tfr::Chunk& chunk)
{
    TaskTimer tt("Adding chunk '%s'", name.c_str());

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
        if (0>status) throw runtime_error("Could not create and write a H5T_COMPOUND type dataset named 'chunk'");

        status = H5Tclose(datatype);
        if (0>status) throw runtime_error("Could not close HDF5 datatype");
    }
}

template<>
Tfr::pChunk Hdf5Input::
        read_exact<Tfr::pChunk>( std::string name)
{
    TaskTimer tt("Reading chunk '%s'", name.c_str());

    herr_t      status;
    stringstream ss;

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);

    if (2!=dims.size()) throw runtime_error(((stringstream&)(ss << "Rank of '" << name << "' is '" << dims.size() << "' instead of 3.")).str());

    Tfr::pChunk chunk( new Tfr::Chunk);
    chunk->min_hz = 20;
    chunk->max_hz = 22050;
    chunk->chunk_offset = 0;
    chunk->sample_rate = 44100;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = dims[1];
    chunk->transform_data.reset( new GpuCpuData<float2>(0, make_cudaExtent( dims[1], dims[0], 1 )));
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
        if (0>status) throw runtime_error("Could not read a H5T_COMPOUND type dataset named '" +name + "'");

        size_t N = dims[0]*dims[1];

        for (unsigned n=0; n<N; n++)
        {
            p[n].x = dp[n].x;
            p[n].y = dp[n].y;
        }

        status = H5Tclose(datatype);
        if (0>status) throw runtime_error("Could not close HDF5 datatype");
    } else if (H5T_FLOAT==class_id){
        GpuCpuData<float1> dbl(0, make_cudaExtent( dims[1], dims[0], 1 ));
        float1* dp = dbl.getCpuMemory();

        status = H5LTread_dataset(_file_id,name.c_str(),H5T_NATIVE_FLOAT,dp);
        if (0>status) throw runtime_error("Could not read a H5T_NATIVE_FLOAT type dataset named '" +name + "'");

        size_t N = dims[0]*dims[1];
        for (unsigned n=0; n<N; n++)
        {
            p[n].x = dp[n].x;
            p[n].y = 0;
        }
    } else {
        throw runtime_error(((stringstream&)(ss << "Class id for '" << name << "' is '" << class_id << "' instead of H5T_COMPOUND.")).str());
    }

    return chunk;
}

template<>
void Hdf5Output::
        add( std::string name, const double& v)
{
    TaskTimer tt("Adding double '%s'", name.c_str());

    hsize_t one[]={1};
    herr_t status = H5LTmake_dataset(_file_id,name.c_str(),1,one,H5T_NATIVE_DOUBLE,&v);
    if (0>status) throw runtime_error("Could not create and write a double type dataset named '" + name + "'");
}

template<>
double Hdf5Input::
        read_exact<double>( std::string name )
{
    TaskTimer tt("Reading double '%s'", name.c_str());

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);

    BOOST_FOREACH( const hsize_t& t, dims)
            BOOST_ASSERT( t == 1 );

    double v;
    herr_t status = H5LTread_dataset(_file_id,name.c_str(),H5T_NATIVE_DOUBLE,&v);
    if (0>status) throw runtime_error("Could not read a H5T_NATIVE_DOUBLE type dataset named '" + name + "'");

    return v;
}

template<>
void Hdf5Output::
        add( std::string name, const std::string& s)
{
    TaskTimer tt("Adding string '%s'", name.c_str());

    const char* p = s.c_str();

    const unsigned RANK=1;
    hsize_t     dims[RANK]={s.size()};

    herr_t status = H5LTmake_dataset(_file_id,name.c_str(),RANK,dims,H5T_C_S1,p);
    if (0>status) throw runtime_error("Could not create and write a H5T_C_S1 type dataset named '" + name + "'");
}

template<>
std::string Hdf5Input::
        read_exact<std::string>( std::string name)
{
    TaskTimer tt("Reading string: %s", name.c_str());

    findDataset(name);

    H5T_class_t class_id=H5T_NO_CLASS;
    vector<hsize_t> dims = getInfo(name, &class_id);
    std::string v; v.reserve( dims[0]+1 );

    herr_t status = H5LTread_dataset(_file_id,name.c_str(),H5T_C_S1,&v[0]);
    if (0>status) throw runtime_error("Could not read a H5T_C_S1 type dataset named '" + name + "'");

    return v;
}

static const char* dsetBuffer="buffer";
static const char* dsetChunk="chunk";
static const char* dsetOffset="offset";
static const char* dsetSamplerate="samplerate";

Hdf5Sink::
        Hdf5Sink( std::string filename, bool saveChunk)
:   _saveChunk(saveChunk),
    _filename(filename)
{
}

void Hdf5Sink::
        put( Signal::pBuffer b, Signal::pSource src )
{
    if (_saveChunk) {
        Tfr::pChunk chunk = getChunk( b, src );
        chunk = cleanChunk(chunk);

        if (chunk->n_valid_samples != b->number_of_samples()) {
            TaskTimer("!!! Warning: requested %u sampels but chunk contains %u samples",
                      b->number_of_samples(), chunk->n_valid_samples).suppressTiming();
        }

        Hdf5Sink::saveChunk(_filename, *chunk);
    } else {
        Hdf5Sink::saveBuffer(_filename, *b);
    }
}

// TODO save and load all properties of chunks and buffers, not only raw data.
// The Hdf5 file is well suited for storing such data as well.
void Hdf5Sink::
        saveBuffer( string filename, const Signal::Buffer& cb)
{
    Hdf5Output h5(filename);

    h5.add<Signal::Buffer>( dsetBuffer, cb );
    h5.add<double>( dsetOffset, cb.sample_offset );
    h5.add<double>( dsetSamplerate, cb.sample_rate );
}

void Hdf5Sink::
        saveChunk( string filename, const Tfr::Chunk &chunk )
{
    Hdf5Output h5(filename);
    h5.add<Tfr::Chunk>( dsetChunk, chunk );
    h5.add<double>( dsetOffset, chunk.chunk_offset );
    h5.add<double>( dsetSamplerate, chunk.sample_rate );
}

Signal::pBuffer Hdf5Sink::
        loadBuffer( string filename )
{
    Hdf5Input h5(filename);

    Signal::pBuffer b = h5.read<Signal::pBuffer>( dsetBuffer );
    b->sample_offset = h5.read<double>( dsetOffset );
    b->sample_rate = h5.read<double>( dsetSamplerate );

    return b;
}

Tfr::pChunk Hdf5Sink::
        loadChunk( string filename )
{
    Hdf5Input h5(filename);

    Tfr::pChunk c = h5.read<Tfr::pChunk>( dsetChunk );
    c->chunk_offset = h5.read<double>( dsetOffset );
    c->sample_rate = h5.read<double>( dsetSamplerate );

    return c;
}

} // namespace Sawe
