#include "sawe-hdf5.h"
#include <sstream>
#include <fstream>
#include "tfr-cwt.h"
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std;

namespace Sawe
{

static const char* dsetBuffer="buffer";
static const char* dsetChunk="chunk";
static const char* dsetOffset="offset";
static const char* dsetSamplerate="samplerate";

Hdf5::
        Hdf5( std::string filename, bool saveChunk)
:   _saveChunk(saveChunk),
    _filename(filename)
{
}

void Hdf5::
        put( Signal::pBuffer b, Signal::pSource src )
{
    if (_saveChunk) {
        Tfr::pChunk chunk = getChunk( b, src );
        chunk = cleanChunk(chunk);

        Hdf5::saveChunk(_filename, *chunk);
    } else {
        Hdf5::saveBuffer(_filename, *b);
    }
}

// TODO save and load all properties of chunks and buffers, not only raw data.
// The Hdf5 file is well suited for storing such data as well.
void Hdf5::
        saveBuffer( string filename, const Signal::Buffer& cb)
{
    TaskTimer tt("Saving buffer in HDF5-file %s", filename.c_str());

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

    hid_t       file_id;
    herr_t      status;

    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (0>file_id) throw runtime_error("Could not create HDF5 file named '" + filename + "'");

    status = H5LTmake_dataset(file_id,dsetBuffer,RANK,dims,H5T_NATIVE_FLOAT,p);
    if (0>status) throw runtime_error("Could not create and write a float type dataset named '" + string(dsetBuffer) + "'");

    hsize_t one[]={1};
    double offs = b->sample_offset;
    status = H5LTmake_dataset(file_id,dsetOffset,1,one,H5T_NATIVE_DOUBLE,&offs);
    if (0>status) throw runtime_error("Could not create and write a hsize type dataset named '" + string(dsetOffset) + "'");

    double rate = b->sample_rate;
    status = H5LTmake_dataset(file_id,dsetSamplerate,1,one,H5T_NATIVE_DOUBLE,&rate);
    if (0>status) throw runtime_error("Could not create and write a hsize type dataset named '" + string(dsetSamplerate) + "'");

    status = H5Fclose (file_id);
    if (0>status) throw runtime_error("Could not close HDF5 file");
}

void Hdf5::
        saveChunk( string filename, const Tfr::Chunk &chunk )
{
    TaskTimer tt("Saving chunk in HDF5-file %s", filename.c_str());

    float2* p = chunk.transform_data->getCpuMemory();
    cudaExtent s = chunk.transform_data->getNumberOfElements();

    const unsigned RANK=2;
    hsize_t     dims[RANK]={s.height,s.width};

    hid_t       file_id;
    herr_t      status;

    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (0>file_id) throw runtime_error("Could not create HDF5 file named '" + filename + "'");

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

        status = H5LTmake_dataset(file_id,dsetChunk,RANK,dims,datatype,dp);
        if (0>status) throw runtime_error("Could not create and write a float type dataset named 'chunk'");

        status = H5Tclose(datatype);
        if (0>status) throw runtime_error("Could not close HDF5 datatype");
    }

    hsize_t one[]={1};
    double offs = chunk.chunk_offset;
    status = H5LTmake_dataset(file_id,dsetOffset,1,one,H5T_NATIVE_DOUBLE,&offs);
    if (0>status) throw runtime_error("Could not create and write a hsize type dataset named '" + string(dsetOffset) + "'");

    double rate = chunk.sample_rate;
    status = H5LTmake_dataset(file_id,dsetSamplerate,1,one,H5T_NATIVE_DOUBLE,&rate);
    if (0>status) throw runtime_error("Could not create and write a hsize type dataset named '" + string(dsetSamplerate) + "'");

    status = H5Fclose (file_id);
    if (0>status) throw runtime_error("Could not close HDF5 file");
}

Signal::pBuffer Hdf5::
        loadBuffer( string filename )
{
    TaskTimer tt("Load HDF5 buffer: %s", filename.c_str());

    string err;

    for (int i=0; i<2; i++) try
    {
        string sdset = dsetBuffer;
        string soffdset = dsetOffset;
        string sfsdset = dsetSamplerate;
        switch(i) {
        case 0: sdset = "/" + sdset + "/value";
                soffdset = "/" + soffdset + "/value";
                sfsdset = "/" + sfsdset + "/value"; break;
        case 1: break;
        }
        const char*dset = sdset.c_str();
        const char*odset = soffdset.c_str();
        const char*fsdset = sfsdset.c_str();

        hid_t       file_id;
        herr_t      status;
        stringstream ss;

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (0>file_id) throw runtime_error("Could not open HDF5 file named '" + filename + "'");

        status = H5LTfind_dataset ( file_id, dsetBuffer );
        if (1!=status) throw runtime_error("'" + filename + "' does not contain a dataset named '" + dsetBuffer + "'");

        int RANK=0;
        status = H5LTget_dataset_ndims ( file_id, dset, &RANK );
        if (0>status) throw runtime_error("get_dataset_ndims failed");
        if (1!=RANK && 2!=RANK) throw runtime_error(((stringstream&)(ss << (const char*)"Rank of '" << dset << "' is '" << RANK << "' instead of 1 or 2.")).str());

        H5T_class_t class_id=H5T_NO_CLASS;
        vector<hsize_t> dims(RANK);
        status = H5LTget_dataset_info ( file_id, dset, dims.data(), &class_id, 0 );
        if (0>status) throw runtime_error("get_dataset_info failed");
        if (H5T_FLOAT!=class_id) throw runtime_error(((stringstream&)(ss << "Class id for '" << dset << "' is '" << class_id << "' instead of H5T_FLOAT.")).str());

        Signal::pBuffer buffer( new Signal::Buffer(0, dims[0], 44100 ) );
        float* p = buffer->waveform_data->getCpuMemory();

        status = H5LTread_dataset(file_id, dset, H5T_NATIVE_FLOAT, p);
        if (0>status) throw runtime_error("Could not read a float type dataset named '" + sdset + "'");

        double offs=0;
        status = H5LTread_dataset(file_id,odset,H5T_NATIVE_DOUBLE,&offs);
        if (0>status) throw runtime_error("Could not read a hsize type dataset named '" + soffdset + "'");
        buffer->sample_offset=offs;

        double samplerate=0;
        status = H5LTread_dataset(file_id,fsdset,H5T_NATIVE_DOUBLE,&samplerate);
        if (0>status) throw runtime_error("Could not read a hsize type dataset named '" + sfsdset + "'");
        buffer->sample_rate=samplerate;

        status = H5Fclose (file_id);
        if (0>status) throw runtime_error("Could not close HDF5 file");

        return buffer;
    } catch (const std::runtime_error& x) {
        err = err + x.what() + "\n";
    }
    throw std::runtime_error(err.c_str());
}

Tfr::pChunk Hdf5::
        loadChunk( string filename )
{
    TaskTimer tt("Load HDF5 chunk: %s", filename.c_str());

    string err;

    for (int i=0; i<2; i++) try
    {
        string sdset = dsetChunk;
        string soffdset = dsetOffset;
        string sfsdset = dsetSamplerate;
        switch(i) {
        case 0: sdset = "/" + sdset + "/value";
                soffdset = "/" + soffdset + "/value";
                sfsdset = "/" + sfsdset + "/value"; break;
        case 1: break;
        }
        const char*dset = sdset.c_str();
        const char*odset = soffdset.c_str();
        const char*fsdset = sfsdset.c_str();

        hid_t       file_id;
        herr_t      status;
        stringstream ss;

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (0>file_id) throw runtime_error("Could not open HDF5 file named '" + filename + "'");

        status = H5LTfind_dataset ( file_id, dsetChunk );
        if (1!=status) throw runtime_error("'" + filename + "' does not contain a dataset named '" + dsetChunk + "'");

        int RANK=0;
        status = H5LTget_dataset_ndims ( file_id, dset, &RANK );
        if (0>status) throw runtime_error("get_dataset_ndims failed");
        if (2!=RANK) throw runtime_error(((stringstream&)(ss << "Rank of '" << dsetChunk << "' is '" << RANK << "' instead of 3.")).str());

        H5T_class_t class_id=H5T_NO_CLASS;
        vector<hsize_t> dims(RANK);
        status = H5LTget_dataset_info ( file_id,dset, dims.data(), &class_id, 0 );
        if (0>status) throw runtime_error("get_dataset_info failed");

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

            status = H5LTread_dataset(file_id,dset,datatype,dp);
            if (0>status) throw runtime_error("Could not read a compound type dataset named '" +sdset + "'");

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

            status = H5LTread_dataset(file_id,dset,H5T_NATIVE_FLOAT,dp);
            if (0>status) throw runtime_error("Could not read a compound type dataset named '" +sdset + "'");

            size_t N = dims[0]*dims[1];
            for (unsigned n=0; n<N; n++)
            {
                p[n].x = dp[n].x;
                p[n].y = 0;
            }
        } else {
            throw runtime_error(((stringstream&)(ss << "Class id for '" << dsetBuffer << "' is '" << class_id << "' instead of H5T_COMPOUND.")).str());
        }

        double offs=0;
        status = H5LTread_dataset(file_id,odset,H5T_NATIVE_DOUBLE,&offs);
        if (0>status) throw runtime_error("Could not read a hsize type dataset named '" + soffdset + "'");
        chunk->chunk_offset=offs;

        double samplerate=0;
        status = H5LTread_dataset(file_id,fsdset,H5T_NATIVE_DOUBLE,&samplerate);
        if (0>status) throw runtime_error("Could not read a hsize type dataset named '" + sfsdset + "'");
        chunk->sample_rate=samplerate;

        status = H5Fclose (file_id);
        if (0>status) throw runtime_error("Could not close HDF5 file");

        return chunk;
    } catch (const std::runtime_error& x) {
        err = err + x.what() + "\n";
    }
    throw std::runtime_error(err.c_str());
}

} // namespace Sawe
