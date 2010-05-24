#include "sawe-hdf5.h"
#include <sstream>
#include <fstream>
#include "tfr-cwt.h"

#include "hdf5.h"
#include "hdf5_hl.h"

using namespace std;

#define RANK 3

namespace Sawe
{

string hdf5_number()
{
    string basename = "sonicawe-";
    for (unsigned c = 1; c<10; c++)
    {
        stringstream filename;
        filename << basename << c << ".hdf50";
        fstream hdf5(filename.str().c_str());
        if (!hdf5.is_open())
            return filename.str();
    }
    return basename+"0.hdf5";
}

void Hdf5::
put( Signal::pBuffer b )
{
    string filename = hdf5_number();
    TaskTimer tt("Saving HDF5-file %s", filename.c_str());

	hid_t       file_id;
	herr_t      status;

    Tfr::pChunk chunk = Tfr::CwtSingleton::operate( b );

    float2* p = chunk->transform_data->getCpuMemory();
    cudaExtent s = chunk->transform_data->getNumberOfElements();

	hsize_t     dims[RANK]={2,s.width,s.height};

	/* create a HDF5 file */
	file_id = H5Fcreate ("ex_lite1.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	/* create and write an integer type dataset named "dset" */
	status = H5LTmake_dataset(file_id,"/dset",RANK,dims,H5T_NATIVE_FLOAT,p);

	/* close file */
	status = H5Fclose (file_id);

}

} // namespace Sawe
