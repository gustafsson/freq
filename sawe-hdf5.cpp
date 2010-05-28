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

void Hdf5::
put( Signal::pBuffer b )
{
    TaskTimer tt("Saving HDF5-file");

	hid_t       file_id;
	herr_t      status;

    Tfr::pChunk chunk = Tfr::CwtSingleton::operate( b );

    float2* p = chunk->transform_data->getCpuMemory();
    cudaExtent s = chunk->transform_data->getNumberOfElements();

    hsize_t     dims[RANK]={s.height,s.width,2};

    /* create a HDF5 file */
    file_id = H5Fcreate("sawe_chunk.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* create and write an integer type dataset named "dset" */
    status = H5LTmake_dataset(file_id,"/dset",RANK,dims,H5T_NATIVE_FLOAT,p);

    /* close file */
    status = H5Fclose (file_id);
}

} // namespace Sawe
