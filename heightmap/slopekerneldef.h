#ifndef SLOPEKERNELDEF_H
#define SLOPEKERNELDEF_H

#include "resample.h"
#include "slopekernel.h"

class SlopeFetcher
{
public:
    typedef std::complex<float> T;

    SlopeFetcher( float xscale, float yscale, DataPos size )
        :   xscale( xscale ),
            yscale( yscale ),
            size( size )
    {

    }


    template<typename Reader>
    RESAMPLE_CALL std::complex<float> operator()( ResamplePos const& q, Reader& reader )
    {
        DataPos p(floor(q.x+.5f), floor(q.y+.5f));

        int up=1, left=-1, down=-1, right=1;

        // clamp
        if (p.x == 0)
            left = 0;
        if (p.y == 0)
            down = 0;
        if (p.x + 1 == size.x)
            right = 0;
        if (p.y + 1 == size.y)
            up = 0;

        std::complex<float> slope(
            (reader(DataPos(p.x + right, p.y)) - reader(DataPos(p.x + left, p.y)))*xscale/(right-left),
            (reader(DataPos(p.x, p.y+up)) - reader(DataPos(p.x, p.y+down)))*yscale/(up-down));

        return slope;
    }

private:
    const float xscale;
    const float yscale;
    const DataPos size;
};


extern "C"
void cudaCalculateSlopeKernel(  DataStorage<float>::Ptr heightmapIn,
                                DataStorage<std::complex<float> >::Ptr slopeOut,
                                float /*xscale*/, float /*yscale*/ )
{
    DataStorageSize sz_input = heightmapIn->size();
    DataStorageSize sz_output = slopeOut->size();

    ValidInputs validInputs( 0, 0, sz_input.width, sz_input.height );
    ValidOutputs validOutputs( sz_output.width, sz_output.height );

    resample2d_fetcher(heightmapIn, slopeOut,
                               validInputs, validOutputs,
                               ResampleArea(0, 0, 1, 1),
                               ResampleArea(0, 0, 1, 1),
                               false,
                               SlopeFetcher( 1000, 1000, DataPos( sz_input.width, sz_input.height) ),
                               AssignOperator<std::complex<float> >() );
}

#endif // SLOPEKERNELDEF_H
