#ifndef RESAMPLECPU_H
#define RESAMPLECPU_H

#include <algorithm>
#include <math.h>
using namespace std;

#include "resample.h"

#include "cpumemorystorage.h"


template<typename FetchT, typename OutputT, typename Assignment>
static DefaultWriter<OutputT, Assignment, FetchT> DefaultWriterStorage(
        boost::shared_ptr<DataStorage<OutputT> > outputp,
        DataPos validOutputs,
        Assignment assignment)
{
    OutputT* outPtr = CpuMemoryStorage::ReadWrite<2>(outputp).ptr();

    // TODO use CpuMemoryStorage::WriteAll if we don't need to read from outputp

    unsigned outputPitch = outputp->size().width;

    return DefaultWriter<OutputT, Assignment, FetchT>(outPtr, outputPitch, validOutputs.y, assignment);
}


template<typename T>
class CpuReader: public CpuMemoryAccess<T, 2>
{
public:
    typedef CpuMemoryAccess<T, 2> Access;

    CpuReader(Access access)
        : Access(access)
    {}

    T operator()(DataPos const& p)
    {
        return Access::ref( DataAccessPosition<2>(p.x, p.y) );
    }
};


// todo remove
#include <stdio.h>


template<
        typename Reader,
        typename Fetcher,
        typename Writer,
        typename Transform>
static void resample2d_storage(
        ValidSamples validInputs,
        Fetcher fetcher,
        ValidSamples validOutputs,
        Transform transform,
        Reader reader,
        Writer writer
        )
{
#ifdef resample2d_DEBUG
    printf("\ngetLeft(validInputs) = %u", getLeft(validInputs));
    printf("\ngetTop(validInputs) = %u", getTop(validInputs));
    printf("\ngetRight(validInputs) = %u", getRight(validInputs));
    printf("\ngetBottom(validInputs) = %u", getBottom(validInputs));
    printf("\ninput.getNumberOfElements().x = %u", input.getNumberOfElements().x);
    printf("\ninput.getNumberOfElements().y = %u", input.getNumberOfElements().y);
#endif

#ifdef resample2d_DEBUG
    printf("\nvalidOutputs.x = %u", validOutputs.x);
    printf("\nvalidOutputs.y = %u", validOutputs.y);
    printf("\ngetLeft(validInputs) = %u", getLeft(validInputs));
    printf("\ngetTop(validInputs) = %u", getTop(validInputs));
    printf("\ngetRight(validInputs) = %u", getRight(validInputs));
    printf("\ngetBottom(validInputs) = %u", getBottom(validInputs));
    printf("\n");
    fflush(stdout);
#endif

    const int N = (int)validOutputs.bottom;

    ResampleArea validInputFloat(validInputs);
#pragma omp parallel for
    for (int i=validOutputs.top; i<N; ++i)
    {
        DataPos writePos(0,i);
        for (writePos.x=validOutputs.left; writePos.x<validOutputs.right; ++writePos.x)
        {
            resample2d_elem(
                    writePos,
                    validInputFloat,
                    fetcher,
                    validOutputs,
                    transform,
                    reader,
                    writer
            );
        }
    }
}


template<
        typename Fetcher,
        typename Writer,
        typename InputT
        >
static void resample2d_reader(
        boost::shared_ptr<DataStorage<InputT> > input,
        Writer writer,
        ValidSamples validInputs,
        ValidSamples validOutputs,
        DataPos outputSize,
        ResampleArea inputRegion = ResampleArea(0,0,1,1),
        ResampleArea outputRegion = ResampleArea(0,0,1,1),
        bool transpose = false,
        Fetcher fetcher = Fetcher()
        )
{
    CpuReader<InputT> reader = CpuMemoryStorage::ReadOnly<2>( input );

    resample2d_transform(
            reader,
            writer,
            validInputs, validOutputs, outputSize,
            inputRegion, outputRegion,
            transpose,
            fetcher
    );
}


#endif // RESAMPLECPU_H
