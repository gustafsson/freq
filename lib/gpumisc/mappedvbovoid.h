#ifndef MAPPEDVBOVOID_H
#define MAPPEDVBOVOID_H

#include "datastorage.h"
#include "vbo.h"
#include "gl.h"

typedef boost::shared_ptr<Vbo> pVbo;

class TaskTimer;

class MappedVboVoid
{
    // Use through MappedVbo<T>
protected:
    MappedVboVoid(pVbo vbo);
    virtual ~MappedVboVoid();

    void map(DataStorageVoid* datap);
    void unmap(DataStorageVoid* datap);

private:
    DataStorage<char>::ptr mapped_gl_mem;

    pVbo _vbo;
    bool _is_mapped;

    TaskTimer* _tt; // depends on how TIME_MAPPEDVBO_LOG is defined
};

#endif // MAPPEDVBOVOID_H
