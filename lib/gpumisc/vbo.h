#ifndef VBO_H
#define VBO_H

#include "gl.h"
#include <stddef.h> // size_t

/**
See
http://www.opengl.org/sdk/docs/man/xhtml/glBindBuffer.xml, for types

and
http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml, for access patterns
*/
class Vbo final
{
public:
    // typical vbo_type = GL_ARRAY_BUFFER
    // access_pattern = GL_STATIC_DRAW, or GL_DYNAMIC_DRAW if changed every frame
    Vbo(size_t sz, unsigned vbo_type, unsigned access_pattern, void* data=0);
    Vbo(const Vbo &b)=delete;
    Vbo(Vbo &&b);
    Vbo&operator =(const Vbo &b)=delete;
    Vbo&operator =(Vbo &&b);
    virtual ~Vbo();
    operator GLuint() const;

    size_t size() const { return _sz; }

#ifdef USE_CUDA
    void registerWithCuda();
#endif
    unsigned vbo_type() { return _vbo_type; }

private:
    void init(size_t size, unsigned vbo_type, unsigned access_pattern, void* data);
    void clear();

    size_t _sz;
    GLuint _vbo;

#ifdef USE_CUDA
    bool _registered;
#endif
    unsigned _vbo_type;
};

#endif // VBO_H
