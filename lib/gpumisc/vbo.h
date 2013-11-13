#ifndef VBO_H
#define VBO_H

#if !defined(__gl_h_) && !defined(__GL_H__) && !defined(__X_GL_H)
typedef unsigned int GLuint;
#endif
#if !defined(__glext_h_) && !defined(__GLEXT_H_)
//#define GL_ARRAY_BUFFER                   0x8892
#endif

#include <stddef.h> // size_t

/**
See
http://www.opengl.org/sdk/docs/man/xhtml/glBindBuffer.xml, for types

and
http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml, for access patterns
*/
class Vbo
{
public:
    // typical vbo_type = GL_ARRAY_BUFFER
    // access_pattern = GL_STATIC_DRAW, or GL_DYNAMIC_DRAW if changed every frame
    Vbo(size_t sz, unsigned vbo_type, unsigned access_pattern, void* data=0);
    virtual ~Vbo();
    operator GLuint() const;

    size_t size() { return _sz; }

#ifdef USE_CUDA
    void registerWithCuda();
#endif
    unsigned vbo_type() { return _vbo_type; }

private:
    Vbo(const Vbo &b);
    Vbo& operator=(const Vbo &b);

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
