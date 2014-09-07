#ifndef GLVECTOR_H
#define GLVECTOR_H

#include "gl.h"

#include "tvector.h"
#include "tmatrix.h"

typedef tvector<3,GLfloat> GLvectorf;
typedef tmatrix<4,GLfloat> GLmatrixf;
typedef tvector<3,double> vectord;
typedef tmatrix<4,double> matrixd;

#endif // GLVECTOR_H
