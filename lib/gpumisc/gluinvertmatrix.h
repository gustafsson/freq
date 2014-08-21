#ifndef GLUINVERTMATRIX_H
#define GLUINVERTMATRIX_H

#include "GLvector.h"

// http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
bool gluInvertMatrix(const float m[16], float invOut[16]);
bool gluInvertMatrix(const double m[16], double invOut[16]);
GLmatrix invert(GLmatrix);

#endif // GLUINVERTMATRIX_H
