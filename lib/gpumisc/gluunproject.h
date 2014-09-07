#ifndef GLUUNPROJECT_H
#define GLUUNPROJECT_H

#include "GLvector.h"

vectord gluProject(vectord obj, const vectord::T* model, const vectord::T* proj, const GLint *view, bool *r=0);
vectord gluUnProject(vectord win, const vectord::T* model, const vectord::T* proj, const GLint *view, bool *r=0);

#endif // GLUUNPROJECT_H
