#ifndef GLUUNPROJECT_H
#define GLUUNPROJECT_H

#include "GLvector.h"

GLvector gluProject(GLvector obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0);
GLvector gluUnProject(GLvector win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0);

#endif // GLUUNPROJECT_H
