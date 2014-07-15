#include "gluunproject.h"

//#ifdef GL_ES_VERSION_2_0
#include "gluproject_ios.h"
#define GLU_FALSE 0
#define GLU_TRUE 1
//#endif

GLvector gluProject(GLvector obj, const GLvector::T* model, const GLvector::T* proj, const GLint *view, bool *r)
{
//    //gluProject does this, (win - screenspace).dot() < 1e18
//    tmatrix<4, double> modelmatrix(model);
//    tmatrix<4, double> projmatrix(proj);
//    tvector<4, double> obj4(obj[0], obj[1], obj[2], 1);
//    tvector<4, double> proj4 = projmatrix*modelmatrix*obj4;
//    tvector<3, double> screennorm(proj4[0]/proj4[3], proj4[1]/proj4[3], proj4[2]/proj4[3]);
//    GLvector screenspace(view[0] + (screennorm[0] + 1.0)*0.5*view[2],
//                                   view[1] + (screennorm[1] + 1.0)*0.5*view[3],
//                                   0.5 + 0.5*screennorm[2]);
//    if (r)
//        *r = 0 != proj4[3];
//    return screenspace;

    GLvector win;
    bool s = (GLU_TRUE == ::gluProject(obj[0], obj[1], obj[2], model, proj, view, &win[0], &win[1], &win[2]));
    if (r)
        *r = s;

    return win;
}


GLvector gluUnProject(GLvector win, const GLvector::T* model, const GLvector::T* proj, const GLint *view, bool *r)
{
    GLvector::T obj0=0, obj1=0, obj2=0;
    bool s = (GLU_TRUE == ::gluUnProject(win[0], win[1], win[2], model, proj, view, &obj0, &obj1, &obj2));
    if(r) *r=s;
    return GLvector(obj0, obj1, obj2);
}
