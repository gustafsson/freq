#ifndef PRINTMATRIX_H
#define PRINTMATRIX_H

#include "GLvector.h"
#include "log.h"
#include "tasktimer.h"

#define PRINTMATRIX(M) printMatrix(M, #M)

template<int rows, typename t, int cols>
void printMatrix(tmatrix<rows,t,cols> const& M, const char* label)
{
    if (!label) label = "matrix";
    TaskInfo ti("%s", label);

    std::string ss;
    for(int r=0; r<rows; r++)
    {
        std::string row;
        for(int c=0; c<cols; c++)
            row += (boost::format("%g, ") % M[c][r]).str();
        if (!row.empty ())
            row = row.substr (0, row.size ()-2);
        Log("%s") % row;
    }
}

#endif // PRINTMATRIX_H
