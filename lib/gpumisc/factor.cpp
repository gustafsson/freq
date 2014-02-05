#include "factor.h"
#include "exceptionassert.h"
#include "timer.h"

Factor::vector Factor::
        factor(unsigned n)
{
    vector v;
    if (n<=1)
        v.push_back (n);

    while(n>1)
      {
        for (unsigned i=2; ; ++i)
          {
            unsigned a = n/i;
            if (a*i==n)
              {
                n = a;
                v.push_back (i);
                break;
              }

            if (a < i)
              {
                v.push_back (n);
                n = 1;
                break;
              }
          }
      }

    return v;
}


void Factor::
        test()
{
    // It should factor a number 'n' into its prime factors.
    {
        Timer T;
        EXCEPTION_ASSERT( factor(0) == vector { 0 } );
        EXCEPTION_ASSERT( factor(1) == vector { 1 } );
        EXCEPTION_ASSERT( factor(2) == vector { 2 } );
        EXCEPTION_ASSERT( factor(3*13*53*843487) == (vector { 3, 13, 53, 843487 }) );
        EXCEPTION_ASSERT_LESS( T.elapsed(), 50e-6 );

        EXCEPTION_ASSERT( factor(1743487517) == vector { 1743487517 } );
    }
}
