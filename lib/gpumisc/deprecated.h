#ifndef DEPRECATED_H
#define DEPRECATED_H

/* example

To make A::a deprecated:

class A {
    void a() const;
};

to->

class A {
    DEPRECATED( void a() const );
};

Note A::a must not be inlined in the class definition.
But can be inlined the header with the inline keyword.
*/


#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

#endif // DEPRECATED_H
