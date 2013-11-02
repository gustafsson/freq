#include "implicitordering.h"

#include "exceptionassert.h"

#include <sstream>
#include <map>

using namespace std;

namespace Test {

stringstream lout;

map<void*,int> objmap;

int objectnumber(void*p) {
    if (!objmap.count (p))
        objmap[p] = objmap.size () + 1;
    return objmap[p];
}

class A
{
public:
    A():data(-1) { lout << __FUNCTION__ << " " << objectnumber(this) << endl; }
    virtual ~A() { lout << __FUNCTION__ << " " << objectnumber(this) << endl; }

    int data;
};

A hej()
{
    return A();
}

class B
{
public:
    B() : data2(-2) {}
    virtual ~B() { lout << __FUNCTION__ << " " << objectnumber(this) << endl; }
    int data2;
};

class C: public A, public B
{
public:
    C() : data3(-2) {}
    virtual ~C() { lout << __FUNCTION__ << " " << objectnumber(this) << endl; }
    int data3;
};

class D {
public:
    D() : destroyed(123), copied(123) {}
    ~D() { destroyed = 456; }
    D(const D&); // not implemented
    D& operator=(const D&); // not implemented

    int destroyed;
    int copied;
};

D getD() {
    D d;
    return d;
}

void tsta(A*a)
{
    lout << objectnumber(a) << " a " << a->data << endl;
}

void tstb(B*b)
{
    lout << objectnumber(b) << " b " << b->data2 << endl;
}
void tstc(C*c)
{
    lout << objectnumber(c) << " c " << c->data3 << endl;
}

void ImplicitOrdering::
        test()
{
#ifndef __GCC__
    {A a;}
    lout.seekp (0);
#endif
    {
        C* c = new C;
        A* a = c;
        tsta(c);
        tstb(c);
        tstc(c);
        delete a;
        c = new C;
        B* b = c;
        tsta(c);
        tstb(c);
        tstc(c);
        delete b;
    }
    {
        C c;
        c.data = 1;
        c.data2 = 2;
        c.data3 = 3;
        tsta(&c);
        tstb(&c);
        tstc(&c);
    }
    {
        const A& a = hej();
        ((int)a.data);
    }
    string expected =
            "A 2\n"
            "2 a -1\n"
            "3 b -2\n"
            "2 c -2\n"
            "~C 2\n"
            "~B 3\n"
            "~A 2\n"
            "A 2\n"
            "2 a -1\n"
            "3 b -2\n"
            "2 c -2\n"
            "~C 2\n"
            "~B 3\n"
            "~A 2\n"
            "A 4\n"
            "4 a 1\n"
            "5 b 2\n"
            "4 c 3\n"
            "~C 4\n"
            "~B 5\n"
            "~A 4\n"
            "A 6\n"
            "~A 6\n";

    EXCEPTION_ASSERT_EQUALS(expected, lout.str());

    // Check that neither the copy constructor nor the assignment operator are called here
    D d = getD();
    EXCEPTION_ASSERT_EQUALS(d.copied, 123);
    EXCEPTION_ASSERT_EQUALS(d.destroyed, 123);
}

} // namespace Test
