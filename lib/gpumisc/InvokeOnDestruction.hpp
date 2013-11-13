#pragma once

#include <boost/bind.hpp>
#include <boost/function.hpp>

/**
  InvokeOnDestrcution invokes a given boost function in its destructor.

  If for instance T has a function declared as 'void T::myfunction(void)' and
  'o' is of type T. Then CallAfterScope is initiliazed as

@example InvokeOnDestrcution myVariable( bind<void>(printf, "Foo %d %s!\n", 1, "bar") );
  printf("Foo %d %s!\n", 1, "bar") will be called when 'myVariable' goes out of scope.

@example InvokeOnDestrcution myVariable( bind(&Foo::bar, foo) );
  'foo.bar()' will be called when 'myVariable' goes out of scope.
  */
class InvokeOnDestrcution {
public:
    InvokeOnDestrcution( boost::function<void()> f ):f(f) {}
    ~InvokeOnDestrcution() { f(); }

private:
    boost::function<void()> f;
};

