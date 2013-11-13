#ifndef SINGLETONP_H
#define SINGLETONP_H

#include <boost/shared_ptr.hpp>

/**
  HasSingleton can be used for classes that wants to provide a singleton
  instance as a boost::shared_ptr<baseT>.

  The instance can be released by calling SingletonP().reset(); A new instance
  will then be created by the next call to SingletonP().
  */
template<typename T, typename baseT=T>
class HasSingleton {
public:
    static T& Singleton() {
        return *dynamic_cast<T*>(SingletonP().get());
    }

    static boost::shared_ptr<baseT>& SingletonP() {
        static boost::shared_ptr<baseT> P;

        if (!P)
            P.reset(new T());

        return P;
    }
};

#endif // SINGLETONP_H
