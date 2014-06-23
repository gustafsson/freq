#ifndef JUSTMISC_ZERO_ON_MOVE_H
#define JUSTMISC_ZERO_ON_MOVE_H

namespace JustMisc {

template<class T>
class zero_on_move {
public:
    zero_on_move(T t = 0) : t(t) {}
    zero_on_move(zero_on_move&&b) : t(b.t) {b.t=0;}
    zero_on_move(zero_on_move&b)=delete;
    zero_on_move& operator=(zero_on_move&b)=delete;
    zero_on_move& operator=(T v) { t = v; return *this; }
    operator bool() const { return (bool)t; }
    operator T() const { return t; }
    T* operator &() { return &t; } // Behave like a 'T'
    const T* operator &() const { return &t; } // Behave like a 'T'
private:
    T t;
};

} // namespace JustMisc

#endif // JUSTMISC_ZERO_ON_MOVE_H
