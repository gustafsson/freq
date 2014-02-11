#ifndef DATASTORAGEACCESS_H
#define DATASTORAGEACCESS_H

#ifndef ACCESSCALL
#define ACCESSCALL
#endif

//#include <stddef.h> // size_t
//typedef size_t DataAccessPosition_t;
typedef int DataAccessPosition_t;


template<unsigned>
class DataAccessPosition;

template<>
class DataAccessPosition<1>
{
public:
    typedef DataAccessPosition_t T;

    ACCESSCALL DataAccessPosition( T x=0 )
        :
        x( x )
    {}

    ACCESSCALL DataAccessPosition( DataAccessPosition<2> p );


    union
    {
        T x;
        T v[1];
    };
};


template<>
class DataAccessPosition<2>
{
public:
    typedef DataAccessPosition_t T;

    ACCESSCALL DataAccessPosition( T x=0, T y=0 )
        :
        x( x ),
        y( y )
    {}


    ACCESSCALL DataAccessPosition( DataAccessPosition<1> p )
        :
        x( p.x ),
        y( 0 )
    {}


    union
    {
        struct { T x, y; };
        T v[2];
    };
};


template<>
class DataAccessPosition<3>
{
public:
    typedef DataAccessPosition_t T;

    ACCESSCALL DataAccessPosition( T x=0, T y=0, T z=0 )
        :
        x( x ),
        y( y ),
        z( z )
    {}


    ACCESSCALL DataAccessPosition( DataAccessPosition<1> p )
        :
        x( p.x ),
        y( 0 ),
        z( 0 )
    {}

    ACCESSCALL DataAccessPosition( DataAccessPosition<2> p )
        :
        x( p.x ),
        y( p.y ),
        z( 0 )
    {}


    union
    {
        struct { T x, y, z; };
        T v[3];
    };
};



template<unsigned>
class DataAccessSize;

template<>
class DataAccessSize<1>
{
public:
    typedef DataAccessPosition_t T;

    ACCESSCALL DataAccessSize( T width )
        :
        width( width )
    {}


    ACCESSCALL DataAccessSize<1>( DataAccessSize<2> p );
    ACCESSCALL DataAccessSize<1>( DataAccessSize<3> p );


    ACCESSCALL DataAccessPosition<1> clamp( DataAccessPosition<1> p ) const
    {
        if (p.x > width-1)
            p.x = width-1;
        if (p.x < 0) p.x = 0;
        return p;
    }


    ACCESSCALL DataAccessPosition_t offset( DataAccessPosition<1> p ) const
    {
        DataAccessPosition<1> q = clamp( p );
        return q.x;
    }


    T width;
};


template<>
class DataAccessSize<2>
{
public:
    typedef DataAccessPosition_t T;

    ACCESSCALL DataAccessSize( T width, T height=1 )
        :
        width( width ),
        height( height )
    {}


    ACCESSCALL DataAccessSize( DataAccessSize<1> p )
        :
        width( p.width ),
        height( 1 )
    {}


    ACCESSCALL DataAccessSize<2>( DataAccessSize<3> p );


    ACCESSCALL DataAccessPosition<2> clamp( DataAccessPosition<2> p ) const
    {
        if (p.x > width-1)
            p.x = width-1;
        if (p.x < 0) p.x = 0;
        if (p.y > height-1)
            p.y = height-1;
        if (p.y < 0) p.y = 0;
        return p;
    }


    ACCESSCALL DataAccessPosition_t offset( DataAccessPosition<2> p ) const
    {
        DataAccessPosition<2> q = clamp( p );
        return q.x + q.y*width;
    }

    T width, height;
};


template<>
class DataAccessSize<3>
{
public:
    typedef DataAccessPosition_t T;

    ACCESSCALL DataAccessSize( T width, T height=1, T depth=1 )
        :
        width( width ),
        height( height ),
        depth( depth )
    {}

    ACCESSCALL DataAccessSize( DataAccessSize<1> p )
        :
        width( p.width ),
        height( 1 ),
        depth( 1 )
    {}

    ACCESSCALL DataAccessSize( DataAccessSize<2> p )
        :
        width( p.width ),
        height( p.height ),
        depth( 1 )
    {}


    ACCESSCALL DataAccessPosition<3> clamp( DataAccessPosition<3> p ) const
    {
        if (p.x > width-1)
            p.x = width-1;
        if (p.x < 0) p.x = 0;
        if (p.y > height-1)
            p.y = height-1;
        if (p.y < 0) p.y = 0;
        if (p.z > depth-1)
            p.z = depth-1;
        if (p.z < 0) p.z = 0;
        return p;
    }


    ACCESSCALL unsigned offset( DataAccessPosition<3> p ) const
    {
        // *this is size of data
        DataAccessPosition<3> q = clamp( p );
        return q.x + width*(q.y + height*q.z);
    }


    bool operator==( DataAccessSize<3> const& b ) const
    {
        return width == b.width && height == b.height && depth == b.depth;
    }


    bool operator!=( DataAccessSize<3> const& b ) const
    {
        return !(*this == b);
    }


    T width, height, depth;
};


inline ACCESSCALL DataAccessSize<1>::DataAccessSize( DataAccessSize<2> p )
    :
    width( p.width*p.height )
{}

inline ACCESSCALL DataAccessSize<1>::DataAccessSize( DataAccessSize<3> p )
    :
    width( p.width*p.height*p.depth )
{}

inline ACCESSCALL DataAccessSize<2>::DataAccessSize( DataAccessSize<3> p )
    :
    width( p.width ),
    height( p.height*p.depth )
{}


// don't inherit from nor instanciate this class,
// it's just a _lazy_ concept definition
template<typename T>
class DataStorageAccessConcept
{
public:
    DataAccessSize<1> numberOfElements();

    T& ref( const DataAccessPosition<1>& );

    T read( const DataAccessPosition<1>& );

    void write( const DataAccessPosition<1>&, const T& );

private:
    // don't inherit from nor instanciate this class,
    // it's just a lazy concept definition
    DataStorageAccessConcept();
};


typedef DataAccessSize<3> DataStorageSize;


template<typename C>
inline std::basic_ostream<C>& operator<<(std::basic_ostream<C>& s, const DataStorageSize& v) {
    if (v.depth > 1)
        s << "[" << v.width << ", " << v.height << ", " << v.depth << "]";
    else if (v.height > 1)
        s << "[" << v.width << ", " << v.height << "]";
    else
        s << "[" << v.width << "]";
    return s;
}

#endif // DATASTORAGEACCESS_H
