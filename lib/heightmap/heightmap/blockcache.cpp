#include "blockcache.h"
#include "reference_hash.h"
#include "heightmap/render/blocktextures.h"

#include "tasktimer.h"

using namespace std;

namespace Heightmap {

BlockCache::
        BlockCache()
{
}


pBlock BlockCache::
        find( const Reference& ref ) const
{
    lock_guard<mutex> l(mutex_);

    cache_t::const_iterator itr = cache_.find( ref );
    if (itr != cache_.end())
      {
        return itr->second;
      }
    else
      {
        return pBlock();
      }
}


void BlockCache::
        insert( pBlock b )
{
    lock_guard<mutex> l(mutex_);

    cache_[ b->reference() ] = b;
}


void BlockCache::
        erase (const Reference& ref)
{
    lock_guard<mutex> l(mutex_);

    cache_t::iterator i = cache_.find(ref);
    if (i != cache_.end())
        cache_.erase(i);
}


BlockCache::cache_t BlockCache::
        clear()
{
    lock_guard<mutex> l(mutex_);

    BlockCache::cache_t c;
    c.swap(cache_);
    return c;
}


size_t BlockCache::
        size() const
{
    lock_guard<mutex> l(mutex_);

    return cache_.size();
}


bool BlockCache::
        empty() const
{
    lock_guard<mutex> l(mutex_);

    return cache_.empty();
}


BlockCache::cache_t BlockCache::
        clone() const
{
    lock_guard<mutex> l(mutex_);

    BlockCache::cache_t C = cache_;

    return C;
}

} // namespace Heightmap

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

namespace Heightmap {

void BlockCache::
        test()
{
    std::string name = "BlockCache";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should store allocated blocks readily available
    {
        Reference r1;
        Reference r2 = r1.right ();
        BlockLayout bl(2,2,1);
        Render::BlockTextures::Scoped bt_raii(bl.texels_per_row (),bl.texels_per_column ());
        VisualizationParams::ptr vp;
        pBlock b1(new Block(r1, bl, vp, 0));
        pBlock b2(new Block(r2, bl, vp, 0));

        BlockCache c;
        c.insert (b1);
        c.insert (b2);
        pBlock b3 = c.find(r1);
        pBlock b4 = c.find(r2);
        pBlock b5 = c.find (r2.parentHorizontal ());
        pBlock b6 = c.find (r1.left ());

        EXCEPTION_ASSERT( b1 == b3 );
        EXCEPTION_ASSERT( b2 == b4 );
        EXCEPTION_ASSERT( r2 == r1.right () );
        EXCEPTION_ASSERT_EQUALS( r2.parentHorizontal (), r1 );
        EXCEPTION_ASSERT( b1 == b5 );
        EXCEPTION_ASSERT( b6 == pBlock() );
    }
}

} // namespace Heightmap
