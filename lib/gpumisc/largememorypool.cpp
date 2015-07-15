#include "largememorypool.h"
#include "neat_math.h"
#include <mutex>

// custom memory allocator for a few (<20) large (>1 MB) arrays
class LargeMemoryPool {
public:
    LargeMemoryPool(size_t N_threshold) : N_threshold(N_threshold) {
        Log("LargeMemoryPool: N_threshold=%s") % N_threshold;
    }

    ~LargeMemoryPool() {
        Log("~LargeMemoryPool: %s in %d blocks still allocated")
                % DataStorageVoid::getMemorySizeText (T)
                % blocks.size ();
    }

    const size_t N_threshold;
    size_t T = 0;

    void* getBlock(size_t n) {
        std::unique_lock<std::mutex> l(m);
//        n = std::max(N,n);
        n = spo2g (n-1);

        for (auto& i : blocks)
        {
            if (i.N == n && !i.used)
            {
                i.used = true;
                return i.p;
            }
        }

        void* p = new char[n];
        blocks.push_back (nfo{p,n,true});
        T += n;
        if (T > 1000*N_threshold) Log("LargeMemoryPool: allocated %s, total %s in %d blocks")
                % DataStorageVoid::getMemorySizeText (n)
                % DataStorageVoid::getMemorySizeText (T)
                % blocks.size ();
        return p;
    }

    void releaseBlock(void* p) {
        std::unique_lock<std::mutex> l(m);
        int r = 0;
        for (auto& i : blocks)
        {
            if (i.p == p)
                i.used = false, r++;
        }

        if (r!=1)
            Log("LargeMemoryPool: released %d blocks") % r;
    }

    void cleanPool(bool aggressive) {
        std::unique_lock<std::mutex> l(m);
        size_t C = 0, used_count = 0, used_size = 0;
        std::vector<nfo> newblocks;
        newblocks.reserve (blocks.size ()*3/4);
        std::map<size_t,int> N_unused;
        std::map<size_t,int> N_unused2;

        for (auto i : blocks)
            N_unused[i.N] += !i.used;

        for (auto i : blocks)
        {
            N_unused2[i.N] += !i.used;

            if (i.used)
            {
                used_size += i.N;
                used_count ++;
                newblocks.push_back (i);
            }
            else if (!aggressive && N_unused2[i.N]*2 > N_unused[i.N])
            {
                newblocks.push_back (i);
            }
            else
            {
                C += i.N;
                delete [](char*)i.p;
            }
        }

        blocks.swap (newblocks);
        T -= C;
        if (0 < C)
            Log("LargeMemoryPool: released %s in %d blocks. New total %s in %d blocks. Used: %s in %d blocks")
                % DataStorageVoid::getMemorySizeText (C)
                % (newblocks.size() - blocks.size ())
                % DataStorageVoid::getMemorySizeText (T)
                % blocks.size ()
                % DataStorageVoid::getMemorySizeText (used_size)
                % used_count;
    }

    std::mutex m;
    struct nfo { void* p; size_t N; bool used; };
    std::vector<nfo> blocks;
};


LargeMemoryPool pool {1 << 17};

void* lmp_malloc(size_t n) {
    if (n<=pool.N_threshold)
        return new char[n];
    return pool.getBlock (n);
}

void lmp_free(void* p, size_t n) {
    if (n<=pool.N_threshold)
        delete [](char*)p;
    else
        pool.releaseBlock (p);
}

void lmp_gc(bool aggressive) {
    pool.cleanPool (aggressive);
}
