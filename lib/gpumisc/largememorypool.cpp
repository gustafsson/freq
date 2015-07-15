#include "largememorypool.h"
#include "neat_math.h"
#include "tasktimer.h"
#include "timer.h"
#include <mutex>

#define LOG_ALLOCATION_SUMMARY
//#define LOG_ALLOCATION_SUMMARY if(0)

std::map<size_t,size_t> allocation_summary;

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

        LOG_ALLOCATION_SUMMARY
        {
            TaskInfo ti("Total number of allocations of each size (rounded up to nearest power of 2)");

            for (decltype(allocation_summary)::value_type v : allocation_summary)
                Log("%s: %llu") % DataStorageVoid::getMemorySizeText (v.first) % v.second;
        }
    }


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

        Timer t;
        void* p = new char[n];
        blocks.push_back (nfo{p,n,true});
        T += n;
        if (t.elapsed () > 10e-3 || (n >= 1 << 23 && T >= 1 << 30)) // allocating >=8 MB when the total is >=100 MB
            Log("LargeMemoryPool: allocated %s, total %s in %d blocks in %s")
                % DataStorageVoid::getMemorySizeText (n)
                % DataStorageVoid::getMemorySizeText (T)
                % blocks.size ()
                % TaskTimer::timeToString (t.elapsed ());
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
        Timer t;
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
        if (1 << 24 < C || t.elapsed () > 10e-3)
            Log("LargeMemoryPool: released %s in %d blocks. New total %s in %d blocks. Used: %s in %d blocks. Took %s")
                % DataStorageVoid::getMemorySizeText (C)
                % (newblocks.size() - blocks.size ())
                % DataStorageVoid::getMemorySizeText (T)
                % blocks.size ()
                % DataStorageVoid::getMemorySizeText (used_size)
                % used_count
                % TaskTimer::timeToString (t.elapsed ());
    }


    const size_t N_threshold;
    size_t T = 0;

private:
    std::mutex m;
    struct nfo { void* p; size_t N; bool used; };
    std::vector<nfo> blocks;
};


LargeMemoryPool pool {1 << 10};

void* lmp_malloc(size_t n) {
    LOG_ALLOCATION_SUMMARY allocation_summary[spo2g (n-1)]++;

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
