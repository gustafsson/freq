#include "largememorybank.h"
#include "neat_math.h"
#include <mutex>

// custom memory allocator for a few (<20) large (>1 MB) arrays
class LargeMemoryBank {
public:
    LargeMemoryBank(size_t N) : N(N), N_threshold(N/10) {
        Log("LargeMemoryBank: N=%s, N_threshold=%s") % N % N_threshold;
    }

    const size_t N;
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
        if (0) Log("cpu: allocated %s, total %s in %d blocks")
                % DataStorageVoid::getMemorySizeText (n)
                % DataStorageVoid::getMemorySizeText (T)
                % blocks.size ();
        return p;
    }

    void releaseBlock(void* p) {
        std::unique_lock<std::mutex> l(m);
        for (auto& i : blocks)
        {
            if (i.p == p)
                i.used = false;
        }
    }

    void cleanPool(size_t threshold) {
        std::unique_lock<std::mutex> l(m);
        size_t C = 0, used_count = 0, used_size = 0;
        std::vector<nfo> newblocks;
        newblocks.reserve (blocks.size ()*3/4);
        for (auto i : blocks)
            if (i.used || i.N <= threshold)
            {
                if (i.used) {
                    used_size += i.N;
                    used_count ++;
                }
                newblocks.push_back (i);
            }
            else
            {
                C += i.N;
                delete [](char*)i.p;
            }

        blocks.swap (newblocks);
        T -= C;
        if (0 < C)
            Log("cpu: clean pool %s in %d blocks, new total %s in %d blocks. Used: %s in %d blocks")
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


LargeMemoryBank bank {1 << 20};

void* lmb_malloc(size_t n) {
    if (n<=bank.N_threshold)
        return new char[n];
    return bank.getBlock (n);
}

void lmb_free(void* p, size_t n) {
    if (n<=bank.N_threshold)
        delete [](char*)p;
    else
        bank.releaseBlock (p);
}

void lmb_gc(size_t threshold) {
    bank.cleanPool (threshold);
}
