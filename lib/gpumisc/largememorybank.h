#ifndef LARGEMEMORYBANK_H
#define LARGEMEMORYBANK_H

/**
 * @brief lmb_malloc is a custom memory allocator designed for a few (<20) large (>1 MB) arrays
 * @param n size of memory to allocate
 * @return pointer to memory
 */
void* lmb_malloc(size_t n);

/**
 * @brief lmb_free, like free, compare lmb_malloc/malloc
 * @param p must be a pointer previously returned by lmb_malloc, can only free a pointer once
 * @param n size of memory to free. This is needed because the lmb algorithm uses different
 * strategies depending on the size of the memory so the size is needed to know where to look
 * for memory to release (i.e on the heap with delete[] or in its own pre-allocated data).
 */
void lmb_free(void* p, size_t n); // need 'n' to know if the memory bank should be used or not

/**
 * @brief lmb_gc runs garbage collection
 * @param threshold keep all allocated arrays smaller than or equal to this threshold measured in bytes
 */
void lmb_gc(size_t threshold);

#endif // LARGEMEMORYBANK_H
