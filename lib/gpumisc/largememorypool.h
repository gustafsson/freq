#ifndef LARGEMEMORYPOOL_H
#define LARGEMEMORYPOOL_H

/**
 * @brief lmp_malloc is a custom memory allocator designed for a few (<20) large (>1 MB) arrays
 * @param n size of memory to allocate
 * @return pointer to memory
 */
void* lmp_malloc(size_t n);

/**
 * @brief lmp_free, like free, compare lmp_malloc/malloc
 * @param p must be a pointer previously returned by lmp_malloc, can only free a pointer once
 * @param n size of memory to free. This is needed because the lmp algorithm uses different
 * strategies depending on the size of the memory so the size is needed to know where to look
 * for memory to release (i.e on the heap with delete[] or in its own pre-allocated data).
 */
void lmp_free(void* p, size_t n); // need 'n' to know if the memory pool should be used or not

/**
 * @brief lmp_gc runs garbage collection and releases all unused memory blocks
 * @param threshold keep all allocated arrays smaller than this threshold measured in bytes
 */
void lmp_gc(size_t threshold=0);

#endif // LARGEMEMORYPOOL_H
