#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "io_consistency.h"

#define ASSERT(C) assert(C)

//-----------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static const CUuuid CU_ETID_IoConsistency = { { (char)0x87, (char)0xe8, (char)0xff, (char)0x1d, (char)0xd6, (char)0x16, (char)0x57, (char)0x4b, (char)0xab, (char)0x36, (char)0x02, (char)0x97, (char)0x0a, (char)0x8c, (char)0x53, (char)0xb2 } };

#define CU_IOCONSISTENCY_VERSION 0x00010000
#define CU_IOCONSISTENCY_MAJOR_VERSION_MASK   0xffff0000
#define CU_IOCONSISTENCY_MINOR_VERSION_MASK   0x0000ffff

#define CU_IOCONSISTENCY_MAJOR_VERSION(v) \
    (((v) & CU_IOCONSISTENCY_MAJOR_VERSION_MASK) >> 16)

#define CU_IOCONSISTENCY_MINOR_VERSION(v) \
    (((v) & CU_IOCONSISTENCY_MINOR_VERSION_MASK))

#define CU_IOCONSISTENCY_MAJOR_VERSION_MATCHES(v) \
    (CU_IOCONSISTENCY_MAJOR_VERSION(v) == CU_IOCONSISTENCY_MAJOR_VERSION(CU_IOCONSISTENCY_VERSION))

typedef enum CUioConsistencyAttribute_enum {
    CU_IOCONSISTENCY_ATTRIBUTE_API_VERSION = 0, /**< API version implemented in this ETBL. */
    CU_IOCONSISTENCY_ATTRIBUTE_SUPPORT_HOSTSIDE_FENCE = 1, /**< returns true if the device supports host-side IO fence. */
    CU_IOCONSISTENCY_ATTRIBUTE_MAX
} CUioConsistencyAttribute;


typedef struct CUetblIoConsistency_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /**
     * \brief Returns information about the CUDA IO consistency attributes
     * for the device
     *
     * Returns in \p *pi the integer value of the attribute \p attrib on
     * device \p dev.
     *
     * The supported attributes are:
     * - ::CU_IOCONSISTENCY_ATTRIBUTE_API_VERSION: API version implemented in this ETBL. Use the CU_IOCONSISTENCY_VERSION_COMPATIBLE() predicate to check for compatibility.
     * - ::CU_IOCONSISTENCY_ATTRIBUTE_SUPPORT_HOSTSIDE_FENCE: returns true if the device supports host-side IO fence.
     *
     * \param pi     - Returned attribute value
     * \param attrib - Attribute to query
     * \param dev    - Device handle
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_INVALID_DEVICE
     * \notefnerr
     *
     * \sa ::ioConsistencyFenceCtx, ::ioConsistencyFenceCurrentCtx
     */
    CUresult (CUDAAPI *ioConsistencyGetAttribute)(int *pi, CUioConsistencyAttribute attrib, CUdevice dev);

    /**
     * \brief Issue a fence for ingress traffic originating from third
     * party devices
     *
     * This is equivalent to cuCtxGetDevice() followed by
     * ioConsistencyFenceDevice(), and is provided as an optimization.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_NOT_SUPPORTED,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED
     * ::CUDA_ERROR_INVALID_CONTEXT
     * \notefnerr
     *
     * \sa ::ioConsistencyFenceDevice
     */
    CUresult (CUDAAPI *ioConsistencyFenceCurrentCtx)();

    void *reserved[64];

} CUetblIoConsistency;

#ifdef __cplusplus
}
#endif // __cplusplus

#define VALID_ETBL(IOCONS) (NULL != (IOCONS))

//-----------------------------------------------------------

static CUetblIoConsistency *iocons = NULL;

cudaError_t ioRtConsistencyInit()
{
    cudaError_t rc;
    // non-threadsafe
    if (VALID_ETBL(iocons)) {
        rc = cudaGetExportTable((const void**)&iocons, &CU_ETID_IoConsistency);
        if (rc != cudaSuccess) {
            fprintf(stderr, "CUDA driver does not have required support\n");
            goto out;
        }

        if (VALID_ETBL(iocons)) {
            fprintf(stderr, "this should never happen\n");
            rc = cudaErrorUnknown;
            goto out;
        }
       
    } else {
        //fprintf(stderr, "CUDA driver feature already initialized\n");
        rc = cudaSuccess;
    }
 out:
    return rc;
}

cudaError_t ioRtConsistencyFenceCurrentCtx()
{
    if (VALID_ETBL(iocons))
        return cudaErrorUnknown;
    return (cudaError_t)iocons->ioConsistencyFenceCurrentCtx();
}

cudaError_t ioRtConsistencyDeviceSupportsHostSideFence(int *pflag, int device)
{
    cudaError_t rc;
    CUresult cr;
    int attr = 0;
    CUdevice dev = (device);
    if (VALID_ETBL(iocons)) {
        rc = cudaErrorUnknown;
        goto out;
    }
    cr = iocons->ioConsistencyGetAttribute(&attr, CU_IOCONSISTENCY_ATTRIBUTE_SUPPORT_HOSTSIDE_FENCE, dev);
    if (cr != CUDA_SUCCESS) {
        rc = (cudaError_t)cr;
        goto out;
    }
    ASSERT(pflag);
    *pflag = attr;
out:
    return rc;
}
