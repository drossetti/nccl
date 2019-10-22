#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "io_consistency.h"

#define ASSERT(C) assert(C)

//-----------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static const CUuuid CU_ETID_IoConsistency = { { 0x87, 0xe8, 0xff, 0x1d, 0xd6, 0x16, 0x57, 0x4b, 0xab, 0x36, 0x02, 0x97, 0x0a, 0x8c, 0x53, 0xb2 } };

/**
 * ETBL Versioning
 *
 * In general, version numbers will be incremented as follows:
 * - When a backwards-compatible change is made to the structure layout, the
 *   minor version for that structure will be incremented. Applications
 *   built against an older minor version will continue to work with the newer
 *   minor version of the APIs without recompilation.
 * - When a breaking change is made to the APIs, the major version
 *   will be incremented. Applications built against an older major
 *   version require at least recompilation and potentially additional updates
 *   to use the new API.
 */

#define CU_IOCONSISTENCY_VERSION 0x00010000
#define CU_IOCONSISTENCY_MAJOR_VERSION_MASK   0xffff0000
#define CU_IOCONSISTENCY_MINOR_VERSION_MASK   0x0000ffff

#define CU_IOCONSISTENCY_MAJOR_VERSION(v) \
    (((v) & CU_IOCONSISTENCY_MAJOR_VERSION_MASK) >> 16)

#define CU_IOCONSISTENCY_MINOR_VERSION(v) \
    (((v) & CU_IOCONSISTENCY_MINOR_VERSION_MASK))

#define CU_IOCONSISTENCY_MAJOR_VERSION_MATCHES(v) \
    (CU_IOCONSISTENCY_MAJOR_VERSION(v) == CU_IOCONSISTENCY_MAJOR_VERSION(CU_IOCONSISTENCY_VERSION))

#define CU_IOCONSISTENCY_VERSION_COMPATIBLE(v)    \
    (CU_IOCONSISTENCY_MAJOR_VERSION_MATCHES(v) && \
    (CU_IOCONSISTENCY_MINOR_VERSION(v) >= (CU_IOCONSISTENCY_MINOR_VERSION(CU_IOCONSISTENCY_VERSION))))

typedef enum CUioConsistencyAttribute_enum {
    CU_IOCONSISTENCY_ATTRIBUTE_API_VERSION = 0,     /**< API version implemented in this ETBL. Use the CU_IOCONSISTENCY_VERSION_COMPATIBLE() predicate to check for compatibility. */
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
     * TBD
     *
     * \param dev        - The device on which to issue the operation.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_NOT_SUPPORTED,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * \notefnerr
     *
     * \sa ::ioConsistencyFenceCurrentCtx
     */
    CUresult (CUDAAPI *ioConsistencyFenceCtx)(CUcontext ctx);

    /**
     * \brief Issue a fence for ingress traffic originating from third
     * party devices
     *
     * This is equivalent to cuCtxGetCurrent() followed by
     * ioConsistencyFenceCtx(), and is provided as an optimization.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_NOT_SUPPORTED,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED
     * ::CUDA_ERROR_INVALID_CONTEXT
     * \notefnerr
     *
     * \sa ::ioConsistencyFenceCtx
     */
    CUresult (CUDAAPI *ioConsistencyFenceCurrentCtx)();

} CUetblIoConsistency;

#ifdef __cplusplus
}
#endif // __cplusplus

#define VALID_ETBL(IOCONS) (NULL != (IOCONS))

//-----------------------------------------------------------

static CUetblIoConsistency *iocons = NULL;

static CUresult ioConsistencyGetAttribute(int *pi, CUioConsistencyAttribute attrib, CUdevice dev)
{
    if (VALID_ETBL(iocons))
        return CUDA_ERROR_UNKNOWN;
    return iocons->ioConsistencyGetAttribute(pi, attrib, dev);
}

CUresult ioConsistencyDeviceSupportsCpuFlush(int *pflag, int devId)
{
    CUresult rc;
    int attr = 0;
    CUdevice dev;
    rc = cuDeviceGet(&dev, devId);
    if (rc != CUDA_SUCCESS) {
        goto out;
    }
    rc = ioConsistencyGetAttribute(&attr, CU_IOCONSISTENCY_ATTRIBUTE_SUPPORT_HOSTSIDE_FENCE, dev);
    if (rc != CUDA_SUCCESS) {
        goto out;
    }
    ASSERT(pflag);
    *pflag = (attr != 0);
out:
    return rc;
}

CUresult ioConsistencyFenceCurrentCtx()
{
    ASSERT(iocons);
    return iocons->ioConsistencyFenceCurrentCtx();
}

CUresult ioConsistencyFenceCtx(CUcontext ctx)
{
    if (VALID_ETBL(iocons))
        return CUDA_ERROR_UNKNOWN;
    return iocons->ioConsistencyFenceCtx(ctx);
}

CUresult ioConsistencyInit()
{
    CUresult rc;
    // non-threadsafe
    if (iocons) {
        fprintf(stderr, "CUDA driver feature already initialized\n");
        rc = CUDA_SUCCESS;
        goto out;
    }

    rc = cuGetExportTable((const void**)&iocons, &CU_ETID_IoConsistency);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA driver does not have required support\n");
        goto out;
    }

    if (VALID_ETBL(iocons)) {
        fprintf(stderr, "this should never happen\n");
        rc = CUDA_ERROR_UNKNOWN;
    }

 out:
    return rc;
}

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
    ASSERT(iocons);
    return (cudaError_t)iocons->ioConsistencyFenceCurrentCtx();
}

cudaError_t ioRtConsistencyDeviceSupportsCpuFlush(int *pflag, int device)
{
    return (cudaError_t)ioConsistencyDeviceSupportsCpuFlush(pflag, device);
}
