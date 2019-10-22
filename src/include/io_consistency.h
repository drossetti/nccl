#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CUresult ioConsistencyInit(void);
CUresult ioConsistencyFenceCurrentCtx();
CUresult ioConsistencyFenceCtx(CUcontext ctx);
CUresult ioConsistencyDeviceSupportsCpuFlush(int *pflag, int devId);

cudaError_t ioRtConsistencyInit(void);
cudaError_t ioRtConsistencyFenceCurrentCtx();
cudaError_t ioRtConsistencyDeviceSupportsCpuFlush(int *pflag, int device);

#ifdef __cplusplus
}
#endif // __cplusplus
