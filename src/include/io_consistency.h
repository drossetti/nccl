#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CUresult ioConsistencyInit(void);
CUresult ioConsistencyFenceCurrentCtx();
CUresult ioConsistencyFenceCtx(CUcontext ctx);
CUresult ioConsistencyDeviceSupportsCpuFlush(int devId, int *pflag);

cudaError_t ioRtConsistencyInit(void);
cudaError_t ioRtConsistencyFenceCurrentCtx();

#ifdef __cplusplus
}
#endif // __cplusplus
