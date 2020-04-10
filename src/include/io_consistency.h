#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

cudaError_t ioRtConsistencyInit(void);
cudaError_t ioRtConsistencyFenceCurrentCtx();
cudaError_t ioRtConsistencyDeviceSupportsHostSideFence(int *pflag, int device);

#ifdef __cplusplus
}
#endif // __cplusplus
