#ifndef __FreeLaunch_H
#define __FreeLaunch_H

#define MAX_FL_ARGSZ (1024*1024*128)
__device__ int FL_childBlockSize = 0;
__device__ int FL_count= 0;
__device__ uint FL__counter__ = 0;
__device__ uint FL_childKernelArgSz = 0;
__device__ void FL_syncAllThreads() {
  __syncthreads();
  uint to = gridDim.x-1;//getBlocksPerGrid() - 1;
  if (threadIdx.x==0) {
    volatile uint* counter = &FL__counter__;
    if (atomicInc((uint*) counter, to) < to) {
      while (*counter); // spinning...
    }
  }
  __syncthreads();
}
__device__ int FL_totalBlocks=0;


#define FL_T1_Preloop \
  int FL_ttid = blockIdx.x*blockDim.x+threadIdx.x;\
  int FL_check = -1;\
  int FL_y=0;\
  volatile int *FL_pcount = &FL_count;\
  volatile int *FL_ptotalBlocks = &FL_totalBlocks;\
 B: __threadfence();\
  if(FL_ttid == FL_check) {FL_check=-1;goto C;}\
  else if(FL_check == -2) goto P;\
  for(;FL_y+blockIdx.x<blocks; FL_y+=gridDim.x) \
    {//persistent threads loop

#define FL_T1_Postloop \
    }\
  FL_check = -2;\
 P:FL_syncAllThreads();\
  if(*FL_pcount != 0  ){\
    int ckernelSeqNum=0;int logicalChildBlockSeqNum=0;\
    int tasksPerParentThread = (*FL_ptotalBlocks+FL_childBlockSize*gridDim.x-1)/(FL_childBlockSize*gridDim.x); \
    for(int i=0;(i<tasksPerParentThread)&&(tasksPerParentThread*(FL_childBlockSize*blockIdx.x+threadIdx.x/FL_childBlockSize)+i<*FL_ptotalBlocks);i++) \
      { \
	int kernelSz;\
        memcpy((void*)&kernelSz, (void*)(&FL_Args[0]+ckernelSeqNum*FL_childKernelArgSz), sizeof(int)); \
	if(i==0){ \
	  logicalChildBlockSeqNum = tasksPerParentThread*(FL_childBlockSize*blockIdx.x+threadIdx.x/FL_childBlockSize); \
	  while(logicalChildBlockSeqNum-kernelSz>=0) \
	    { \
	      logicalChildBlockSeqNum-=kernelSz; \
	      ckernelSeqNum++; \
              memcpy((void*)&kernelSz, (void*)(&FL_Args[0]+ckernelSeqNum*FL_childKernelArgSz), sizeof(int)); \
	    } \
	} \
	else{ \
	  logicalChildBlockSeqNum +=1; \
	  if(logicalChildBlockSeqNum-kernelSz >= 0){ \
	    logicalChildBlockSeqNum-=kernelSz; \
	    ckernelSeqNum++; \
	  } \
	}


#define FL_postChildLog \
      } \
      FL_syncAllThreads(); \
      *FL_pcount=0; \
      *FL_ptotalBlocks=0; \
      FL_syncAllThreads(); \
      goto B; \
  } 

#endif	


