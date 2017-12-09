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


#define FL_T2_Preloop \
  int FL_check = -1;\
  int FL_y=0;\
  volatile int *FL_pcount = &FL_count;\
 B: __threadfence();\
  if(FL_check == 0) {FL_check=-2;goto C;}\
  else if(FL_check == -2) goto P;\
  for(;FL_y+blockIdx.x<blocks; FL_y+=gridDim.x) \
    {//persistent threads loop

// every FL_childBlockSize parent threads handle one child kernel; the FL-childBlockSize parent threads process the child block tasks one by one in the inner loop
#define FL_T2_Postloop \
    }\
  FL_check = -2;\
 P:FL_syncAllThreads();\
  if(*FL_pcount != 0  ){\
    int ckernelSeqNum=0;\
  for(int i=0; FL_childBlockSize*(i+blockIdx.x)+threadIdx.x/FL_childBlockSize< *FL_pcount; i+=gridDim.x) \
      { \
	ckernelSeqNum = FL_childBlockSize*(i+blockIdx.x)+threadIdx.x/FL_childBlockSize;\

	
#define FL_postChildLog \
      }}	\
      FL_syncAllThreads(); \
      *FL_pcount=0; \
      FL_syncAllThreads(); \
      goto B; \
  } 

#endif	


