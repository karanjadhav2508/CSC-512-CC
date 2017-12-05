#ifndef __FreeLaunch_H
#define __FreeLaunch_H

#define MAX_FL_ARGSZ (1024*1024*32)
#define MAX_FL_ARGSZ_PER_KERNEL (1024*32)
__device__ int FL_childBlockSize = 0;
__device__ uint FL_childKernelArgSz = 0;

#define FL_T3_Preloop \
  int FL_check = -1;\
__shared__ int FL_blc;				\
char *FL_pArgs = &FL_Args[blockIdx.x*MAX_FL_ARGSZ_PER_KERNEL];	\
  if (threadIdx.x==0) FL_blc=0;\
  __syncthreads();\
B: __threadfence();				\
  if(FL_check == 0) {FL_check=-2;goto C;}\
  else if(FL_check == -2) goto P;\

// every FL_childBlockSize parent threads handle one child kernel
#define FL_T3_Postloop \
  FL_check = -2;\
 P:__syncthreads();\
  if(FL_blc != 0  ){\
    int ckernelSeqNum=0;\
    for(int kk=0; FL_childBlockSize*kk+threadIdx.x/FL_childBlockSize< FL_blc; kk++)\
      { \
	ckernelSeqNum = FL_childBlockSize*kk+threadIdx.x/FL_childBlockSize;\

	
#define FL_postChildLog \
      }}	\
      __syncthreads(); \
      FL_blc=0; \
      __syncthreads(); \
      goto B; \
  } 

#endif	


