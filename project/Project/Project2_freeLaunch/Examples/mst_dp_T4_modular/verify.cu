// No need to add new parameters for this T4 transformation.
__global__ void verify_min_elem(unsigned *mstwt, unsigned nnodes,unsigned nedges,unsigned *noutgoing,unsigned *nincoming,unsigned *edgessrcdst,foru *edgessrcwt, unsigned *srcsrc,unsigned *psrc, unsigned *ele2comp, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid){

  /***********************************************************************************************************/
  /* This part is the copy of the original function body, except the replacements of "return" with "goto P", and the calculation of thread ID by considering the persistent thread usage */
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (inpid < nnodes) id = inpid;
  if (id < nnodes) {
    if(isBoss(ele2comp,id)){
	if(goaheadnodeofcomponent[id] == nnodes)
	  {
	    return;
	  }

	unsigned minwt_node = goaheadnodeofcomponent[id];
	unsigned degree = user_getOutDegree(noutgoing,nnodes,minwt_node);
	foru minwt = minwtcomponent[id];
	if(minwt == MYINFINITY)
	  return;  
	bool minwt_found = false;
  /*************************************************************************************************/
	    
  // FL transformation: removed the call to the child kernel and inline its content with a surrounding loop on the range of threads of this child kernel; this step removes the calculation of the thread global ID that was in the original child kernel.
	int FL_childGridThreads = int((degree+31)/32)*32; // the product of the two parameters of the child kernel
	for (unsigned ii = 0; ii < FL_childGridThreads; ++ii) {
          /***********************************************************************************************************/
	  /* This part is the copy of the original child kernel function body*/
	  /* A potential optimization is to break from the loop as soon as ii>=degree */
	  if(ii<degree) {
		foru wt = user_getWeight(noutgoing,edgessrcwt,srcsrc,psrc,nnodes,nedges,minwt_node, ii);
		if (wt == minwt) {
		  minwt_found = true;
		  unsigned dst = user_getDestination(noutgoing,edgessrcdst,srcsrc,psrc,nnodes,nedges,minwt_node, ii);
		  unsigned tempdstboss = user_find(ele2comp,dst);
		  if(tempdstboss == partners[minwt_node] && tempdstboss != id)
		    {
		      processinnextiteration[minwt_node] = true;
		      return;
		    }
		}
	  }
        /**************************************************************************************/    
	}
    }
  }
}
