__global__ void verify_min_elem(unsigned *mstwt, unsigned nnodes,unsigned nedges,unsigned *noutgoing,unsigned *nincoming,unsigned *edgessrcdst,foru *edgessrcwt, unsigned *srcsrc,unsigned *psrc,unsigned *ele2comp, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (inpid < nnodes) id = inpid;
  if (id < nnodes) {
    if(isBoss(ele2comp,id))
      {
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
	child<<<(degree+31)/32,32>>>(noutgoing,ele2comp,partners,id,edgessrcwt,edgessrcdst,srcsrc,psrc,processinnextiteration,nnodes,nedges,minwt_node,minwt,minwt_found,degree); // only the last two arguments were modified before this call within this function

      }
  }
}

