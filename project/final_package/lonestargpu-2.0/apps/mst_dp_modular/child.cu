__global__ void child(unsigned *noutgoing,unsigned *ele2comp,unsigned *partners,unsigned id,foru *edgessrcwt,unsigned *edgessrcdst,unsigned *srcsrc,unsigned *psrc,bool *processinnextiteration,unsigned nnodes,unsigned nedges,unsigned minwt_node,foru minwt,bool minwt_found,unsigned degree)
{
  unsigned ii = threadIdx.x + blockIdx.x*blockDim.x;
  if(ii<degree){
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
}
