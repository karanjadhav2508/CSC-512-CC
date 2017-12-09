// FL transformation: added two new parameters.
//    blocks: the number of original parent thread blocks (i.e., before using persistent threads).
//    FL_Args: an array, with each element representing the parameters of a children kernel.
__global__ void verify_min_elem(unsigned *mstwt, unsigned nnodes,unsigned nedges,unsigned *noutgoing,unsigned *nincoming,unsigned *edgessrcdst,foru *edgessrcwt, unsigned *srcsrc,unsigned *psrc,unsigned *ele2comp, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, int blocks,char *FL_Args) {

  // FL transformation: insert this macro
  FL_T3_Preloop;

    /* This part is the copy of the original function body, except the replacements of "return" with "goto P", and the calculation of thread ID by considering the persistent thread usage */
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (inpid < nnodes) id = inpid;

  if (id < nnodes) {
    if(isBoss(ele2comp,id))
      {
	if(goaheadnodeofcomponent[id] == nnodes)
	  {
	    //printf("h?\n");
	    goto P;//return;
	  }

	unsigned minwt_node = goaheadnodeofcomponent[id];
	unsigned degree = user_getOutDegree(noutgoing,nnodes,minwt_node);
	foru minwt = minwtcomponent[id];
	if(minwt == MYINFINITY)
	  goto P;
	bool minwt_found = false;
  /************************************/

	  // FL transformation: removed the call to the child kernel and added the following to record the kernel info and go to processing place
	int FL_bllc = atomicAdd(&FL_blc,1);
	FL_childBlockSize=32; // In our current implementation, we assume that the child block sizes are identical to each other
	FL_childKernelArgSz = sizeof(int)+sizeof(unsigned)+sizeof(unsigned)+sizeof(foru)+sizeof(bool)+sizeof(unsigned);
	char * _tmp_p = (char *) (FL_pArgs+FL_bllc*FL_childKernelArgSz);
	int _tmp_childGridSize = (degree+31)/FL_childBlockSize;
	memcpy((void*)_tmp_p, (void*) &_tmp_childGridSize, sizeof(int));
	_tmp_p+=sizeof(int);
	memcpy((void*)_tmp_p, (void*) &id, sizeof(unsigned));
	_tmp_p+=sizeof(unsigned);
	memcpy((void*)_tmp_p, (void*) &minwt_node, sizeof(unsigned));
	_tmp_p+=sizeof(unsigned);
	memcpy((void*)_tmp_p,  (void*)&minwt, sizeof(foru));
	_tmp_p+=sizeof(unsigned);
	memcpy((void*)_tmp_p,  (void*)&minwt_found, sizeof(bool));
	_tmp_p+=sizeof(bool);
	memcpy((void*)_tmp_p,  (void*)&degree, sizeof(unsigned));
	// FL transformation: this part is invariant
	FL_check = 0;
	goto P;
      C:	__threadfence();
  /****** the part of the original parent kernel body after the child kernel call *******/
      }
  }

  // FL Transformation: added a macro
  FL_T3_Postloop;

  // FL transforamtion: added the following statements to retrieve the child kernel modified arguments
  char * _tmp_p = (char*)(FL_pArgs+ckernelSeqNum*FL_childKernelArgSz);
  int FL_childGridSize;
  memcpy((void*)&FL_childGridSize, (void*)_tmp_p, sizeof(int));
  _tmp_p+=sizeof(int);// move to the first function argument
  unsigned id;
  memcpy((void*)&id, (void*)_tmp_p, sizeof(unsigned));
  _tmp_p+=sizeof(unsigned);
  unsigned minwt_node;
  memcpy((void*)&minwt_node, (void*)_tmp_p, sizeof(unsigned));
  _tmp_p+=sizeof(unsigned);
  foru minwt;
  memcpy((void*)&minwt, (void*)_tmp_p, sizeof(foru));
  _tmp_p+=sizeof(unsigned);
  bool minwt_found;
  memcpy((void*)&minwt_found, (void*)_tmp_p, sizeof(bool));
  _tmp_p+=sizeof(bool);
  unsigned degree;
  memcpy((void*)&degree, (void*)_tmp_p, sizeof(unsigned));

  for(int kkk=0;kkk< FL_childGridSize;kkk++){// a loop to allow the parent thread block to handle all the child block tasks of this child kernel
    /***********************************************************************************************************/
  /* the copy of the child kernel body, except the calculation of thread global ID and replacing "return" with "continue", and the replacement of the calculation of ii, that is, the ID of the logical child thread*/
    unsigned ii = kkk *FL_childBlockSize +threadIdx.x%FL_childBlockSize;
    if(ii<degree){
      foru wt = user_getWeight(noutgoing,edgessrcwt,srcsrc,psrc,nnodes,nedges,minwt_node, ii);
      if (wt == minwt) {
	minwt_found = true;
	unsigned dst = user_getDestination(noutgoing,edgessrcdst,srcsrc,psrc,nnodes,nedges,minwt_node, ii);
	unsigned tempdstboss = user_find(ele2comp,dst);
	if(tempdstboss == partners[minwt_node] && tempdstboss != id)
	  {
	    processinnextiteration[minwt_node] = true;
	    continue;
	  }
      }
    }
    /*********************************************************/

    // FL Transformation: added a macro
    FL_postChildLog;
 }
