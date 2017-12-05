/* FL Transformation: adding the delaration and allocation of an array "FL_Arguments" for saving and restoring child kernel
      arguments. Two extra arguments are added to the end of the call to "verify_min_elem": the first "kconf.getNumberOfBlocks()" is
      the first kernel parameter of "verify_min_elem", while the second is the newly added array "FL_Arguments". */

char *FL_Arguments; cudaMalloc((void **)&FL_Arguments,MAX_FL_ARGSZ);
cudaMemset(FL_Arguments,0,MAX_FL_ARGSZ);
    verify_min_elem 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph.nnodes,graph.nedges,graph.noutgoing,graph.nincoming,graph.edgessrcdst,graph.edgessrcwt,graph.srcsrc,graph.psrc,
 cs.ele2comp, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes,kconf.getNumberOfBlocks(),FL_Arguments);
cudaFree(FL_Arguments);
