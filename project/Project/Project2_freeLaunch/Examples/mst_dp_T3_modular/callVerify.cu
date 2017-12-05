char *FL_Arguments; cudaMalloc((void **)&FL_Arguments,MAX_FL_ARGSZ);
cudaMemset(FL_Arguments,0,MAX_FL_ARGSZ);
verify_min_elem 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph.nnodes,graph.nedges,graph.noutgoing,graph.nincoming,graph.edgessrcdst,graph.edgessrcwt,graph.srcsrc,graph.psrc,  cs.ele2comp, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes,kconf.getNumberOfBlocks(),FL_Arguments);
cudaFree(FL_Arguments);
