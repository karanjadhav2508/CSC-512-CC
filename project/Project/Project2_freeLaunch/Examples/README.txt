REAMDE
=======

This folder contains an example program and its variants that show the
four Free Launch transformations documented in "Free Launch:
Optimizing GPU Dynamic Kernel Launches through Thread Reuse" published
at Micro'2015.

---- Folder Content -----

  mst_dp_modular: the original benchmark using dynamic subkernel
  launches.

  mst_dp_T1_modular: the code after the first Free Launch
  transformation.

  mst_dp_T2_modular: the code after the second Free Launch
  transformation.

  mst_dp_T3_modular: the code after the third Free Launch
  transformation.

  mst_dp_T4_modular: the code after the fourth Free Launch
  transformation.

  common.mk: a file to be used for the compilation of the programs.
  
---- Building and Run -----
  Download and build LoneStars v2.0 package
    http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu
  NOTE: the process requires the installation of cub, install
  cub-1.7.3. 

  Create a softlink cub-1.7.3 under the top level folder of lonestars,
  pointing it to the folder of the installed cub-1.7.3.

  Copy common.mk to the "apps" folder in lonestars to replace the old
  common.mk file (some compilation flags are added to enable dynamic
  parallelism in CUDA).

  Copy the other folders in this directory to the "apps" folder in
  lonestars.

  To compiler a version, cd into that folder, run "make clean;make".

  To run a version, cd into that folder, execute "run". The file
  "REF-RESULTS" provides the reference of the correct final output of
  the executions on three inputs. The "run" command, by default, only
  processes the first input. 
  
---- Explanation of the versions -----
* The original version

  The program is a minimum spanning tree program. The original version
  "mst_dp_modular" came from the LoneStars package
  (http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu), and was
  adapted as follows:

   (1) A loop in kernel, "verify_min_elem", was replaced with a call
  to a sub-kernel "child".

   (2) The program code was reorganized slightly to make it easier to
   compare with the transformed variants. Specifically, the kernel,
   "verify_min_elem", was moved into a separate file "verify.cu", the
   call of it was moved into a separate file "callVerify.cu", and the
   "child" kernel was moved into a separate file "child.cu".

* The transformed versions

  Common assumptions:
    * In the original program, there is only one child kernel to be
  optimized, which is the target of the Free Launch transformations.
    * Different launches of the child kernel (by different parent
  threads) could differ in the number of child thread blocks, but
  share the same child thread block size.
    * The size of a parent thread block is a multiple of the size of a
  child thread block.
    
  (1) mst_dp_T1_modular

  This version distributes all child block tasks to all parent
      threads.

  Major code changes include:
      (1.1) added a header file, "freeLaunch_T1.h"
      (1.2) modified verify.cu (see code comments)
      (1.3) modified callVerify.cu (see code comments)

  (2) mst_dp_T2_modular

  Similar to mst_dp_T1_modular. This version distributes an entire child kernel
  rather than child block tasks.

  The main code differences from mst_dp_T1_modular are in
  "verfiy.cu" and "freeLaunch_T2.h". See comments. 

  (3) mst_dp_T3_modular

  Similar to mst_dp_T1_modular. In this version, a child kernel is
  to be executed by the parent thread block that is supposed to launch
  it. 

  The main code differences from mst_dp_T1_modular are in
  "verfiy.cu" and "freeLaunch_T2.h". See comments. 

  (4) mst_dp_T4_modular

  This version is the simplest to produce. It uses the parent thread
  to run the child kernel that it is supposed to launch.

  The code change from the original version is simply inlining the
  child kernel into the call site and add a surrounding loop. See code
  comments in verify.cu. 

=============================================================================
Xipeng Shen (xshen5@ncsu.edu)
9/6/2017
