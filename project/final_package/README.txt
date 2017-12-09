-------------------------------------------- Folder Content --------------------------------------------------------------
  final_package contains:
  *transformers:
     t1: Clang tool to generate transformation 1 after subkernel launch removal
     t2: Clang tool to generate transformation 2 after subkernel launch removal
     t3: Clang tool to generate transformation 3 after subkernel launch removal
     t4: Clang tool to generate transformation 4 after subkernel launch removal
   
   To get the information about the structure of each clang tool, refer to the comments in the code, the report
   gives a general overview of our tool.
  
  *lonestars-2.0/apps:
     mst_dp_modular: the original benchmark using dynamic subkernel launches.

     mst_dp_T1_modular: the code after the first Free Launch transformation.

     mst_dp_T2_modular: the code after the second Free Launch transformation.

     mst_dp_T3_modular: the code after the third Free Launch transformation.

    mst_dp_T4_modular: the code after the fourth Free Launch transformation.
     
  *cub-1.7.4: 

-----------------------------------------STEPS to compile and run code on VM-----------------------------------------------------------

1. Untar the tar file in the the home directory of the VM i.e $HOME

2. First, you will need to install CUDA on VM. To do the installation, download "cuda.tgz" from Velocity, create a      
directory /usr/local/cuda, and enter that directory and run "sudo tar xzf ~/Download/cuda.tgz" (replace ~/Download with the path where your cuda.tgz is). 

3. The script that sets up the tools and generates the transformed versions
   is called "run.sh". Run it : ./run.sh
   NOTE: Warnings will be shown while compiling the tools, this is due to the presence of unused code, you can ignore them as they are not fatal.

---------------------------------------STEPS to run code on ARC cluster for testing----------------------------------------------------
4. Inorder to run code on ARC, scp the lonestargpu-2.0 folder and cub-1.7.4 folder present in final_package and create a softlink cub-1.7.4 under the top level folder of lonestars,pointing it to the folder of cub-1.7.4 just scp'd.
   NOTE:Please use the same cub folder we have provided as our lonestar folder is dependent on this particular version.

5. To download test cases, do the following:
   cd lonestargpu-2.0
   make inputs

6.
lonestargpu-2.0/apps/mst_dp_T1_modular
lonestargpu-2.0/apps/mst_dp_T2_modular
lonestargpu-2.0/apps/mst_dp_T3_modular
lonestargpu-2.0/apps/mst_dp_T4_modular

For each of these folders, do the following:
1. cd into the folder
2. make clean
3. make
4. chmod +x run
5. ./run
    NOTE : Please do not change or comment/uncomment anything in run.

=====Interpreting the results====
6. Compare the output of run to output of following command:
	head -1 lonestargpu-2.0/apps/mst_dp_modular/REF-RESULTS
They should match, to indicate success.
 
===Bugs and/or Limitations===
Doesn't run on other two test cases provided in run(they are commented out)
NOTE : Provided reference transformed code didn't run on them either.
