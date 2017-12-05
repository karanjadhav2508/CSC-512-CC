set(LLVM_LINK_COMPONENTS support)

add_clang_executable(cudaCall-find
  CudacallFind.cpp
    )
    target_link_libraries(cudaCall-find
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
