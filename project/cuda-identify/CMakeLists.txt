set(LLVM_LINK_COMPONENTS support)

add_clang_executable(cuda-identify
	CudaIdentify.cpp
    )
    target_link_libraries(cuda-identify
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
