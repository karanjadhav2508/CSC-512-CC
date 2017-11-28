set(LLVM_LINK_COMPONENTS support)

add_clang_executable(cuda-rewrite
	CudaRewrite.cpp
    )
    target_link_libraries(cuda-rewrite
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
