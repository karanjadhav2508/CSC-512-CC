set(LLVM_LINK_COMPONENTS support)

add_clang_executable(loop-rewrite
  LoopRewrite.cpp
    )
    target_link_libraries(loop-rewrite
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
