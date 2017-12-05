set(LLVM_LINK_COMPONENTS support)

add_clang_executable(loop-find
  LoopFind.cpp
    )
    target_link_libraries(loop-find
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
