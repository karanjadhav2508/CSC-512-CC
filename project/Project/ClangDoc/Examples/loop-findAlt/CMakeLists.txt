set(LLVM_LINK_COMPONENTS support)

add_clang_executable(loop-findAlt
  LoopFindAlt.cpp
    )
    target_link_libraries(loop-findAlt
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
