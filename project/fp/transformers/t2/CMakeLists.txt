set(LLVM_LINK_COMPONENTS support)

add_clang_executable(t2
	FLT2.cpp
    )
    target_link_libraries(t2
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
