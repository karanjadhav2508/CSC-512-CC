set(LLVM_LINK_COMPONENTS support)

add_clang_executable(t1
	FLT1.cpp
    )
    target_link_libraries(t1
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
