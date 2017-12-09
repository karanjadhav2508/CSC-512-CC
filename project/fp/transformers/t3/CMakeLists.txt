set(LLVM_LINK_COMPONENTS support)

add_clang_executable(t3
	FLT3.cpp
    )
    target_link_libraries(t3
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
