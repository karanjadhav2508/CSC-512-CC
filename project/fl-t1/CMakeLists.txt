set(LLVM_LINK_COMPONENTS support)

add_clang_executable(fl-t1
	FLT1.cpp
    )
    target_link_libraries(fl-t1
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
