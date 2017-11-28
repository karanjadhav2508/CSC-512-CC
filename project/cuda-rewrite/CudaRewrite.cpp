#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PPCallbacks.h>
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include <string>
#include <iostream>
#include <iterator>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;
using namespace std;


// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

std::vector<const FunctionDecl *> kernelFuncs;

clang::SourceManager *sm;
clang::LangOptions lopt;

std::string decl2str(const clang::FunctionDecl *d) {
    if(sm == NULL) cout << "sm NULL" << endl;
    clang::SourceLocation b(d->getLocStart()), _e(d->getLocEnd());
    clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, *sm, lopt));
    return std::string(sm->getCharacterData(b),
        sm->getCharacterData(e)-sm->getCharacterData(b));
}

clang::CUDAKernelCallExpr * hasCudaKernelCallExpr(Stmt *s) {
  Stmt * ckce;
  iterator_range<StmtIterator> s_children = s->children();
  for(StmtIterator child = s_children.begin(); child != s_children.end(); child++) {
    if(*child != NULL) {
      if(isa<CUDAKernelCallExpr>(*child)) {
	return (CUDAKernelCallExpr*) *child;
      }
      ckce = hasCudaKernelCallExpr(*child);
      if(ckce != NULL && isa<CUDAKernelCallExpr>(ckce)) {
	return (CUDAKernelCallExpr*) ckce;
      }
    }
  }
  return NULL;
}

std::string getCudaKernelCallExprName(Stmt *c) {
  std::string ckceName;
  iterator_range<StmtIterator> c_children = c->children();
  for(StmtIterator child = c_children.begin(); child != c_children.end(); child++) {
    if(*child != NULL) {
      if(isa<DeclRefExpr>(*child)) {
        return ((DeclRefExpr*) *child)->getNameInfo().getName().getAsString();
      }
      ckceName = getCudaKernelCallExprName(*child);
      if(ckceName != "") {
        return ckceName;
      }
    }
  }
  return "";
}

class KernelFuncDefPrinter : public MatchFinder::MatchCallback {
public :
  KernelFuncDefPrinter(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    const FunctionDecl *kf = Result.Nodes.getNodeAs<clang::FunctionDecl>("kernelFunc");
    CUDAKernelCallExpr *ck = hasCudaKernelCallExpr(kf->getBody());
    if(ck == NULL) {
      kernelFuncs.push_back(kf);
    }
    else {
      cout << "PARENT : " << kf->getNameInfo().getName().getAsString() << endl;
      std::string childKernelName = getCudaKernelCallExprName((Stmt*) ck);
      cout << "child kernel call : " << childKernelName << endl;
      for(std::vector<const FunctionDecl *>::iterator f = kernelFuncs.begin(); f != kernelFuncs.end(); f++) {
	if((*f)->getNameInfo().getName().getAsString() == childKernelName) {
	  
	  /*
	  cout << "Kernel config :" << endl;
	  cout << "blocks" << endl;
	  ck->getConfig()->getArg(0)->child_begin()->child_begin()->child_begin()->child_begin()->child_begin()->dumpColor();
	  cout << "threads" << endl;
	  ck->getConfig()->getArg(1)->child_begin()->child_begin()->child_begin()->child_begin()->child_begin()->dumpColor();
	  */
	  
	  //ck->dumpColor();
	  
	  CXXConstructExpr *b = (CXXConstructExpr*) ck->getConfig()->getArg(0);
	  CXXConstructExpr *t = (CXXConstructExpr*) ck->getConfig()->getArg(1);

	  const SourceManager *sm = Result.SourceManager;
	  cout << "blocks : " << Lexer::getSourceText(CharSourceRange::getTokenRange(b->getSourceRange()), *sm, LangOptions()).str() << endl;
	  cout << "threads : " << Lexer::getSourceText(CharSourceRange::getTokenRange(t->getSourceRange()), *sm, LangOptions()).str() << endl;
	  break;
	}
      }
    }
  }

private:
  Rewriter &Rewrite;
};

class MyASTConsumer: public ASTConsumer {
public:
  MyASTConsumer (Rewriter &R) : KernelFuncPrinter(R) {
    Finder.addMatcher(KernelFuncMatcher, &KernelFuncPrinter);
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Finder.matchAST(Context);
  }

private:
  KernelFuncDefPrinter KernelFuncPrinter;
  DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("kernelFunc");
  MatchFinder Finder;
};

class IncludeFinder : public clang::PPCallbacks {
public:
  IncludeFinder (const clang::CompilerInstance &compiler) : compiler(compiler) {
    const clang::FileID mainFile = compiler.getSourceManager().getMainFileID();
    cout << "mainFile" << endl;
    name = compiler.getSourceManager().getFileEntryForID(mainFile)->getName();
    cout << name << endl;
  }

private:
  const clang::CompilerInstance &compiler;
  std::string name;
};

class IncludeFinderAction : public PreprocessOnlyAction {
public:
  IncludeFinderAction() {}

  void ExecuteAction() override {
    IncludeFinder includeFinder(getCompilerInstance());
    cout << "includeFinder" << endl;
    getCompilerInstance().getPreprocessor().addPPCallbacks((std::unique_ptr<clang::PPCallbacks>) (&includeFinder));
    cout << "addPPCallbacks" << endl;

    clang::PreprocessOnlyAction::ExecuteAction();
    cout << "ExecuteAction" << endl;
  }
};


// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    //TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
						 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
		 OptionsParser.getSourcePathList());

  //return Tool.run(newFrontendActionFactory<IncludeFinderAction>().get());
  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
