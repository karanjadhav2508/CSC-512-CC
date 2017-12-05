Project: SM-Centric Transformation

* Introduction

  In this project, the goal is to develop a source-to-source
  translator that can convert an input CUDA code that uses dynamic
  subkernel calls to a free launch
  form.

  A free launch form usually makes the code run much faster.  Paper
  freelaunchPaper_micro2015.pdf contained in this folder explains the
  details.  Folder "Examples" provides one example of code before and
  after free launch transformations, along with a README.txt file to
  explain the code changes invovled in the transformations.

* How to approach this project

  This part provides some general suggestions on how to approach this
  project.

  First, read the freelaunchPaper_micro2015.pdf paper to build up some
  background knowledge.

  Second, go through "Examples" folder. By reading the README.txt and
  trying out the examples, you will get an idea of the kind of
  code changes your compiler needs to materialize for a given GPU
  program.

  Third, learn about Clang if you haven't.

  Fourth, start implementing the compiler. Start with the simple
  cases, where, all the assumptions in the "Examples/README.txt" could
  be taken. After building up the support for the basic cases,
  try to develop more complete versions of the compiler with some or
  all of the assumptions relaxed.

  


