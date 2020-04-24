# Nonbonded.jl

## To add package locally and use it in a Julia REPL:
1. In the directory that holds the package folder, run Julia (or cd to this location)
2. Using the Julia package manage (by clicking ']'), run **'activate LennardJones'**
3. If you'd like to make changes to the source code and update the Julia REPL, add Revise package (using the Julia package manager, by clicking ']' followed by **'add Revise'**) and run **'using Revise'**
3. Run **'using LennardJones'**

## To run package tests
1. In an activated REPL (see above), go to the Julia package manager (by clicking ']') and run **'test'**

## To use package locally in a script
1. Depending on the location of the package folder, add the following lines to the beggining of the script
```
push!(LOAD_PATH, [package location here])
using LennardJones
```

## To test the benchmark speed of the supplied functions in a machine
1. Run **julia ./benchmark.jl**, the file is located in the package root folder.
