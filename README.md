<h1 align="center"> Nonbonded.jl </h1>
<h3 align="center"> Sérgio M. Santos & José M. Pereira </h3>
<h5 align="center"> CICECO & Departamento de Química - <a href="https://www.ua.pt">Universidade de Aveiro</a></h5>



## :scroll: Description

This repository contains step-by-step examples on code optimization for Lennard-Jones calculations (as an example on an N<sup>2</sup> problem), starting from a non-optimized prototype script all the way up to SIMD and GPU accelerated scripts.

## :clipboard: Installation

### A) To add package locally and use it in a Julia REPL:
1. In the directory that holds the package folder, run Julia (or cd to this location)
2. Using the Julia package manage (by clicking ']'), run **'activate LennardJones'**
3. If you'd like to make changes to the source code and update the Julia REPL, add Revise package (using the Julia package manager, by clicking ']' followed by **'add Revise'**) and run **'using Revise'**
3. Run **'using LennardJones'**

### B) To use package locally in a script
1. Depending on the location of the package folder, add the following lines to the beggining of the script
```
push!(LOAD_PATH, [package location here])
using LennardJones
```

## :round_pushpin: Benchmark

### To test the benchmark speed of the supplied functions in a machine
1. Run **julia ./benchmark.jl**, the file is located in the package root folder.

## :email: Contacts

For any question or curiosity, please contact jose.manuel.pereira@ua.pt.

## :trophy: Acknowledgments

<p align="center"> 
  <img src="./src/assets/ProtoSyn-acknowledgments.png" alt="Acknowledgments">
</p>