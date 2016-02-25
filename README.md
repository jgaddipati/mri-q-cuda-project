# mri-q-cuda-project

## Instructions

There are three folders for each of the stages described in the report  
(Stage 1 is not successful for all the test cases so not included).


* Stage 2 : using shared memory tiling technique
* Stage 3 : using constant memory tiling technique
* Stage 4 : using pinned memory asynchronous transfer technique


Compiling:
Each folder has its own Makefile to compile. 'make' command compiles and
produces the respective executable 'mri-q-cuda'. Input datasets folder is
commonly placed outside of the project folders. tools folder containing
'compare-output' script and its dependency scripts are also placed outside
project folders.

## Usage
```
 $ make
 $ ./mri-q-cuda -i ../datasets/small/input/32_32_32_dataset.bin -o output.bin
 $ ../tools/compare-output output.bin ../datasets/small/output/32_32_32_dataset.out
```
