# Note for readers

For the 13_merge_sort, there are 3 files here:
- 13_merge_sort.cpp -> This uses OpenMPI tasks for parallelization. While its parallelized and runs on multiple cores, its performance is way worse than the original. I submit this as my solution, since the goal was to parallelize.
- 13_merge_sort_original.cpp -> This is the original file.
- 13_merge_sort_scope.cpp -> This uses OpenMPI sections for parallelization. While the workload is parallelized, it doesn't use all the cores. **I'm including this because this is the fastest working solution I got.**
