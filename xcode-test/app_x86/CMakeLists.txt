cmake_minimum_required(VERSION 3.14)
# cmake_minimum_required(VERSION 3.18)

project (all_algs)



set(SUBPATH "test_main" CACHE PATH "test_main")

add_subdirectory (${SUBPATH})
# add_subdirectory (test_main)

# cmake ..
# make -j32
