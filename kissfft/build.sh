cmake -G Ninja -B build -DCMAKE_INSTALL_PREFIX=build/install -DKISSFFT_TEST=OFF kissfft

cmake --build build --target install -j4

