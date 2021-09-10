// 
#include "keras_model.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
using namespace std;
using namespace keras;
#include <chrono>
// Step 1
// Dump keras model and input sample into text files
// python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet
// Step 2
// Use text files in c++ example. To compile:
// g++ keras_model.cc example_main.cc
// To execute:
// a.out

int main() {

  KerasModel m("./example/dumped.nnet", true); 
  const clock_t begin_time = clock();
  auto t_start = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(4);
  double start_time = omp_get_wtime();
  #pragma omp parallel default(none) shared (m, std::cout)
  {
    //std::cout << "here1 " << omp_get_thread_num() << " " << omp_get_num_threads() << endl;

    //cout << "The 1-layer CNN model for delecting speed limit loading in to C++.\n"
    //        << "Keras model will be used in C++ for prediction only." << endl;
    #pragma omp single
    {
      #pragma omp task default(none) shared(m, std::cout)
      { 
        std::vector< std::vector<float>> Mat;
        std::vector<float> f;
        for (int i = 0; i<28; i ++) {
          std::vector<float> c;
          for (int j = 0; j<28; j++) {
            c.push_back(0);
            //cout << i << "---" << j << endl;
          }
          //cout << "here" << c.size() << endl;
          Mat.push_back(c);
        
        }
        
        DataChunk *sample = new DataChunk2D();
        sample->read_vector (Mat); // read from an array directly
        std::cout << "sample 3d size: " << sample->get_3d().size() << std::endl;
      
        f = m.compute_output(sample);
        for(size_t i = 0; i < f.size(); ++i) std::cout << f[i] << " ";
        delete sample;
      }
      #pragma omp task default(none) shared(m, std::cout)
      {
        std::vector< std::vector<float>> Mat;
        std::vector<float> f;
        for (int i = 0; i<28; i ++) {
          std::vector<float> c;
          for (int j = 0; j<28; j++) {
            c.push_back(0);
            //cout << i << "---" << j << endl;
          }
          //cout << "here" << c.size() << endl;
          Mat.push_back(c);
        
        }
        
        DataChunk *sample = new DataChunk2D();
        sample->read_vector (Mat); // read from an array directly
        std::cout << "sample 3d size: " << sample->get_3d().size() << std::endl;
      
        f = m.compute_output(sample);
        for(size_t i = 0; i < f.size(); ++i) std::cout << f[i] << " ";
        delete sample;
      }
    }
  }
  cout << "hahaha" << endl;
  double time = omp_get_wtime() - start_time;
  cout << "compute_inverse time" <<  time << " " << endl;

  std::cout << "time = " << float( clock () - begin_time ) << endl;;
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << "time2 = " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << endl;

  // std::cout << "DataChunkFlat values:" << std::endl;
  // for(size_t i = 0; i < f.size(); ++i) std::cout << f[i] << " ";
  std::cout << std::endl;

  
  return 0;
}
