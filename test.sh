#! /bin/bash
g++ -g -c rnn.cpp
g++ -g -c rnn_f.cpp
g++ -o test_rnn rnn.cpp rnn_f.cpp test_rnn.cpp
./test_rnn
