#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>

struct val {
  std::string word;
  size_t count;
  int newid;
};

int newid_ctr = 0;

std::map<size_t, val> top_words;

using rec = std::pair<size_t, val>;

constexpr const int N = 400000;

std::map<std::pair<size_t, size_t>, double> mat;

int main() {
  size_t num_words;
  std::cin >> num_words;
  std::vector<rec> words;
  words.resize(num_words);
  for(size_t i = 0; i < num_words; ++i) {
    auto & w = words[i];
    std::cin >> w.first >> w.second.count >> w.second.word;
    w.second.newid = -1;
  }

  /*
  std::sort(words.begin(), words.end(),
      [](const rec & a, const rec & b) { return a.second.count > b.second.count; });
   */

  for(int i = 0; i < N; i++) {
    words[i].second.newid = ++newid_ctr;
    top_words.insert(words[i]);
  }

  double norm = 0.0;
  std::cin >> norm;

  int i = 0, j = 0;
  double rel = 0.0;

  double minrel = INFINITY;
  double maxrel = 0.0;

  while(std::cin >> i >> j >> rel) {
    auto i_it = top_words.find(i);
    auto j_it = top_words.find(j);
    if(i_it != top_words.end() && j_it != top_words.end()) {
      if(rel < minrel) {
        minrel = rel;
      }
      if(rel > maxrel) {
        maxrel = rel;
      }
      mat.insert(std::make_pair(std::make_pair(i,j), rel));
    }
  }

  std::ofstream dict("dict");
  std::ofstream X("X");

  size_t nnz = 0;
  for(auto & rel : mat) {
    double newrel = rel.second / maxrel;
    if(newrel > 0.0000001) {
      auto i_it = top_words.find(rel.first.first);
      auto j_it = top_words.find(rel.first.second);
      if(i_it->second.newid == -1) {
        i_it->second.newid = ++newid_ctr;
      }
      if(j_it->second.newid == -1) {
	j_it->second.newid = ++newid_ctr;
      }
      X << i_it->second.newid << " " << j_it->second.newid << " " << (1000000.0 * newrel) << "\n";
      nnz++;
    }
  }

  for(auto & word : top_words) {
    if(word.second.newid != -1) {
      dict << word.second.newid << " " << word.second.word << "\n";
    }
  }

  std::cout << "minrel: " << minrel << ", maxrel: " << maxrel << "\n";
  std::cout << "N: " << newid_ctr << "\n";
  std::cout << "nnz: " << nnz << "\n";

  return 0;
}
