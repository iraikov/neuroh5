// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
//==============================================================================
///  @file sort_permutation.hh
///
///  Functions for sorting vectors according to a comparison
///  function, and applying to resulting index permutation to other
///  vectors.
///
///  Code based on Stack Overflow question
///  http://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of
///
///  Copyright (C) 2016-2017 Project NeuroH5.
//==============================================================================


#ifndef SORT_PERMUTATION_HH
#define SORT_PERMUTATION_HH

#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

#include "throw_assert.hh"


namespace neuroh5
{
  namespace data
  {
    // Given a std::vector<T> and a comparison for type T, returns
    // the permutation of indices that results from sorting the
    // input vector using the comparison.
    template <typename T, typename Compare>
    std::vector<std::size_t> sort_permutation(const std::vector<T>& vec,
                                              Compare& compare)
    {
      std::vector<std::size_t> p(vec.size());
      if (vec.size() > 0)
        {
          std::iota(p.begin(), p.end(), 0);
          std::sort(p.begin(), p.end(),
                    [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
        }
      return p;
    }


    // Given a std::vector<T> and a permutation, build a new
    // std::vector<T> that is reordered according to the permutation.
    template <typename T>
    std::vector<T> apply_permutation(const std::vector<T>& vec,
                                     const std::vector<std::size_t>& p)
    {
      std::vector<T> sorted_vec(vec.size());
      if (vec.size() > 0)
        {
          std::transform(p.begin(), p.end(), sorted_vec.begin(),
                         [&](std::size_t i){ return vec[i]; });
        }
      return sorted_vec;
    }

    // In-place permutation
    template <typename T>
    void apply_permutation_in_place(std::vector<T>& vec,
                                    const std::vector<std::size_t>& p)
    {
      std::vector<bool> done(vec.size());
      throw_assert(vec.size() == p.size(),
                   "apply_permutation_in_place: permutation and value vectors have different sizes");
      for (std::size_t i = 0; i < vec.size(); ++i)
        {
          if (done[i])
            {
              continue;
            }
          done[i] = true;
          std::size_t prev_j = i;
          std::size_t j = p[i];
          while (i != j)
            {
              std::swap(vec[prev_j], vec[j]);
              done[j] = true;
              prev_j = j;
              j = p[j];
            }
        }
    }


  }

}

#endif
