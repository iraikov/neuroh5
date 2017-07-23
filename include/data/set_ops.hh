#ifndef SET_OPS_H_
#define SET_OPS_H_

#include <set>
#include <algorithm>

using namespace std;

template <class  T>
bool operator==(const std::set<T> &A, const std::set<T> &B)
{

  if (A.size() != B.size())
    return false;

  // both set are of equal size.
  // check element by element
  typedef typename std::set<T>::const_iterator set_iter;

  set_iter pA = A.begin();
  set_iter pB = B.begin();
  for ( ; pA != A.end(); pA++, pB++)
  {
    if (*pA != *pB)
        return false;
  }
  return true;
}


template <class  T>
std::set<T> operator*(const std::set<T> &A, const std::set<T> &B)
{
   std::set<T> res;

   std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), 
        inserter(res, res.begin()));

  return res;
}


template <class T>
std::set<T> & operator+=(std::set<T> &A, const std::set<T> &B)
{
  // A.insert(B.begin(), B.end());
  for (typename std::set<T>::const_iterator p=B.begin(); p!=B.end(); p++)
    A.insert(*p);

  return A;
}

template <class T>
std::set<T> & operator-=(std::set<T> &A, const std::set<T> &B)
{
  // A.erase(B.begin(), B.end());
  for (typename std::set<T>::const_iterator p = B.begin(); p!=B.end(); p++)
    A.erase(*p);

  return A;
}


/**
    @return a new set, the union of A and B.
*/
template <class  T>
std::set<T> operator+(const std::set<T> &A, const std::set<T> &B)
{
   std::set<T> res;

   std::set_union(A.begin(), A.end(), B.begin(), B.end(), 
        inserter(res, res.begin()));

  return res;
}


/**
    @return the A - B: elements in A but not in B.
*/
template <class  T>
std::set<T> operator-(const std::set<T> &A, const std::set<T> &B)
{
  std::set<T> res;

  std::set_difference(A.begin(), A.end(), B.begin(), B.end(),
    inserter(res, res.begin()));

   return res;
}


/**
    @return a new (possibly empty) set, the symmetric difference of A and B.
    That is, elements in only one set, but not the other.  Mathematically,
    this is  A+B - (A*B)
*/
template <class  T>
std::set<T> symm_diff(const std::set<T> &A, const std::set<T> &B)
{
  std::set<T> res;

  std::set_symmetric_difference(A.begin(), A.end(), B.begin(), B.end(),
    inserter(res, res.begin()));

   return res;
}

/**
    @return true, if element a is in set A
*/

template <class T, class constT>
inline bool includes_elm( const std::set<T> &A, constT & a)
{
    return  (  (A.find(a) != A.end()) ? 
							 true : false );
}


//  NOTE: This algorithm asuumes the std::set is ordered.
//        It runs in O(n+m) steps.
//
// this is an optimzied version of intersect which 
// merely *counts* the size of the intersection,
// rather than explicitly create it.  (Saves a lot
// needless copying of elements.)
//
// NOTE: if m >> n, then use big_small_intersection_size below.

// NOTE: the O(n+m) algorithm works best if m and n are
// approximately equal.  If one is much bigger than the other,
// then a better approach is to look each element of the smaller
// set individually.  This runs in O(n log m), where m >> n.
//
template <class T>
int intersection_size( const std::set<T> &A, const std::set<T> &B)
{
  int res = 0;

  typename std::set<T>::const_iterator first1 = A.begin(),
                        last1  = A.end(),
                        first2 = B.begin(),
                        last2  = B.end();

  for (; first1 != last1 && first2 != last2 ;)
  {
    if ( *first1 < *first2)
      ++first1;
    else if ( *first2 < *first1 )
      ++first2;
    else
    {
       ++res;
       ++first1;
       ++first2;
    }
  }

  return res;
}

// This is a version of set intersection that is  optimized
// for the case where m >> n.  It runs in O(n log m), rather
// than O(n+m) steps.
//
//  It is assumed that A is the much larger set.
//
template <class T>
int big_small_intersection_size( const std::set<T> &A, const std::set<T> &B)
{
  int res = 0; 
  typename std::set<T>::const_iterator first=B.begin(), last=B.end();
  for(; first != last; first++)
  {
     if (includes_elm(A, *first)) res++;
  }
  return res;
}


//  NOTE: This algorithm asuumes the std::set is ordered.
//        It runs in O(n+m) steps.
//
// this is an optimzied version of Union which 
// merely *counts* the size of the std::set union,
// rather than explicitly create it.  (Saves a lot
// needless copying of elements.)
//
template <class T>
int union_size( const std::set<T> &A, const std::set<T> &B)
{
  int res = 0;

  typename std::set<T>::const_iterator first1 = A.begin(),
                        last1  = A.end(),
                        first2 = B.begin(),
                        last2  = B.end();

  for (; first1 != last1 && first2 != last2 ;)
  {
    if ( *first1 < *first2)
    {
      ++res;
      ++first1;
    }
    else if ( *first2 < *first1 )
    {
      ++res;
      ++first2;
    }
    else
    {
       ++res;
       ++first1;
       ++first2;
    }
  }

  return res;
}

template <class T>
int set_difference_size( const std::set<T> &A, const std::set<T> &B)
{

  return (A.size() - intersection_size(A,B)) ;
}

#endif
// SET_OPS_H_
