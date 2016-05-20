/* a six-bit code describes the alignment relationship
   between a simplex and a lower-dimensional simplex
   on its boundary:

   "which_down" given a canonical ordering of the
   lower-dimensional simplices on the boundary, which
   one is this one ?
   (3 bits)

The other two pieces describe alignment between two
representations of the same simplex.

  "rotation" curl-aligned, counterclockwise rotation
  (indices move up in the canonical ordering).
  0 1 2 -> 2 0 1 -> 1 2 0
  (2 bits)

  "is_flipped" only applies to triangles, swap
  the last two vertices.
  0 1 2 -> 0 2 1
  (1 bit)

We define the rotation to take place first, so a
code containing both a flip and rotation means
that to get from one entity to another one must
first rotate and then flip the vertex list.
*/

INLINE I8 make_code(bool is_flipped, Int rotation, Int which_down) {
  return static_cast<I8>((which_down << 3) | (rotation << 1) | is_flipped);
}

INLINE bool code_is_flipped(I8 code) {
  return code & 1;
}

INLINE Int code_rotation(I8 code) {
  return (code >> 1) & 3;
}

INLINE Int code_which_down(I8 code) {
  return (code >> 3);
}

template <Int deg>
INLINE Int rotate_index(Int index, Int rotation) {
  return (index + rotation) % deg;
}

/* all the following can probably be optimized
   down to a few integer ops by an expert... */

template <Int deg>
INLINE Int flip_index(Int index) {
  switch(index) {
    case 1: return 2;
    case 2: return 1;
    default: return 0;
  }
}

template <Int deg>
INLINE Int align_index(Int index, I8 code) {
  index = rotate_index<deg>(index, code_rotation(code));
  if (code_is_flipped(code))
    index = flip_index<deg>(index);
  return index;
}

INLINE Int align_index(Int deg, Int index, I8 code) {
  if (deg == 3)
    return align_index<3>(index, code);
  if (deg == 2)
    return align_index<2>(index, code);
  NORETURN(0);
}

template <Int deg>
INLINE Int invert_rotation(Int rotation) {
  return (deg - rotation) % deg;
}

template <Int deg>
INLINE Int rotation_to_first(Int new_first) {
  return invert_rotation<deg>(new_first);
}

template <Int deg>
INLINE I8 invert_alignment(I8 code) {
  if (code_is_flipped(code))
    return code; // I think flipped codes are their own inverses
  return make_code(false,
      invert_rotation<deg>(code_rotation(code)), 0);
}

INLINE I8 invert_alignment(Int deg, I8 code) {
  if (deg == 3)
    return invert_alignment<3>(code);
  if (deg == 2)
    return invert_alignment<2>(code);
  NORETURN(0);
}

/* returns the single transformation equivalent
   to applying the (code1) transformation followed
   by the (code2) one. */
template <Int deg>
INLINE I8 compound_alignments(I8 code1, I8 code2) {
  /* we can look for the inverse of the compound
     by looking at what happens to the index
     that used to be first (0) */
  Int old_first = align_index<deg>(align_index<deg>(0, code1), code2);
  /* the inverse transformation would bring that
     index back to being the first */
  Int rotation = rotation_to_first<deg>(old_first);
  bool is_flipped = (code_is_flipped(code1) ^ code_is_flipped(code2));
  return invert_alignment<deg>(make_code(is_flipped, rotation, 0));
}

template <Int deg, typename T>
INLINE void rotate_adj(Int rotation,
    T const in[], T out[]) {
  for (I8 j = 0; j < deg; ++j)
    out[rotate_index<deg>(j, rotation)] = in[j];
}

template <typename T>
INLINE void flip_adj(T adj[]) {
  swap2(adj[1], adj[2]);
}

template <Int deg, typename T>
INLINE void align_adj(I8 code,
    T const in[], T out[]) {
  rotate_adj<deg>(code_rotation(code), in, out);
  if (code_is_flipped(code))
    flip_adj(out);
}
