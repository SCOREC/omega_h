#ifndef OMEGA_H_TAG_HPP
#define OMEGA_H_TAG_HPP

#include <unordered_map>
#include <Omega_h_array.hpp>
#ifdef OMEGA_H_USE_MPI
#include <mpi.h>
#endif

namespace Omega_h {

inline void check_tag_name(std::string const& name) {
  OMEGA_H_CHECK(!name.empty());
}

enum class ArrayType {
  NotSpecified,
  VectorND, // vector with N components
  SymmetricSquareMatrix, // symmetric matrix with dim*(dim+1)/2 components
};

inline void check_array_type(ArrayType array_type) {
#ifndef NDEBUG 
  int rank = 0;
#ifdef OMEGA_H_USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank == 0) {
    if (array_type == ArrayType::NotSpecified) {
      fprintf(stderr,
        "Warning: Tag array type is NotSpecified. This will be deprecated in a future version. The default type will become VectorND, which treats the array as an n-dimensional vector based on ncomponents. It is recommended to set a specific array type for better clarity and to avoid unexpected behavior.\n");
    }
  }

#endif
}

 const std::unordered_map<ArrayType, std::string> ArrayTypeNames = {
    {ArrayType::NotSpecified, "NotSpecified"},
    {ArrayType::VectorND, "VectorND"},
    {ArrayType::SymmetricSquareMatrix, "SymmetricSquareMatrix"}
};

const std::unordered_map<std::string, ArrayType> NamesToArrayType = {
    {"NotSpecified", ArrayType::NotSpecified},
    {"VectorND", ArrayType::VectorND},
    {"SymmetricSquareMatrix", ArrayType::SymmetricSquareMatrix}
};

class TagBase {
 public:
  TagBase(std::string const& name_in, Int ncomps_in);
  TagBase(std::string const& name_in, Int ncomps_in, LOs class_ids_in);
  TagBase(std::string const& name_in, Int ncomps_in, ArrayType array_type_in);
  TagBase(std::string const& name_in, Int ncomps_in, LOs class_ids_in,
      ArrayType array_type_in);
  virtual ~TagBase();
  std::string const& name() const;
  Int ncomps() const;
  virtual Omega_h_Type type() const = 0;
  LOs class_ids() const;
  ArrayType array_type() const;

 private:
  std::string name_;
  Int ncomps_;
  LOs class_ids_;
  ArrayType array_type_ = ArrayType::NotSpecified;
};

template <typename T>
class Tag : public TagBase {
 public:
  Tag(std::string const& name_in, Int ncomps_in);
  Tag(std::string const& name_in, Int ncomps_in, LOs class_ids_in);
  Tag(std::string const& name_in, Int ncomps_in, ArrayType array_type_in);
  Tag(std::string const& name_in, Int ncomps_in, LOs class_ids_in,
      ArrayType array_type_in);
  Read<T> array() const;
  void set_array(Read<T> array_in);
  virtual Omega_h_Type type() const override;

 private:
  Read<T> array_;
};

template <typename T>
bool is(TagBase const* t);

template <typename T>
Tag<T> const* as(TagBase const* t);
template <typename T>
Tag<T>* as(TagBase* t);

#define OMEGA_H_EXPL_INST_DECL(T)                                              \
  extern template bool is<T>(TagBase const* t);                                \
  extern template Tag<T> const* as<T>(TagBase const* t);                       \
  extern template Tag<T>* as<T>(TagBase * t);                                  \
  extern template class Tag<T>;
OMEGA_H_EXPL_INST_DECL(I8)
OMEGA_H_EXPL_INST_DECL(I32)
OMEGA_H_EXPL_INST_DECL(I64)
OMEGA_H_EXPL_INST_DECL(Real)
#undef OMEGA_H_EXPL_INST_DECL

}  // namespace Omega_h

#endif
