#include "Omega_h_adios2.hpp"
#include <adios2.h>
#include <iostream>
#include "Omega_h_inertia.hpp" //<inertia::Rib>
#include "Omega_h_timer.hpp"
#include "Omega_h_tag.hpp"
#include <map>

using namespace std;

namespace Omega_h {

long unsigned int comm_size, rank;

template <typename T>
void write_value(adios2::IO &io, adios2::Engine &writer, 
		T val, std::string &name, bool global)
{
  if (global)
  {
    adios2::Variable<T> bpData = io.DefineVariable<T>(name);
    writer.Put(bpData, val);
  }
  else // local
  {
    std::vector<T> myData;
    myData.push_back(val);

    adios2::Variable<T> bpData =
          io.DefineVariable<T>(name, {comm_size}, {rank},
          {1}, adios2::ConstantDims);
    writer.Put(bpData, myData.data());
  }
  //std::cout <<__func__<<" "<<name<<": value "<<val<<"\n";
}

template <typename T>
void read_value(adios2::IO &io, adios2::Engine &reader, 
	T *val, std::string &name, bool global)
{
  adios2::Variable<T> bpData = io.InquireVariable<T>(name);
  if (bpData) // found
  {
    if (global) reader.Get(bpData, val);
    else
    {
      bpData.SetSelection({{rank}, {1}});
      // read only the chunk corresponding to this rank 
      std::vector<T> myData;
      reader.Get(bpData, myData, adios2::Mode::Sync);
      *val = myData[0];
    }
  }
  //std::cout <<__func__<<" "<<name<<": "<<*val<<"\n";
}

template <typename T>
void write_array(adios2::IO &io, adios2::Engine &writer, Read<T> array, 
		std::string &name)
{
//  std::cout<<__func__<<" "<<name<<": value ";
  if( !array.exists() ) return;

  std::vector<T> myData;
  for (size_t i = 0; i<array.size(); ++i)
  {
     myData.push_back(array.data()[i]);
  //   cout<<array.data()[i]<<" ";
  }
  //std::cout<<"\n";

  const std::size_t Nx = myData.size();
  std::string subname = name + "_size";
  write_value(io, writer, Nx, subname);

  subname = name + "_data";
  adios2::Variable<T> bpData = 
	  io.DefineVariable<T>(
          subname, {comm_size * Nx}, {rank * Nx}, {Nx}, adios2::ConstantDims);
  writer.Put(bpData, myData.data());
}

template <typename T>
void read_array(adios2::IO &io, adios2::Engine &reader,
               Read<T> &array, std::string &name)
{
  // usage: Read<int32_t> array;
  // read_array(io, reader, array, string);
  
  std::string subname = name + "_size";
  size_t Nx=1;
  read_value(io, reader, &Nx, subname);

  HostWrite<T> array_(Nx);

 adios2::Variable<T> bpData = io.InquireVariable<T>(name+"_data");
  if (bpData) // means found
  {
    std::vector<T> myData;

    // read only the chunk corresponding to this rank
    bpData.SetSelection({{Nx * rank}, {Nx}});
    reader.Get(bpData, myData, adios2::Mode::Sync);
    for (LO x=0; x<(LO)Nx; ++x)
      array_.set(x, myData[x]);

    array=Read<T>(array_.write());

    /* let's print out
    std::cout<<__func__<<" "<<name<<": "; 
    for (int x=0; x<Nx; ++x)
      cout<<array[x]<<" ";
    std::cout << "\n";
    */
  }
}


void write_down(adios2::IO &io, adios2::Engine &writer, Mesh* mesh, int d, std::string pref)
{
  auto down = mesh->ask_down(d, d - 1);
  std::string name = pref+"down.ab2b_" + std::to_string(d);
  write_array(io, writer, down.ab2b, name);
  if (d > 1) {
      name=pref+"down.codes_"+ std::to_string(d);
      write_array(io, writer, down.codes, name);
    }
}

void read_down(adios2::IO &io, adios2::Engine &reader, Mesh* mesh, int d, std::string pref)
{
  std::string name = pref+"down.ab2b_" + std::to_string(d);
  Adj down;
  read_array( io, reader, down.ab2b, name);
  if (d > 1) {
    name=pref+"down.codes_"+ std::to_string(d);
    read_array( io, reader, down.codes, name);
  }
  mesh->set_ents(d, down);
}

void write_meta(adios2::IO &io, adios2::Engine &writer, Mesh* mesh, std::string pref)
{
  std::string name=pref+"family";
  write_value(io, writer, (int32_t)mesh->family(), name);
  name=pref+"dim"; write_value(io, writer, (int32_t)mesh->dim(), name);
  name=pref+"comm_size";  write_value(io, writer, mesh->comm()->size(), name);
  name=pref+"rank"; write_value(io, writer, mesh->comm()->rank(), name);
  name=pref+"parting"; write_value(io, writer, (int32_t)mesh->parting(), name);
  name=pref+"nghost_layers"; write_value(io, writer, (int32_t)mesh->nghost_layers(), name);
  auto hints = mesh->rib_hints();
  int32_t have_hints = (hints != nullptr);
  name=pref+"have_hints"; write_value(io, writer, (int32_t)have_hints, name);
  if (have_hints) {
    int32_t naxes = int32_t(hints->axes.size());
    name=pref+"naxes"; write_value(io, writer, naxes, name);
    for (auto axis : hints->axes) {
      for (Int i = 0; i < 3; ++i) 
      {
	name=pref+"axes_"+to_string(i); 
	write_value(io, writer, axis[i], name);
      }
    }
  }
}

// assumption: version>=7
void read_meta(adios2::IO &io, adios2::Engine &reader, Mesh* mesh, std::string pref)
{
  int32_t family, dim, commsize, commrank, parting, nghost_layers, have_hints, naxes;
  std::string name=pref+"family";
  read_value(io, reader, &family, name);
  mesh->set_family(Omega_h_Family(family));

  name=pref+"dim"; read_value(io, reader, &dim, name);
  mesh->set_dim(Int(dim));

  name=pref+"comm_size"; read_value(io, reader, &commsize, name);
  OMEGA_H_CHECK(mesh->comm()->size() == commsize);

  name=pref+"rank"; read_value(io, reader, &commrank, name);
  OMEGA_H_CHECK(mesh->comm()->rank() == commrank);

  name=pref+"parting"; read_value(io, reader, &parting, name);
  OMEGA_H_CHECK(parting == (OMEGA_H_ELEM_BASED) ||
                parting == (OMEGA_H_GHOSTED) ||
                parting == (OMEGA_H_VERT_BASED));


  name=pref+"nghost_layers"; read_value(io, reader, &nghost_layers, name);
  mesh->set_parting(Omega_h_Parting(parting), nghost_layers, false);

  name=pref+"have_hints"; read_value(io, reader, &have_hints, name);
  if (have_hints) { 
    name=pref+"naxes"; read_value(io, reader, &naxes, name);
    auto hints = std::make_shared<inertia::Rib>();
    for (I32 i = 0; i < naxes; ++i) 
    {
      Vector<3> axis;
      for (Int j = 0; j < 3; ++j)
      {
        name=pref+"axes_"+to_string(j); 
	double value;
	read_value(io, reader, &value, name);
	axis[j] = value;
      }
      hints->axes.push_back(axis);
    }
    mesh->set_rib_hints(hints);
  }
}

void write_tag(adios2::IO &io, adios2::Engine &writer, 
		TagBase const* tag, string &pre_name)
{
  std::string name = pre_name+"name";
  adios2::Variable<std::string> bpString = io.DefineVariable<std::string>(name);
  writer.Put(bpString, tag->name());

  name=pre_name+"ncomps";
  write_value(io, writer, (int32_t)tag->ncomps(), name, true);

  name=pre_name+"type";
  write_value(io, writer, (int8_t)tag->type(), name, true);


  auto class_ids = tag->class_ids();
  int32_t n_class_ids = 0;
  if (class_ids.exists()) {
    n_class_ids = class_ids.size();
  }

  name = pre_name+"n_class_ids";
  write_value(io, writer, n_class_ids, name, true);
  if (n_class_ids > 0) {
    name =pre_name+ "class_ids";
    write_array(io, writer, class_ids, name);
  }

  auto f = [&](auto type) {
    using T = decltype(type);
    name = pre_name+"data";
    write_array(io, writer, as<T>(tag)->array(), name);
  };
  apply_to_omega_h_types(tag->type(), std::move(f));
}

void read_tag(adios2::IO &io, adios2::Engine &reader, Mesh* mesh, 
	      int32_t d, string &pre_name)
{
  std::string name = pre_name+"name";
  adios2::Variable<std::string> bpString = io.InquireVariable<std::string>(name);
  std::string tag_name;
  reader.Get(bpString, tag_name);

  name = pre_name+"ncomps";
  int32_t ncomps;
  read_value(io, reader, &ncomps, name, true);

  name=pre_name+"type";
  int8_t type;
  read_value(io, reader, &type, name, true);

  name = pre_name+"n_class_ids";
  //TODO: read class id info for rc tag to file
  Read<int32_t> class_ids = {};
  int32_t n_class_ids;
    read_value(io, reader, &n_class_ids, name, true);
    if (n_class_ids > 0) {
      name = pre_name+"class_ids"; 
      read_array(io, reader, class_ids, name);
    }

  auto f = [&](auto t) {
    using T = decltype(t);
    Read<T> array;
    name = pre_name+"data";
    read_array(io, reader, array, name);
    if(is_rc_tag(tag_name)) {
      mesh->set_rc_from_mesh_array(d,ncomps,class_ids,tag_name,array);
    }
    else {
      mesh->add_tag(d, tag_name, ncomps, array, true);
    }
  };
  apply_to_omega_h_types(static_cast<Omega_h_Type>(type), std::move(f));

}

void write_tags(adios2::IO &io, adios2::Engine &writer, Mesh* mesh, int d, std::string pref)
{
  std::string name = pref+"ntags_" + to_string(d);
  write_value(io, writer, mesh->ntags(d), name, true);

  for (int32_t i = 0; i < mesh->ntags(d); ++i)
  {
    name = pref+"tag_" + to_string(d) + "_" + to_string(i) + "_";
    write_tag(io, writer, mesh->get_tag(d, i), name);
  }

  name = pref+"nrctags_" + to_string(d);
  write_value(io, writer, mesh->nrctags(d), name, true);

  int32_t i=0;
  for (const auto& rc_tag : mesh->get_rc_tags(d)) 
  {
    auto rc_postfix_found = ((rc_tag.get()->name()).find("_rc") != std::string::npos);
    OMEGA_H_CHECK(rc_postfix_found);
    const auto rc_mesh_tag = mesh->get_rc_mesh_tag_from_rc_tag(d, rc_tag.get());
    name = pref+"rctag_" + to_string(d) + "_" + to_string(i++) + "_";
    write_tag(io, writer, rc_mesh_tag.get(), name);
  }
}

void read_tags(adios2::IO &io, adios2::Engine &reader, Mesh* mesh, int d, std::string pref)
{
  int32_t ntags;
  std::string name = pref+"ntags_" + to_string(d);
  read_value(io, reader, &ntags, name, true);

  for (Int i = 0; i < ntags; ++i)
  {
    name = pref+"tag_" + to_string(d) + "_" + to_string(i) + "_";
    read_tag(io, reader, mesh, d, name);
  }

  name = pref+"nrctags_" + to_string(d);
  read_value(io, reader, &ntags, name, true);
  for (Int i = 0; i < ntags; ++i)
  {
    name = pref+"rctags_" + to_string(d) + "_" + to_string(i);
    read_tag(io, reader, mesh, d, name);
  }
}

void write_pbdry(adios2::IO &io, adios2::Engine &writer, Mesh* mesh, int d, std::string pref)
{
  if (mesh->comm()->size() == 1) return; 
  auto owners = mesh->ask_owners(d);
  std::string name = pref+"owner_"+to_string(d)+"_ranks";
  write_array(io, writer, owners.ranks, name);
  name = pref+"owner_"+to_string(d)+"_idxs";
  write_array(io, writer, owners.idxs, name);
}

void read_pbdry(adios2::IO &io, adios2::Engine &reader, Mesh* mesh, int d, std::string pref)
{
  if (mesh->comm()->size() == 1) return;
  Remotes owners;
  std::string name = pref+"owner_"+to_string(d)+"_ranks";
  read_array(io, reader, owners.ranks, name);
  name = pref+"owner_"+to_string(d)+"_idxs";
  read_array(io, reader, owners.idxs, name);
  mesh->set_owners(d, owners);
}

void write_sets(adios2::IO &io, adios2::Engine &writer, Mesh* mesh, std::string pref)
{
  std::string name = pref+"gclas_size";
  write_value(io, writer, (int32_t)mesh->class_sets.size(), name, true);

  int32_t i=0;
  for (auto& set : mesh->class_sets)
  {
    name=pref+"gclas_"+to_string(i)+"_name";
    adios2::Variable<std::string> bpString = io.DefineVariable<std::string>(name);
    writer.Put(bpString, set.first);

    name=pref+"gclas_"+to_string(i)+"_npairs";
    int32_t npairs = (int32_t)set.second.size();
    write_value(io, writer, npairs, name, true);

    HostWrite<int32_t> gclas_dim_(npairs);
    HostWrite<int32_t> gclas_id_(npairs);
    int32_t x=0;
    for (auto& pair : set.second) {
      gclas_dim_.set(x, pair.dim); 
      gclas_id_.set(x, pair.id);
      ++x;
    }

    Read<int32_t> gclas_dim=Read<int32_t>(gclas_dim_.write());
    Read<int32_t> gclas_id=Read<int32_t>(gclas_id_.write());

    name=pref+"gclas_"+to_string(i)+"_dim";
    write_array(io, writer, gclas_dim, name);
    name=pref+"gclas_"+to_string(i)+"_id";
    write_array(io, writer, gclas_id, name);
    ++i;
  }
}

void read_sets(adios2::IO & io, adios2::Engine &reader, Mesh* mesh, std::string pref)
{
  std::string name = pref+"gclas_size";
  int32_t n;
  read_value(io, reader, &n, name, true);

  for (int32_t i = 0; i < n; ++i) 
  {
    name=pref+"gclas_"+to_string(i)+"_name";
    adios2::Variable<std::string> bpString = io.InquireVariable<std::string>(name);
    std::string gclas_name;
    reader.Get(bpString, gclas_name);

    name=pref+"gclas_"+to_string(i)+"_npairs";
    int32_t npairs;
    read_value(io, reader, &npairs, name, true);

    Read<int32_t> gclas_dim = {};
    Read<int32_t> gclas_id = {};

    name=pref+"gclas_"+to_string(i)+"_dim";
    read_array(io, reader, gclas_dim, name);

    name=pref+"gclas_"+to_string(i)+"_id";
    read_array(io, reader, gclas_id,name);

    for (int32_t j = 0; j < npairs; ++j) {
      ClassPair pair;
      pair.dim=gclas_dim[j];
      pair.id=gclas_id[j];
      mesh->class_sets[gclas_name].push_back(pair);
    }
  }
}

void write_parents(adios2::IO &io, adios2::Engine &writer, Mesh* mesh, std::string pref)
{
  int32_t has_parents = mesh->has_any_parents();
  std::string name = pref+"has_parents";
  write_value(io, writer, has_parents, name);
  if (has_parents) 
  {
    for (int32_t d = 0; d <= mesh->dim(); ++d) 
    {
      auto parents = mesh->ask_parents(d);
      name = pref+"parent_"+to_string(d)+"_idx";
      write_array(io, writer, parents.parent_idx, name);
      name = pref+"parent_"+to_string(d)+"_codes";
      write_array(io, writer, parents.codes, name);
    }
  }
}

void read_parents(adios2::IO &io, adios2::Engine &reader, Mesh* mesh, std::string pref)
{
  int32_t has_parents;
  std::string name = pref+"has_parents";
  read_value(io, reader, &has_parents, name);
  if (has_parents) 
  {
    for (int32_t d = 0; d <= mesh->dim(); ++d) 
    {
      Parents parents;
      name = pref+"parent_"+to_string(d)+"_idx";
      read_array(io, reader, parents.parent_idx, name);
      name = pref+"parent_"+to_string(d)+"_codes";
      read_array(io, reader, parents.codes, name);
      mesh->set_parents(d, parents);
    }
  }
}

void _write_adios2(adios2::IO &io, adios2::Engine & writer,
                  Mesh* mesh, std::string pref)
{
  comm_size = mesh->comm()->size();
  rank = mesh->comm()->rank();

  write_meta(io, writer, mesh, pref);
  int32_t nverts = mesh->nverts();
  std::string name=pref+"nverts";
  write_value(io, writer, nverts, name);

  for (int32_t d = 1; d <= mesh->dim(); ++d)
    write_down(io, writer, mesh, d, pref);

  for (Int d = 0; d <= mesh->dim(); ++d)
  {
    write_tags(io, writer, mesh, d, pref);
    write_pbdry(io, writer, mesh, d, pref);
  }

  write_sets(io, writer, mesh, pref);
  write_parents(io, writer, mesh, pref);
}

void write_adios2(filesystem::path const& path, 
		  std::map<Mesh*, std::string>& mesh_map)
{
  std::map<Mesh*, std::string>::iterator it=mesh_map.begin();
  Mesh* mesh=it->first;

  if (path.extension().string() != ".bp" && can_print(mesh)) {
    std::cout
        << "it is strongly recommended to end Omega_h paths in \".bp\",\n";
    std::cout << "instead of just \"" << path << "\"\n";
  }
  filesystem::create_directory(path);

  adios2::ADIOS adios(mesh->comm()->get_impl());
  adios2::IO io = adios.DeclareIO("mesh-writer");
  std::string filename=path.c_str();

  adios2::Engine writer = io.Open(filename, adios2::Mode::Write);
  writer.BeginStep();
  for (; it!=mesh_map.end(); ++it)
    _write_adios2(io, writer, it->first, it->second);
  writer.EndStep();
  writer.Close();
}


void write_adios2(filesystem::path const& path, Mesh* mesh, std::string pref)
{
  std::map<Mesh*, std::string> mesh_map;
  mesh_map[mesh] = pref;
  write_adios2(path, mesh_map);
}

Mesh read_adios2(filesystem::path const& path, Library* lib, std::string pref)
{
  comm_size = lib->world()->size();
  rank = lib->world()->rank();

  adios2::ADIOS adios(lib->world()->get_impl());
  adios2::IO io = adios.DeclareIO("mesh-reader");

  comm_size = lib->world()->size();
  rank = lib->world()->rank();
  Mesh mesh(lib->world()->library());
  mesh.set_comm(lib->world());

  string filename = path.c_str();

  adios2::Engine reader = io.Open(filename, adios2::Mode::Read);
  reader.BeginStep();
  read_meta(io, reader, &mesh, pref);
  int32_t nverts;
  std::string name=pref+"nverts";
  read_value(io, reader, &nverts, name);
  mesh.set_verts(nverts);

  for (Int d = 1; d <= mesh.dim(); ++d)
    read_down(io, reader, &mesh, d, pref);

  for (Int d = 0; d <= mesh.dim(); ++d)
  {
    read_tags(io, reader, &mesh, d, pref);
    read_pbdry(io, reader, &mesh, d, pref);
  }

  read_sets(io, reader, &mesh, pref);
  read_parents(io, reader, &mesh, pref);

  reader.EndStep();
  reader.Close();  

  return mesh;
}

}  // end namespace Omega_h
