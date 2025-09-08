#include <Omega_h_element.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_tag.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_cmdline.hpp>
#include <iomanip>

template <typename T>
void printTagInfo(Omega_h::Mesh mesh, std::ostringstream& oss, int dim, int tag, std::string type) {
    auto tagbase = mesh.get_tag(dim, tag);
    auto array = Omega_h::as<T>(tagbase)->array();

    Omega_h::Real min = get_min(array);
    Omega_h::Real max = get_max(array);

    oss << std::setw(18) << std::left << tagbase->name().c_str()
        << std::setw(5) << std::left << dim 
        << std::setw(7) << std::left << type 
        << std::setw(5) << std::left << tagbase->ncomps() 
        << std::setw(10) << std::left << min 
        << std::setw(10) << std::left << max 
        << "\n";
}

template <typename T>
int getNumEq(Omega_h::Mesh mesh, std::string tagname, int dim, int value) {
    auto array = mesh.get_array<T>(dim, tagname);
    auto each_eq_to = Omega_h::each_eq_to<T>(array, value);
    return Omega_h::get_sum(each_eq_to);
}

void printMPIChar(std::string output, int comm_size, int comm_rank) {
#ifdef OMEGA_H_USE_MPI
    if (comm_rank) {
        MPI_Send(output.c_str(), output.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else for (int sender=1; sender < comm_size; sender++) {
        std::cout << output;
        MPI_Status status;
        MPI_Probe(sender, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        char* buf = new char[count];
        MPI_Recv(&buf, count, MPI_CHAR, sender, 0, MPI_COMM_WORLD, &status);
        std::cout << buf;
        delete [] buf;
    }
#else
    std::cout << output;
#endif
}

int main(int argc, char** argv)
{
    auto lib = Omega_h::Library(&argc, &argv);
    auto comm = lib.world();

    Omega_h::CmdLine cmdline;
    cmdline.add_arg<std::string>("mesh.osh");
    cmdline.add_flag("-v", "verbose");
    auto& tagInfoFlag = cmdline.add_flag("--tag-info", "space seperated \"name dim value\"");
    tagInfoFlag.add_arg<std::string>("name");
    tagInfoFlag.add_arg<int>("dim");
    tagInfoFlag.add_arg<int>("value");
    if (!cmdline.parse(comm, &argc, argv) ||
        !Omega_h::CmdLine::check_empty(comm, argc, argv)) {
        cmdline.show_help(comm, argv);
        return -1;
    }

    std::string meshPath = cmdline.get<std::string>("mesh.osh");
    Omega_h::Mesh mesh = Omega_h::read_mesh_file(meshPath, lib.world());

    const int rank = comm->rank();

    std::array<Omega_h::GO, 4> counts;
    for(int dim=0; dim <= mesh.dim(); dim++)
       counts[dim] = mesh.nglobal_ents(dim);

    std::array<double, 4> imb;
    for(int dim=0; dim <= mesh.dim(); dim++)
       imb[dim] = mesh.imbalance(dim);

    std::ostringstream oss;
    // always print two places to the right of the decimal
    // for floating point types (i.e., imbalance)
    oss.precision(2);
    oss << std::fixed;

    if(!rank) {
        oss << "\nMesh Entity Type: " << Omega_h::topological_singular_name(mesh.family(), mesh.dim()) << "\n";

        oss << "\nGlobal Mesh Entity Count and Imbalance (max/avg): (Dim, Entity Count, Imbalance)\n";
        for(int dim=0; dim <= mesh.dim(); dim++)
            oss << "(" << dim << ", " << counts[dim] << ", " << imb[dim] << ")\n";

        oss << "\nTag Properties by Dimension: (Name, Dim, Type, Number of Components, Min. Value, Max. Value)\n";
        for (int dim=0; dim <= mesh.dim(); dim++)
        for (int tag=0; tag < mesh.ntags(dim); tag++) {
            auto tagbase = mesh.get_tag(dim, tag);
            if (tagbase->type() == OMEGA_H_I8)
                printTagInfo<Omega_h::I8>(mesh, oss, dim, tag, "I8");
            if (tagbase->type() == OMEGA_H_I32)
                printTagInfo<Omega_h::I32>(mesh, oss, dim, tag, "I32");
            if (tagbase->type() == OMEGA_H_I64)
                printTagInfo<Omega_h::I64>(mesh, oss, dim, tag, "I64");
            if (tagbase->type() == OMEGA_H_F64)
                printTagInfo<Omega_h::Real>(mesh, oss, dim, tag, "F64");
        }

        std::cout << oss.str();
    }

    if(cmdline.parsed("-v")) {
        comm->barrier(); // write the per-part data at the end
        if(!rank) {
          std::cout << "\nPer Rank Mesh Entity Count: (Rank: Entity Count by Dim <0,1,2,3>)\n";
        }
        oss.str(""); // clear the stream

        std::array<Omega_h::LO, 4> counts = {0,0,0,0};
        for(int dim=0; dim <= mesh.dim(); dim++)
            counts[dim] = mesh.nents(dim);
        oss << "(" << rank << ": " << counts[0] << ", "
                                        << counts[1] << ", "
                                        << counts[2] << ", "
                                        << counts[3] << ")\n";
        printMPIChar(oss.str(), comm->size(), rank);

        comm->barrier();
        if (!rank) std::cout << "\nPer Rank Mesh Entity Owned: (Rank: Entity Owned by Dim <0,1,2,3>)\n";
        oss.str(""); // clear the stream
        for (int dim=0; dim <= mesh.dim(); dim++)
            counts[dim] = mesh.nents_owned(dim);
        oss << "(" << rank << ": " << counts[0] << ", "
                                        << counts[1] << ", "
                                        << counts[2] << ", "
                                        << counts[3] << ")\n";
        printMPIChar(oss.str(), comm->size(), rank);
    }

    if (cmdline.parsed("--tag-info")) {
        std::string name = cmdline.get<std::string>("--tag-info", "name");
        int dim = cmdline.get<int>("--tag-info", "dim");
        int value = cmdline.get<int>("--tag-info", "value");
        auto tagbase = mesh.get_tagbase(dim, name);
        int numEq = 0;
        if (tagbase->type() == OMEGA_H_I8)
            numEq = getNumEq<Omega_h::I8>(mesh, name, dim, value);
        if (tagbase->type() == OMEGA_H_I32)
            numEq = getNumEq<Omega_h::I32>(mesh, name, dim, value);
        if (tagbase->type() == OMEGA_H_I64)
            numEq = getNumEq<Omega_h::I64>(mesh, name, dim, value);
        if (tagbase->type() == OMEGA_H_F64)
            numEq = getNumEq<Omega_h::Real>(mesh, name, dim, value);
        
        if (!rank) std::cout << "\nInput: (" << name << " " << dim << " " << value <<"), Output: (Rank, Num Entities Equal to Value)\n";
        comm->barrier();
        std::cout << "(" << rank << ", " << numEq << ")\n";
        return 0;
    }
    return 0;
}
