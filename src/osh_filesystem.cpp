#include <filesystem>
#include <cstring>
#include <iostream>



int main(int argc, char** argv) {
  if (argc == 2 && 0 == strcmp(argv[1], "pwd")) {
    std::cout << std::filesystem::current_path() << '\n';
  } else if (argc == 2 && 0 == strcmp(argv[1], "ls")) {
    for (std::filesystem::directory_iterator
             it(std::filesystem::current_path()),
         end;
         it != end; ++it) {
      std::cout << it->path() << '\n';
    }
  } else if (argc == 3 && 0 == strcmp(argv[1], "rm")) {
    std::filesystem::remove(argv[2]);
  } else if (argc == 4 && 0 == strcmp(argv[1], "rm") &&
             0 == strcmp(argv[2], "-r")) {
    std::filesystem::remove_all(argv[3]);
  } else {
    std::cout << "usage: osh_filesystem pwd\n";
    std::cout << "       osh_filesystem ls\n";
    std::cout << "       osh_filesystem rm <file>\n";
    std::cout << "       osh_filesystem rm -r <dir>\n";
  }
}
