#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost>

void list_files_with_suffix(const std::string& pathname,
                            const std::string& suffix,
                            std::vector<std::string>& files) {
    namespace fs = boost::filesystem;
    fs::path dir_path(pathname);
    fs::directory_iterator end_iter;
    files.clear();
    if (fs::exists(dir_path) && fs::is_directory(dir_path)) {
        for (fs::directory_iterator dir_iter(dir_path); 
                dir_iter != end_iter; ++dir_iter) {
            if (fs::is_regular_file(dir_iter->status())) {
#if BOOST_FILESYSTEM_VERSION >= 3 
                const std::string filename = dir_iter->path().filename().string();
#else
                const std::string filename = dir_iter->leaf();
#endif
                if (suffix.size() > 0 && !boost::ends_with(filename, suffix)) 
                    continue;
                files.push_back(filename);
            }
        }
    }
    std::sort(files.begin(), files.end());
} // end of list files with suffix  


int main () {
    
    std::vector<std::string> files;
    list_files_with_suffix("movielens/", ".train", files);
    std::string line;
    /*while (getline(file, line)) {
        if(!line.length()) 
            continue; //skip empty lines
        std::stringstream parseline = std::stringstream(line);
        unsigned int user_id, movie_id, rating;
        parseline >> user_id >> movie_id >> rating;
        std::cout << user_id << " " << movie_id << " " << rating << std::endl;
    }*/

    return 0;
}


