#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <limits>
#include <boost/unordered_map.hpp>
#include <Eigen/Dense>

using namespace Eigen;

typedef boost::unordered_map<unsigned int, unsigned int> map_rat;
typedef boost::unordered_map<unsigned int, map_rat> map_usr;

unsigned int uimax = std::numeric_limits<int>::max();

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
                files.push_back(pathname + filename);
            }
        }
    }
    std::sort(files.begin(), files.end());
} // end of list files with suffix  

void list_files_with_prefix(const std::string& pathname,
                            const std::string& prefix,
                            std::vector<std::string>& files) {
    namespace fs = boost::filesystem;  
    fs::path dir_path(pathname);
    fs::directory_iterator end_iter;
    files.clear();
    if ( fs::exists(dir_path) && fs::is_directory(dir_path)) {
        for( fs::directory_iterator dir_iter(dir_path) ; 
                dir_iter != end_iter ; ++dir_iter) {
            if (fs::is_regular_file(dir_iter->status()) ) {
                const std::string filename = dir_iter->path().filename().string();
                if (prefix.size() > 0 && !boost::starts_with(filename, prefix)) {
                    continue;
                }
                files.push_back(pathname + dir_iter->path().string());
            }
        }
    }
    std::sort(files.begin(), files.end());
} // end of list files with prefix



int main () {
    
    std::string path = "movielens/";
    std::string suffix = ".train";
    
    map_usr users;

    std::vector<std::string> files;
    list_files_with_suffix(path, suffix, files);
 
    // Load the test user rating data
    for(std::vector<std::string>::iterator it = files.begin(); it != files.end(); ++it) {
        std::string filename = *it;
        std::cout << "Reading file: " << filename << std::endl;
        std::ifstream infile(filename.c_str());
        std::string line;
    
        while (std::getline(infile, line)) {
            if(!line.length()) 
                continue; //skip empty lines
            std::stringstream parseline(line);
            unsigned int user_id, movie_id, rating;
            parseline >> user_id >> movie_id >> rating;
            user_id =  uimax - user_id;
            users[user_id][movie_id] = rating;
            // std::cout << user_id << " " << movie_id << " " << rating << std::endl;
        }
        infile.close();
    }
    // Load the movie graph data (weights between movies)
    std::string prefix = "out_fin_";
    MatrixXd weights(bla, bla);
    for ...
        unsigned int movie_id1, movie_id2;
        double weight;
        parseline >> movie_id1 >> movie_id2 >> weight;
        weights(movie_id1, movie_id2) = weight
    ...


    return 0;
}
