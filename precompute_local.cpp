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
#include <algorithm>    // std::max

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
        unsigned int user_id, movie_id, rating;

        while (std::getline(infile, line)) {
            if(!line.length()) 
                continue; //skip empty lines
            std::stringstream parseline(line);
            parseline >> user_id >> movie_id >> rating;
            user_id =  uimax - user_id;
            users[user_id][movie_id] = rating;
            // std::cout << user_id << " " << movie_id << " " << rating << std::endl;
        }
        infile.close();
    }
    // Load the movie graph data (weights between movies)
    std::string prefix = "out_fin_";

    unsigned int init_size = 1000;
    unsigned int fin_size = 0;
    MatrixXd weights(init_size, init_size);
    weights.setZero();
    files.clear();

    list_files_with_prefix("./", prefix, files);

    for(std::vector<std::string>::iterator it = files.begin(); it != files.end(); ++it) {
        std::string filename = *it;
        std::cout << "Reading file: " << filename << std::endl;
        std::ifstream infile(filename.c_str());
        std::string line;
        unsigned int movie_id1, movie_id2;
        double weight;

        while (std::getline(infile, line)) {
            if(!line.length()) 
                continue; //skip empty lines
            std::stringstream parseline(line);
            parseline >> movie_id1 >> movie_id2 >> weight;
            
            unsigned int max_val = std::max(movie_id1, movie_id2);
            if (max_val >= weights.rows())
                weights.conservativeResize(max_val * 2, max_val * 2);
            if (max_val > fin_size)
                fin_size = max_val;

            weights(movie_id1, movie_id2) = weight;
        }
        infile.close();        
    }
    weights.conservativeResize(fin_size, fin_size);
    std::cout << "Number of movies: " << fin_size << std::endl;

    // Compute Laplacian matrix for each user sub-graph
    unsigned int user_id, movie_id;
    MatrixXd ww;
    map_rat ratings;
    for (map_usr::iterator it = users.begin(); it != users.end(); ++it){
        user_id = it->first;
        ratings = it->second;
        ww.resize(ratings.size(), ratings.size());
        std::vector<double> movie_list(ratings.size());
        
        // Retrieve list of movies rated by the user
        for (map_rat::iterator it2 = ratings.begin(); it2 != ratings.end(); ++it2){
            movie_id = it->first;
            movie_list.push_back(movie_id);
        }

        // Save adjacency matrix W
        for (unsigned i = 0; i < movie_list.size(), ++i)
            for (unsigned j = 0; j < movie_list.size(), ++j)
                 ww(i, j) = weights(movie_list[i], movie_list[j]);
        
        // Calculate D: Diagonal Degree Matrix
        MatrixXd dd(mat_size, mat_size);
        dd.setZero();
          
        double count;
        for (unsigned i = 0; i < ww.rows();  ++i) {
            count = 0;
            for (unsigned j = 0; j < ww.cols();  ++j)
                count += ww(i, j);
            dd(i, i) = count;
        }
    
        // Calculate L: Laplacian Matrix
        MatrixXd ll(mat_size, mat_size);
        ll = dd - ww;

        // Calculate L2: Normalized Laplacian Matrix
        MatrixXd ll2(mat_size, mat_size); // Normalized Laplacian Matrix
        MatrixXd dd2 = dd.inverse(); // Store D^(-1/2)
        
        for (unsigned i = 0; i < dd.rows();  ++i)
            for (unsigned j = 0; j < dd.cols();  ++j)
                dd2(i, j) = sqrt(dd2(i, j));

        ll2 = dd2 * ll * dd2;


    }

    return 0;
}
