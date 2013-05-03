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
#include <boost/thread.hpp>
#include <boost/threadpool.hpp>
#include <boost/thread/mutex.hpp> 


using namespace Eigen;
using namespace boost::threadpool;

typedef boost::unordered_map<unsigned int, unsigned int> map_rat;
typedef boost::unordered_map<unsigned int, map_rat> map_usr;

unsigned int uimax = std::numeric_limits<int>::max();
boost::mutex out_monitor;
MatrixXd weights;
unsigned int n_users_done = 0, n_users_total;

std::string matrix_to_matlab(const MatrixXd mm) {
    std::stringstream strm;
    strm << "[ ";
    for (unsigned i = 0; i < mm.rows(); ++i) {
        for (unsigned j = 0; j < mm.cols(); ++j) {
            strm << mm(i, j) << " ";
        }
        if (i < mm.rows() - 1)
            strm << ";";
    }
    strm << "]; ";
    return strm.str();
}

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

void save_output(std::string filename, std::stringstream &strm) {
    boost::mutex::scoped_lock lock(out_monitor);

    std::ofstream offile(filename.c_str(), std::ofstream::out | std::ofstream::app);
    offile << strm.str();
    offile.close();

    n_users_done++;
    std::cout << (float)n_users_done / n_users_total * 100 << "%" << std::endl;
}

void compute_eigens(unsigned int user_id, map_rat ratings) {
    
    unsigned int movie_id;
    MatrixXd ww(10, 10);
    
    std::vector<unsigned> movie_list;
    std::vector<double> sigs_min;

    // Retrieve list of movies rated by the user
    for (map_rat::iterator it2 = ratings.begin(); it2 != ratings.end(); ++it2){
        movie_id = it2->first;
        //std::cout << movie_id << " ";
        movie_list.push_back(movie_id);
    }
    unsigned int mat_size = movie_list.size();
    ww.resize(mat_size, mat_size);

    // Save adjacency matrix W
    for (unsigned i = 0; i < movie_list.size(); ++i) {
        for (unsigned j = 0; j < movie_list.size(); ++j) {
            if (std::max(movie_list[i], movie_list[j]) >= weights.rows())
                ww(i, j) = 0;
            else
                ww(i, j) = weights(movie_list[i], movie_list[j]);
        }
    }
    //std::cout << ww << std::endl;

    // Calculate D: Diagonal Degree Matrix
    MatrixXd dd(mat_size, mat_size);
    dd.setZero();

    double count;
    for (unsigned i = 0; i < ww.rows();  ++i) {
        count = 0;
        for (unsigned j = 0; j < ww.cols();  ++j)
            count += ww(i, j);
        if (count == 0)
            dd(i, i) = 1;
        else
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
    //if (user_id == 2147483333) {
    //std::cout << matrix_to_matlab(ww);
    //std::cout << ll << std::endl << std::endl;
    //std::cout << ll2 << std::endl << std::endl;
    //}

    // SVD descomposition
    // !! This Matrix is big, this operation takes time
    SelfAdjointEigenSolver<MatrixXd> es(ll2);
    MatrixXd eigen_vectors = es.eigenvectors();
    VectorXd eigen_values = es.eigenvalues();

    // Find the maximum sig_min
    float sig_min_max = 0;
    for (unsigned i = 0; i < ll2.rows(); ++i) {

        float sig_min = 0;
        for (unsigned j = 0; j < ll2.cols(); ++j) {
            sig_min += pow(ll2(i, j), 2);
        }
        sig_min = std::sqrt(sig_min);
        sigs_min.push_back(sig_min + 0.01);

        if (sig_min_max < sig_min)
            sig_min_max = sig_min;
    }
    sig_min_max += 0.01;

    // Find how many eigenvectors-eivenvalues to store
    unsigned int lim;
    for (lim = 0; lim < eigen_values.rows(); ++lim)
        if (eigen_values(lim, 0) > sig_min_max)
            break;

    if (lim < 2)
        lim = 2;

    eigen_values.conservativeResize(lim);
    eigen_vectors.conservativeResize(eigen_vectors.rows(), lim);

    std::stringstream strm;
    strm << user_id << " " << movie_list.size() << " " << eigen_values.rows() << " ";
    for (unsigned i = 0; i < movie_list.size(); ++i)
        strm << movie_list[i] << " " << sigs_min[i] << " ";
    strm << std::endl;
    for (unsigned i = 0; i < eigen_values.rows(); ++i)
        strm << eigen_values(i, 0) << " ";
    strm << std::endl;
    //std::cout << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
    for (unsigned i = 0; i < eigen_vectors.rows(); ++i) {
        for (unsigned j = 0; j < eigen_vectors.cols(); ++j) {
            strm << eigen_vectors(i, j) << " ";
        }
    }
    strm << std::endl;
    save_output("out_eigen_", strm);
    //std::cout << strm.str();
}

int main (int argc, const char* argv[]) {
    
    if (argc < 2) {
        std::cout << "Usage:\n" << argv[0] << " n_threads\n";
        return 1;
    }

    unsigned int n_threads = atoi(argv[1]);

    std::string path = "movielens/";
    std::string suffix = ".validate";
    
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
    n_users_total = users.size();
    // Load the movie graph data (weights between movies)
    std::string prefix = "out_fin_";

    unsigned int init_size = 2000; // This should be >= to the number of movies
    unsigned int fin_size = 0;
    weights.resize(init_size, init_size);
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
            // !! This is unsafe because the new values of the matrix will be uninitialized
            //if (max_val >= weights.rows())
            //    weights.conservativeResize(max_val * 2, max_val * 2);
            if (max_val > fin_size)
                fin_size = max_val;

            weights(movie_id1, movie_id2) = weight;
        }
        infile.close();        
    }
    
    std::ofstream offile("out_eigen_", std::ofstream::out);
    offile.close();
    
    // The column and row 0 will not be used: movie_ID starts from 1 to fin_size
    weights.conservativeResize(fin_size + 1, fin_size + 1);
    //std::cout << weights << std::endl;

    std::cout << "Number of movies: " << fin_size << std::endl;
    std::cout << "Number of users: " << users.size() << std::endl;

    // Use 4 threads
    pool tp(n_threads); // tp is handle to the pool

    // Compute Laplacian matrix for each user sub-graph
    unsigned int user_id;
    map_rat ratings;

    for (map_usr::iterator it = users.begin(); it != users.end(); ++it){
       
        user_id = it->first;
        ratings = it->second;
        
        tp.schedule(boost::bind(compute_eigens, user_id, ratings));
    }

    tp.wait();

    return 0;
}
