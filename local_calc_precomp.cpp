/**
 * \file local_calc_precomp.cpp
 * 
 * \brief This code will collect the subgraph for each test node and apply
 * some calculations on it using the precomputed data by precompute_local.
 *
 * In this code, for each node not only the connected nodes to the analyzed node
 * will be gathered but also the connections between those nodes. This way we
 * can find the subgraph for a node. Instead of calculating it's laplacian matrix
 * to get the SVD decomposition, the Eigenvalues and Eigenvectors precomputed and
 * saved by precompute_local will be used. This should speed the process considerably.
 */ 

#include <string>
#include <list>
#include <graphlab.hpp>
#include <boost/unordered_map.hpp>
#include <math.h>
//#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <assert.h>     /* assert */

using namespace graphlab;
//using namespace boost::numeric::ublas;
using namespace Eigen;

typedef boost::unordered_map<vertex_id_type, double> map;
typedef boost::unordered_map<vertex_id_type, unsigned int> int_map;
//typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
//typedef Matrix<double, Dynamic, 1>       VectorXd;

typedef struct {
    unsigned int user_id;
    float mse;
    unsigned int kk;
} result;

typedef struct {
    //std::vector<unsigned int> movie_list;
    int_map movie_list;
    std::vector<double> sigs_min;
    VectorXd eigen_values;
    MatrixXd eigen_vectors;
} user_precomp_data;

typedef boost::unordered_map<vertex_id_type, user_precomp_data> user_map;

// Global variables
bool verbose = false;
unsigned int comp_pct = 100; // Percentage of nodes where the computation will happen
user_map user_data;

/**
 * \brief The vertex data stores the movie rating information.
 */
struct vertex_data {
    
    /** \brief The ratings each user has given to the movie */
    map ratings;

    /** \brief Store the results: userID, MSE, K neighbours used for prediction */
    std::vector<result> res;

    vertex_data() { }
    vertex_data(map ratings): ratings(ratings) { }

    /** \brief Save the vertex data to a binary archive */
    void save(graphlab::oarchive& arc) const { 
        arc << ratings;
    }
    /** \brief Load the vertex data from a binary archive */
    void load(graphlab::iarchive& arc) { 
        arc >> ratings;
    }
}; // end of vertex data

typedef boost::unordered_map<vertex_id_type, vertex_data> vert_map;

/**
 * \brief The edge data stores the weights between movies.
 */
struct edge_data : public graphlab::IS_POD_TYPE {
    /**
     * \brief The type of data on the edge;
     *
     * \li *Train:* the observed value is correct and used in training
     * \li *Validate:* the observed value is correct but not used in training
     */
    enum data_role_type { TRAIN, VALIDATE  };
    
    /** \brief the observed value for the edge */
    double obs;
    
    /** \brief The train/validation/test designation of the edge */
    data_role_type role;
    
    /** \brief basic initialization */
    edge_data(double obs = 0, data_role_type role = TRAIN) :
    obs(obs), role(role) { }  
}; // end of edge data

typedef graphlab::distributed_graph<vertex_data, edge_data> graph_type;
typedef graph_type::vertex_type vertex_type;
typedef graph_type::edge_type   edge_type;

/**
 * \brief The graph loader function is a line parser used for
 * distributed graph construction.
 */
bool graph_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
    
    // Parse the line
    std::stringstream strm(line);
    graph_type::vertex_id_type va(-1), vb(-1);
    float weight;
    
    strm >> va >> vb >> weight;
    if (weight > 0.1)
        graph.add_edge(va, vb, edge_data(weight));

    return true; // successful load
} // end of graph_loader

bool graph_test_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
    
    // Parse the line
    std::stringstream strm(line);
    graph_type::vertex_id_type vt(-1), user(-1);
    float rating;
    map ratings;
    
    strm >> vt;
    
    while(1) {
        strm >> user >> rating;
        if (strm.fail())
            break;
        ratings[user] = rating;
    }
    if (ratings.size() >= 1) 
        graph.add_vertex(vt, vertex_data(ratings));

    return true; // successful load
} // end of graph_loader



class gather_type_neigh {
public:

    vert_map vertices;
    
    /** \brief basic default constructor */
    gather_type_neigh() { }
    
    gather_type_neigh(graph_type::vertex_id_type vv, vertex_data data) {
        vertices[vv] = data;
    }
    
    /** \brief Save the values to a binary archive */
    void save(graphlab::oarchive& arc) const { arc << vertices; }
    
    /** \brief Read the values from a binary archive */
    void load(graphlab::iarchive& arc) { arc >> vertices; }
    
    /** 
     * \brief joins two neighs maps
     */
    gather_type_neigh& operator+=(const gather_type_neigh& other) {
        vert_map other_vertices = other.vertices;
        
        for (vert_map::iterator it = other_vertices.begin(); it != other_vertices.end(); ++it){
            vertices[it->first] = other_vertices[it->first];
        }
        return *this;
    } // end of operator+=

}; // end of gather type


/**
 * \brief Collect neighbour information on each vertex
 */
class neigh_program : 
    public graphlab::ivertex_program<graph_type, gather_type_neigh>,
    public graphlab::IS_POD_TYPE {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::OUT_EDGES; 
    }; // end of gather_edges 

    /** The gather function */
    gather_type_neigh gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        return gather_type_neigh(edge.target().id(), edge.target().data());
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type_neigh& sum) {
        
        std::vector<result> res; // Store all the results
        result res_usr; // Store user result
        double rat_real;
        vert_map vertex_neighs = sum.vertices; // Copy of neighbour map
        // Iterate over all the ratings of the movie
        for (map::iterator it = vertex.data().ratings.begin(); 
             it != vertex.data().ratings.end(); ++it) {
            
            unsigned usr = it->first;
            user_precomp_data user_data_usr = user_data[usr];
            rat_real = it->second;
            VectorXd usr_rat(vertex_neighs.size(), 1);
            VectorXd vv(vertex_neighs.size(), 1);
            MatrixXd uu_hh(vertex_neighs.size(), vertex_neighs.size());
        
            MatrixXd uu = user_data_usr.eigen_vectors;
        
            unsigned movie_ind = user_data_usr.movie_list[vertex.id()];
            for (unsigned i = 0; i < uu.cols(); ++i)
                vv(i, 0) = uu(movie_ind, i);

            std::cout << "Iep VV:" << std::endl << vv << std::endl;
                
            unsigned ind = 0;
            // Iterate over all the movies rated by the same user
            for (int_map::iterator it2 = user_data_usr.movie_list.begin(); 
                 it2 != user_data_usr.movie_list.end(); ++it2) {
                
                // Check if movie is connected to the test (central) movie
                if (vertex_neighs.find(it2->first) != vertex_neighs.end()) {
                    usr_rat(ind, 0) = vertex_neighs[it2->first].ratings[it->first];
                    for (unsigned i = 0; i < uu.cols(); ++i)
                        uu_hh(ind, i) = uu(it2->second, i);
                    ind++;
                }
            }
            // Resize the matrices and vectors
            usr_rat.conservativeResize(ind);
            uu_hh.conservativeResize(ind, uu_hh.cols());
            
            double w_lim = user_data_usr.sigs_min[movie_ind];
            unsigned lim;
            VectorXd eigen_values = user_data_usr.eigen_values;
            for (lim = 0; lim < eigen_values.rows(); ++lim)
                if (eigen_values(lim, 0) > w_lim)
                    break;
                
            if (lim < 2)
                lim = 2;
            
            vv.resize(lim);
            uu_hh.resize(uu_hh.rows(), lim);
            
            // Compute rating prediction
            MatrixXd mm(lim, lim);
            mm = uu_hh.transpose() * uu_hh;
            
            double rat_mean = usr_rat.sum() / usr_rat.rows();
            VectorXd usr_rat_unmean = usr_rat.array() - rat_mean;
            //double rat_pred = vv.transpose() * mm.inverse() * uu_hh.transpose * (usr_rat_clean - rat_mean);
            double rat_pred = vv.transpose() * (mm.inverse() * (uu_hh.transpose() * usr_rat_unmean));
            rat_pred += rat_mean;
            
            // Set boundaries to the rating result (should be between 1 and 5)
            if (rat_pred > 5)
                rat_pred = 5;
            if (rat_pred < 1)
                rat_pred = 1;
            
            double err = pow(rat_real - rat_pred, 2);

            if (verbose && vertex.id() == 71) {
                std::cout << "==== Showing movieID: " << vertex.id() << " userID: " << usr << " ====" << std::endl;
                std::cout << "EigenVectors: " << std::endl << uu << std::endl;
                std::cout << "EigenValues: " << std::endl << eigen_values << std::endl;
                std::cout << "ratings: " << std::endl << usr_rat << std::endl;
                std::cout << "Real rat: " << rat_real << std::endl;
                //std::cout << "w_lim: " << w_lim << std::endl;
                std::cout << "vv: " << std::endl << vv << std::endl;
                std::cout << "uu_hh" << std::endl << uu_hh << std::endl;
                std::cout << "Pred rat: " << rat_pred << std::endl;
                //std::cout << "uu: " << std::endl << uu << std::endl;
                std::cout << std::endl << std::endl;
                assert(0);
            }
            // std::cout << err << " ";
            
            res_usr.user_id = usr;
            res_usr.mse = err;
            res_usr.kk = usr_rat.rows();
            res.push_back(res_usr);
        }
        vertex.data().res = res;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of als vertex program
    
/**
 * \brief Saves the results
 */
struct graph_writer {
    std::string save_vertex(graph_type::vertex_type vt) {
        std::stringstream strm;
        for (std::vector<result>::iterator it = vt.data().res.begin(); it != vt.data().res.end(); ++it) {
            strm << vt.id() << " " << it->user_id << " " 
                                   << it->mse << " "
                                   << it->kk << "\n";
        }
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer

void load_precomputed_data(std::string filename, user_map &usr_data) {
    std::cout << "Reading file: " << filename << std::endl;
    std::ifstream infile(filename.c_str());
    std::string line;
    unsigned int user_id, movie_id, kk, mm;
    double val, sig_min;
    //std::vector<unsigned int> movie_list;
    int_map movie_list;
    std::vector<double> sigs_min;
    VectorXd eigen_values;
    MatrixXd eigen_vectors;
    user_precomp_data user;

    unsigned int state = 0;
    while (std::getline(infile, line)) {
        if(!line.length()) 
            continue; //skip empty lines
        std::stringstream parseline(line);

        switch (state) {
            case 0: // UserID + MovieID list
                parseline >> user_id >> kk >> mm;
                //std::cout << user_id << " " << kk << " " << mm << std::endl;
                eigen_values.resize(mm);
                eigen_vectors.resize(kk, mm);
                movie_list.clear();
                for (unsigned i = 0; i < kk; ++i) {
                    parseline >> movie_id >> sig_min;
                    assert(!parseline.fail());                        
                    //movie_list.push_back(movie_id);
                    movie_list[movie_id] = i;
                    sigs_min.push_back(sig_min);
                }
                user.movie_list = movie_list;
                user.sigs_min = sigs_min;
                state = 1;
                break;

            case 1: // EigenValues
                for (unsigned i = 0; i < mm; ++i) {
                    parseline >> val;
                    assert(!parseline.fail());
                    eigen_values(i, 0) = val;
                }
                user.eigen_values = eigen_values;
                state = 2;
                break;

            case 2: // EigenVectors
                //for (unsigned i = 0; i < mm*kk; ++i) {
                //    parseline >> val;
                //    eigen_vectors(i / mm, i % mm) = val;
                //}
                for (unsigned i = 0; i < kk; ++i) {
                    for (unsigned j = 0; j < mm; ++j) {
                        parseline >> val;
                        assert(!parseline.fail());
                        //std::cout << val << " ";
                        eigen_vectors(i, j) = val;
                    }
                }
                user.eigen_vectors = eigen_vectors;
                if (verbose) {
                    std::cout << "Eigen Values:" << std::endl << eigen_values << std::endl;
                    std::cout << "Eigen Vectors:" << std::endl << eigen_vectors << std::endl;
                }
                usr_data[user_id] = user;
                state = 0;
                //std::cout << user.movie_list.size() << " " << std::endl;
                //std::cout << "Eigen Vectors: " << std::endl << eigen_vectors << std::endl;
                break;
        }

    }
    infile.close();
}

int main(int argc, char** argv) {
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;

    /* initialize random seed: */
    srand (time(NULL));
    
    
    // Parse command line options
    graphlab::command_line_options clopts("Local graph computation.");
    clopts.attach_option("pct", comp_pct, "Percentage of nodes used for computation. Required ");
    clopts.add_positional("pct");
    int verbosity = 0;
    clopts.attach_option("verbosity", verbosity, "Enable verbosity.");
    clopts.add_positional("verbosity");
    if (!clopts.parse(argc, argv)) {
        dc.cout() << "Error in parsing command line arguments." << std::endl;
        return EXIT_FAILURE;
    }
    if (verbosity == 1)
        verbose = true;

    dc.cout() << "Loading precomputed data." << std::endl;

    load_precomputed_data("out_eigen_", user_data);

    dc.cout() << "Loading graph." << std::endl;
    graphlab::timer timer; 
    graph_type graph(dc);
    // Load the graph containing the weights and connections
    graph.load("out_fin_", graph_loader);
    // Load the test user ratings (not used to build the graph)
    graph.load("out_test_rat_", graph_test_loader); 
    dc.cout() << "Loading graph. Finished in " 
    << timer.current_time() << std::endl;
    
    dc.cout() << "Finalizing graph." << std::endl;
    timer.start();
    graph.finalize();
    dc.cout() << "Finalizing graph. Finished in " 
              << timer.current_time() << std::endl;
    
    dc.cout() 
        << "========== Graph statistics on proc " << dc.procid() 
        << " ==============="
        << "\n Num vertices: " << graph.num_vertices()
        << "\n Num edges: " << graph.num_edges()
        << "\n Num replica: " << graph.num_replicas()
        << "\n Replica to vertex ratio: " 
        << float(graph.num_replicas())/graph.num_vertices()
        << "\n --------------------------------------------" 
        << "\n Num local own vertices: " << graph.num_local_own_vertices()
        << "\n Num local vertices: " << graph.num_local_vertices()
        << "\n Replica to own ratio: " 
        << (float)graph.num_local_vertices()/graph.num_local_own_vertices()
        << "\n Num local edges: " << graph.num_local_edges()
        //<< "\n Begin edge id: " << graph.global_eid(0)
        << "\n Edge balance ratio: " 
        << float(graph.num_local_edges())/graph.num_edges()
        << std::endl;
        
    dc.cout() << "Creating engine 1" << std::endl;
    graphlab::omni_engine<neigh_program> engine(dc, graph, "sync");
        
    engine.signal_all();
        
    // Run neighbour information gathering
    dc.cout() << "Running ..." << std::endl;
    timer.start();
    engine.start();

    const double runtime = timer.current_time();
    dc.cout() << "----------------------------------------------------------"
            << std::endl
            << "Final Runtime (seconds):   " << runtime 
            << std::endl
            << "Updates executed: " << engine.num_updates() << std::endl
            << "Update Rate (updates/second): " 
            << engine.num_updates() / runtime << std::endl;


    /* engine.add_vertex_aggregator<float>("error",
                                        error_vertex_data,
                                        print_finalize);
    engine.aggregate_now("error"); */

    // Save the results into a file -------------------------------------------------
    graph.save("out_res", graph_writer(),
                false,    // do not gzip
                true,     // save vertices
                false);   // do not save edges
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
