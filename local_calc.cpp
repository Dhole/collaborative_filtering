/**
 * \file local_calc.cpp
 * 
 * \brief This code will collect the subgraph for each test node and apply
 * some calculations on it.
 *
 * In this code, for each node not only the connected nodes to the analyzed node
 * will be gathered but also the connections between those nodes. This way we
 * can find the subgraph for a node and then calculate it's laplacian matrix, etc.
 */ 

#include <string>
#include <list>
#include <graphlab.hpp>
#include <boost/unordered_map.hpp>
#include <math.h>
//#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <Eigen/Dense>

using namespace graphlab;
//using namespace boost::numeric::ublas;
using namespace Eigen;

typedef boost::unordered_map<vertex_id_type, double> map;
typedef boost::unordered_map<vertex_id_type, unsigned int> int_map;
//typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
//typedef Matrix<double, Dynamic, 1>       VectorXd;

/**
 * \brief The vertex data stores the movie rating information.
 */
struct vertex_data {
    
    /** \brief The ratings each user has given to the movie */
    map ratings;
    
    /** \brief The information of the neighbours to the vertex */
    map neighs;

    vertex_data() { }
    vertex_data(map ratings): ratings(ratings) { }

    /** \brief Save the vertex data to a binary archive */
    void save(graphlab::oarchive& arc) const { 
        arc << ratings << neighs;
    }
    /** \brief Load the vertex data from a binary archive */
    void load(graphlab::iarchive& arc) { 
        arc >> ratings >> neighs;
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
    
    map neighs;
    
    /** \brief basic default constructor */
    gather_type_neigh() { }
    
    gather_type_neigh(graph_type::vertex_id_type vv, double obs) {
        neighs[vv] = obs;
    }
    
    /** \brief Save the values to a binary archive */
    void save(graphlab::oarchive& arc) const { arc << neighs; }
    
    /** \brief Read the values from a binary archive */
    void load(graphlab::iarchive& arc) { arc >> neighs; }
    
    /** 
     * \brief joins two neighs maps
     */
    gather_type_neigh& operator+=(const gather_type_neigh& other) {
        map other_neighs = other.neighs;
        
        for (map::iterator it = other_neighs.begin(); it != other_neighs.end(); ++it){
            neighs[it->first] = other_neighs[it->first];
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
        return gather_type_neigh(edge.target().id(), edge.data().obs);
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type_neigh& sum) {
        vertex.data().neighs = sum.neighs;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of als vertex program

class gather_type_2 {
public:
    
    vert_map vertices;
    
    /** \brief basic default constructor */
    gather_type_2() { }
    
    gather_type_2(graph_type::vertex_id_type vv, vertex_data data) {
        vertices[vv] = data;
    }
    
    /** \brief Save the values to a binary archive */
    void save(graphlab::oarchive& arc) const { arc << vertices; }
    
    /** \brief Read the values from a binary archive */
    void load(graphlab::iarchive& arc) { arc >> vertices; }
    
    /** 
     * \brief joins two neighs maps
     */
    gather_type_2& operator+=(const gather_type_2& other) {
        vert_map other_vertices = other.vertices;
        
        for (vert_map::iterator it = other_vertices.begin(); it != other_vertices.end(); ++it){
            vertices[it->first] = other_vertices[it->first];
        }
        return *this;
    } // end of operator+=
}; // end of gather type


/**
 * \brief Compute the KNN for each rating in the vertices
 */
class vertex_program : 
    public graphlab::ivertex_program<graph_type, gather_type_2>,
    public graphlab::IS_POD_TYPE {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::OUT_EDGES; 
    }; // end of gather_edges 

    /** The gather function */
    gather_type_2 gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        return gather_type_2(edge.target().id(), edge.target().data());
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type_2& sum) {
        int_map indices;
        int mat_size = sum.vertices.size() + 1;
        // W: Adjacency Matrix
        // MatrixXd ww(mat_size, mat_size);
        MatrixXd ww(mat_size, mat_size);
        
        for (unsigned i = 0; i < ww.rows();  ++i)
            for (unsigned j = 0; j < ww.cols();  ++j)
                ww(i, j) = 0;
        //std::cout << "check1" << std::endl;

        // Saves all the vertices of the local graph
        vert_map vertices = sum.vertices;
        vert_map indexed_vertices;
        map vertex_neighs = vertex.data().neighs;
        map neighs_neighs;
        
        
        // Map the vertices into indices
        unsigned int ind = 0;
        indices[vertex.id()] = ind;
        ind++;
        for (vert_map::iterator it = vertices.begin(); it != vertices.end(); ++it){
            indices[it->first] = ind;
            indexed_vertices[ind] = vertices[it->first];
            ind++;
        }
        
        // Add the main vertex to the map
        vertices[vertex.id()] = vertex.data();
        indexed_vertices[0] = vertex.data();
        
        // Calculate adjacency matrix
        for (map::iterator it = vertex_neighs.begin(); it != vertex_neighs.end(); ++it){
            ww(indices[it->first] , 0) = vertex_neighs[it->first];
        }
        //std::cout << "check2" << std::endl;
        for (map::iterator it = vertex_neighs.begin(); it != vertex_neighs.end(); ++it){
            neighs_neighs = vertices[it->first].neighs;
            for (map::iterator it2 = neighs_neighs.begin(); it2 != neighs_neighs.end(); ++it2){
                ww(indices[it2->first] , indices[it->first]) = neighs_neighs[it2->first];
            }
        }
        //std::cout << "check3" << std::endl;
        
        // Print the W matrix for testing purposes (Use only with very small datasets)
        /*for (unsigned i = 0; i < ww.rows();  ++i) {
            for (unsigned j = 0; j < ww.cols();  ++j) {
                std::cout << ww(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;*/

        // Calculate D: Diagonal Degree Matrix
        MatrixXd dd(mat_size, mat_size);
        for (unsigned i = 0; i < dd.rows();  ++i)
            for (unsigned j = 0; j < dd.cols();  ++j)
                dd(i, j) = 0;
          
        double count;
        for (unsigned i = 0; i < ww.rows();  ++i) {
            count = 0;
            for (unsigned j = 0; j < ww.cols();  ++j) {
                count += ww(i, j);
            }
            dd(i, i) = count;
        }

        // Calculate L: Laplacian Matrix
        MatrixXd ll(mat_size, mat_size);
        ll = dd - ww;
        /*for (unsigned i = 0; i < dd.rows();  ++i)
            for (unsigned j = 0; j < dd.cols();  ++j)
                ll(i, j) = dd(i, j) - ww(i, j);*/

        // Calculate L2: Normalized Laplacian Matrix
        MatrixXd ll2(mat_size, mat_size);
        MatrixXd dd2(mat_size, mat_size);
        
        for (unsigned i = 0; i < dd.rows();  ++i)
            for (unsigned j = 0; j < dd.cols();  ++j)
                dd2(i, j) = 1 / sqrt(dd(i, j));

        ll2 = dd2 * ll * dd2;
    
        // Calculate limiting frequency w
        double w_lim = 0;
        for (unsigned j = 0; j < ll2.cols();  ++j)
            w_lim += ll2(1, j) * ll2(1, j);
        w_lim = sqrt(w_lim) + 0.001;
        
        // SVD descomposition
        SelfAdjointEigenSolver<MatrixXd> es(ll2);

        unsigned lim;
        for (lim = 0; lim < es.eigenvalues().rows(); ++lim)
            if (es.eigenvalues()[lim] > w_lim):
                break;

        if (lim < 2)
            lim = 2;
        
        // Find U head, U head head and v
        MatrixXd uu_h(mat_size, lim);
        for (unsigned i = 0; i < es.eigenvectors().rows(); ++i)
            for (unsigned j = 0; j < lim; ++j)
                uu_h(i, j) = es.eigenvectors(i, j);

        VectorXd vv(lim, 1);
        MatrixXd uu_hh(lim, mat_size - 1);
        
        for (unsigned i = 0; i < lim; ++i)
            vv(i, 1) = uu_h(1, i);

        for (unsigned i = 0; i < mat_size - 1; ++i)
            for (unsigned j = 0; i < lim; ++j)
                uu_hh(i, j) = uu_h(i + 1, j);

        // Compute rating prediction
        MatrixXd mm(lim, lim);
        mm = uu_hh.transpose() * uu_hh;

        double r_pred = vv.transpose() * mm.inverse() * uu_hh.transpose * ...

        
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }
    
    /**
    * \brief Signal all test vertices (25%)
    */
    static graphlab::empty signal_test(icontext_type& context,
                                       const vertex_type& vertex) {
        if(vertex.id() % 4  == 0) 
            context.signal(vertex);
        return graphlab::empty();
    } // end of signal_left

}; // end of als vertex program

int main(int argc, char** argv) {
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    
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
            
    dc.cout() << "Creating engine 2" << std::endl;
    graphlab::omni_engine<vertex_program> engine2(dc, graph, "sync");
        
    //TODO Signal test vertices (test user ratings)
    //engine2.map_reduce_vertices<graphlab::empty>(vertex_program::signal_test);
    engine2.signal_all();
        
    // Run 2nd engine
    dc.cout() << "Running ..." << std::endl;
    timer.start();
    engine2.start();

    const double runtime2 = timer.current_time();
    dc.cout() << "----------------------------------------------------------"
            << std::endl
            << "Final Runtime (seconds):   " << runtime2
            << std::endl
            << "Updates executed: " << engine2.num_updates() << std::endl
            << "Update Rate (updates/second): " 
            << engine2.num_updates() / runtime << std::endl;
            
    
    /* engine.add_vertex_aggregator<float>("error",
                                        error_vertex_data,
                                        print_finalize);
    engine.aggregate_now("error"); */
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
