/**
 * \file knn2.cpp
 * 
 * \brief The second step for KNN rating prediction
 *
 * This file contains the second step for KNN rating prediction. It reads the
 * input created by the first step and loads it as graph where each vertex
 * is a movie. It then calculates the weight of the edges between movies using
 * the vector cosine simmilarity. The result will be saved in a file.
 * This will output the weighted graph, which can be used for other algorithms.
 */

#include <string>
#include <list>
#include <graphlab.hpp>
#include <boost/unordered_map.hpp>
#include <math.h>

using namespace graphlab;

const int SAFE_NEG_OFFSET = 2; //add 2 to negative node id
//to prevent -0 and -1 which arenot allowed

typedef boost::unordered_map<vertex_id_type, double> map;


/**
 * \brief The vertex data stores the movie rating information.
 */
struct vertex_data {
    
    /** \brief The ratings each user has given to the movie */
    map ratings;

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
bool graph_vertex_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
    
    // Parse the line
    std::stringstream strm(line);
    graph_type::vertex_id_type vt(-1);
    
    strm >> vt;
    map ratings;
    
    while(1) {
        graphlab::vertex_id_type user;
        double rating;
        strm >> user;
        strm >> rating;
        if (strm.fail())
            break;
        ratings[user] = rating;
    }
    graph.add_vertex(vt, vertex_data(ratings));

    return true; // successful load
} // end of graph_loader

bool graph_edge_loader(graph_type& graph, 
                       const std::string& filename,
                       const std::string& line) {
    
    // Parse the line
    std::stringstream strm(line);
    graph_type::vertex_id_type vt(-1), tmp(-1);
    
    strm >> vt;
    while(1) {        
        strm >> tmp;
        if (strm.fail())
            break;
        graph.add_edge(vt, tmp);
    }

    return true; // successful load
} // end of graph_loader

/**
 * \brief Edge program to calculate the vector cosine simmilarity on all the
 * edges of the graph
 */
void weights_calc(edge_type& edge) {
    map ma, mb;
    float num = 0, den1 = 0, den2 = 0;
    int num_rat = 0;
    ma = edge.source().data().ratings;
    mb = edge.target().data().ratings;
    for (map::iterator it = ma.begin(); it != ma.end(); ++it) {
        if (mb.find(it->first) != mb.end()) {
            num_rat++;
            num += it->second * mb[it->first];
            den1 += it->second * it->second;
            den2 += mb[it->first] * mb[it->first];
        }
    }
    // Check number of common ratings between vertices
    if (num_rat > 5) //Should be arround 5-10
        edge.data().obs = num / (sqrt(den1) * sqrt(den2));
    else
        edge.data().obs = 0;
}

/**
 * \brief The vertex data stores the movie rating information.
 */
struct graph_writer {
    std::string save_vertex(graph_type::vertex_type vt) {
        return "";
    }
    std::string save_edge(graph_type::edge_type ed) { 
        std::stringstream strm;
        strm << ed.source().id() << " " << ed.target().id() << " "
             << ed.data().obs << "\n";
        return strm.str();
    }
}; // end of pagerank writer


int main(int argc, char** argv) {
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    
    dc.cout() << "Loading graph." << std::endl;
    graphlab::timer timer; 
    graph_type graph(dc);
    graph.load("out_rat_", graph_vertex_loader);
    graph.load("out_edg_", graph_edge_loader); 
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
        
    dc.cout() << "Calculating edge values." << std::endl;
    timer.start();
    graph.transform_edges(weights_calc);
    dc.cout() << "Finished calculating edge values in "
             << timer.current_time() << std::endl;
             
    dc.cout() << "Saving resulting graph." << std::endl;
    // Save the final graph -----------------------------------------------------
    graph.save("out_fin", graph_writer(),
                false,    // do not gzip
                false,    // do not save vertices
                true);   // save edges
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
