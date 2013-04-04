/**
 * \file knn3.cpp
 * 
 * \brief The third step for KNN rating prediction
 *
 * This file contains the third step for KNN rating prediction. It reads the
 * graph created by the second step and loads it as graph where each vertex
 * is a movie and the weight of the edges is the vector cosine simmilarity. 
 * It will load into each vertex the test data. Then each user rating will be
 * predicted using the KNN algorithm. After that, the Mean Square Error will be
 * computed as an average between all the test ratings and the KNN predicted 
 * ratings.
 */

#include <string>
#include <list>
#include <graphlab.hpp>
#include <boost/unordered_map.hpp>
#include <math.h>
#include <boost/math/special_functions/round.hpp>

using namespace graphlab;

typedef boost::unordered_map<vertex_id_type, double> map;


/**
 * \brief The vertex data stores the movie rating information.
 */
struct vertex_data {
    
    /** \brief The ratings each user has given to the movie */
    map ratings;
    /** \brief The predicted ratings found using KNN */
    map ratings_knn;

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


class gather_type {
public:
    
    map ratings;
    map weights;
    
    /** \brief basic default constructor */
    gather_type() { }
    
    gather_type(vertex_id_type user, double rating) {
        ratings[user] = rating;
    }
    
    gather_type(map rat, float wei) {
        //ratings.insert(rat.begin(), rat.end());
        ratings = rat;
        for (map::iterator it = ratings.begin(); it != ratings.end(); ++it)
            weights[it->first] = wei;
    }
    
    /** \brief Save the values to a binary archive */
    void save(graphlab::oarchive& arc) const { arc << ratings; }
    
    /** \brief Read the values from a binary archive */
    void load(graphlab::iarchive& arc) { arc >> ratings; }
    
    /** 
     * \brief joins two maps(ratings) along with the weights for each user
     */
    gather_type& operator+=(const gather_type& other) {
        map sum_ratings;
        map sum_weights;
        map other_ratings = other.ratings;
        map other_weights = other.weights;
        for (map::iterator it = ratings.begin(); it != ratings.end(); ++it) {
            sum_ratings[it->first] = 0;
            sum_weights[it->first] = 0;
        }
        for (map::const_iterator it = other_ratings.begin(); it != other_ratings.end(); ++it) {
            sum_ratings[it->first] = 0;
            sum_weights[it->first] = 0;
        }
        
        for (map::iterator it = sum_ratings.begin(); it != sum_ratings.end(); ++it){
            if(ratings.find(it->first) != ratings.end()) {
                sum_ratings[it->first] += ratings[it->first];
                sum_weights[it->first] += weights[it->first];
            }
            if(other_ratings.find(it->first) != other_ratings.end()) {
                sum_ratings[it->first] += other_ratings[it->first];
                sum_weights[it->first] += other_weights[it->first];
            }
                
        }
        ratings = sum_ratings;
        weights = sum_weights;
        return *this;
    } // end of operator+=
}; // end of gather type

/**
 * \brief Compute the KNN for each rating in the vertices
 */
class knn_program : 
    public graphlab::ivertex_program<graph_type, gather_type>,
    public graphlab::IS_POD_TYPE {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::OUT_EDGES; 
    }; // end of gather_edges 

    /** The gather function */
    gather_type gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        map wei_rat;
        vertex_data etd = edge.target().data();
        for (map::iterator it = etd.ratings.begin(); it != etd.ratings.end(); ++it) 
            wei_rat[it->first] = edge.data().obs * etd.ratings[it->first];
        
        return gather_type(wei_rat, edge.data().obs);
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type& sum) {
        //for (map::iterator it; it != sum.ratings.end(); ++it)
        //    graph.add_edge(vertex.id(), it->first);
        map norm_knn;
        map sum_ratings = sum.ratings;
        map sum_weights = sum.weights;
        for (map::const_iterator it = sum.ratings.begin(); it != sum.ratings.end(); ++it) {
            //std::cout << "(" << vertex.id() << " " << sum_ratings[it->first] << " " << sum_weights[it->first] << ") ";
            norm_knn[it->first] = sum_ratings[it->first] / sum_weights[it->first];
        }
        vertex.data().ratings_knn = norm_knn;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of als vertex program

typedef graphlab::omni_engine<knn_program> engine_type;

/**
 * \brief Compute the error between the KNN predicted value and the test rating
 */
float error_vertex_data(engine_type::icontext_type& context,
                        const graph_type::vertex_type& vertex) {
    float err = 0, tmp;
    vertex_data vd = vertex.data();
    
    if (vd.ratings.size() > 0) {
        for (map::iterator it = vd.ratings.begin(); it != vd.ratings.end(); ++it) {
            //std::cout << "(" << vertex.id() << " " << vd.ratings[it->first] << " " << vd.ratings_knn[it->first] << ") ";
            //Check if the KNN was properly computed or not (Maybe the vertex didn't have neighbours)
            if (vd.ratings_knn[it->first] < 0.1) 
                tmp = 0;
            else
                tmp = (vd.ratings[it->first] - boost::math::round(vd.ratings_knn[it->first]));
            err += tmp * tmp;
        }
        if (isnan(err)) {
            //std::cout << "NaN ";
            return 0;
        } else
            return err / vd.ratings.size();
    } else
        return 0;
}

/**
 * \brief Output the result error
 */
void print_finalize(engine_type::icontext_type& context,
                    float total) {
    context.cout() << "Knn Average MSE: " << total / context.num_vertices() << "\n";
}

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
        
    dc.cout() << "Creating engine" << std::endl;
    graphlab::omni_engine<knn_program> engine(dc, graph, "sync");
        
    engine.signal_all();
        
    // Run KNN
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
            
    engine.add_vertex_aggregator<float>("error",
                                        error_vertex_data,
                                        print_finalize);
    engine.aggregate_now("error");
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
