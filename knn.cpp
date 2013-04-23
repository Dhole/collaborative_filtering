/**
 * \file knn.cpp
 * 
 * \brief The first step for KNN rating prediction
 *
 * This file contains the first step for KNN rating prediction. It reads the
 * input user ratings and creates outputs with: movies containing each user
 * rating in one line per movie (one file for training and one for validation)
 * and one file containing the movie connections (two movies are connected if at 
 * least one same user has rated them both).
 */

#include <string>
#include <graphlab.hpp>
#include <boost/unordered_map.hpp>
#include <list>
#include <limits>

unsigned int uimax = std::numeric_limits<int>::max();

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
    map ratings_test;
    bool is_movie;

    vertex_data(bool is_movie = false): is_movie(is_movie) { }

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
 * distributed graph construction. We load the data as user to movie edges, 
 * where the edge values are the ratings from user to movie.
 */
bool graph_loader(graph_type& graph, 
                  const std::string& filename,
                  const std::string& line) {

    // Determine the role of the data
    edge_data::data_role_type role = edge_data::TRAIN;
    if(boost::ends_with(filename,".validate")) 
        role = edge_data::VALIDATE;
    else if(boost::ends_with(filename, ".train")) 
        role = edge_data::TRAIN;
    
    // Parse the line
    std::stringstream strm(line);
    graph_type::vertex_id_type source_id(-1), target_id(-1);
    double obs(0);
    strm >> source_id >> target_id >> obs;
    
    // map target id into a separate number space
    //source_id = -(graphlab::vertex_id_type(source_id + SAFE_NEG_OFFSET));
    //!! This could be dangerous, will uimax be the same in different platforms???
    source_id = uimax - source_id;

    // Create an edge and add it to the graph
    graph.add_vertex(source_id);
    graph.add_vertex(target_id, vertex_data(true));
    graph.add_edge(source_id, target_id, edge_data(obs, role));

    return true; // successful load
} // end of graph_loader


class gather_type {
public:
    
    map ratings; //Accumulate the user ratings
    map ratings_test; //Accumulate the user ratings for testing purposes
    
    /** \brief basic default constructor */
    gather_type() { }
    
    gather_type(vertex_id_type user, double rating) {
        ratings[user] = rating;
    }
    
    gather_type(vertex_id_type user, double rating, bool test) {
        if (test)
            ratings_test[user] = rating;
        else
            ratings[user] = rating;
    } // end of constructor for gather type
    
    gather_type(map rat) {
        //ratings.insert(rat.begin(), rat.end());
        ratings = rat;
    }
    
    /** \brief Save the values to a binary archive */
    void save(graphlab::oarchive& arc) const { arc << ratings; }
    
    /** \brief Read the values from a binary archive */
    void load(graphlab::iarchive& arc) { arc >> ratings; }
    
    /** 
     * \brief joins two maps
     */
    gather_type& operator+=(const gather_type& other) {
        ratings.insert(other.ratings.begin(), other.ratings.end());
        ratings_test.insert(other.ratings_test.begin(), other.ratings_test.end());
        return *this;
    } // end of operator+=
    
}; // end of gather type

/**
 * \brief The first step saves all the user ratings in two different maps
 * inside the vertex representing the movies.
 */
class vertex_program : 
    public graphlab::ivertex_program<graph_type, gather_type>,
    public graphlab::IS_POD_TYPE {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::IN_EDGES; 
    }; // end of gather_edges 

    /** The gather function */
    gather_type gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        if (edge.data().role == edge_data::TRAIN) {
            //printf("%i ", edge.source().id());
            return gather_type(edge.source().id(), edge.data().obs, false);
        } else {
            return gather_type(edge.source().id(), edge.data().obs, true);
        }
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
                const gather_type& sum) {
        //for (map::iterator it; it != sum.ratings.end(); ++it)
        //    graph.add_edge(vertex.id(), it->first);
        vertex.data().ratings = sum.ratings;
        vertex.data().ratings_test = sum.ratings_test;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

    /**
    * \brief Signal all vertices on one side of the bipartite graph
    */
    static graphlab::empty signal_right(icontext_type& context,
                                        const vertex_type& vertex) {
        if(vertex.num_out_edges() == 0) 
            context.signal(vertex);
        return graphlab::empty();
    } // end of signal_right
}; // end of als vertex program

/**
 * \brief The second step saves a map in every user vertex containing all the
 * movies ID which that user has rated. (This will be used in the next step
 * in order to find the movie connections)
 */
class vertex2_program : 
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
        return gather_type(edge.target().id(), edge.target().id());
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
                const gather_type& sum) {
        //for (map::iterator it; it != sum.ratings.end(); ++it)
        //    graph.add_edge(vertex.id(), it->first);
        vertex.data().ratings = sum.ratings;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

    /**
    * \brief Signal all vertices on one side of the bipartite graph
    */
    static graphlab::empty signal_left(icontext_type& context,
                                       const vertex_type& vertex) {
        if(vertex.num_in_edges() == 0) 
            context.signal(vertex);
        return graphlab::empty();
    } // end of signal_left
}; // end of als vertex program

/**
 * \brief The third and las step joins the maps from the user vertex which
 * contained the movies the user rated, into the vertex movie. This way, we have
 * now in each movie vertex a map containing all the movies ID it should be
 * connected to
 */
class vertex3_program : 
    public graphlab::ivertex_program<graph_type, gather_type>,
    public graphlab::IS_POD_TYPE {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::IN_EDGES; 
    }; // end of gather_edges 

    /** The gather function */
    gather_type gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        return gather_type(edge.source().data().ratings);
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
                const gather_type& sum) {
        //for (map::iterator it; it != sum.ratings.end(); ++it)
        //    graph.add_edge(vertex.id(), it->first);
        vertex.data().ratings = sum.ratings;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

    /**
    * \brief Signal all vertices on one side of the bipartite graph
    */
    static graphlab::empty signal_right(icontext_type& context,
                                        const vertex_type& vertex) {
        if(vertex.num_out_edges() == 0) 
            context.signal(vertex);
        return graphlab::empty();
    } // end of right_left
}; // end of als vertex program

/**
 * \brief Saves the users rating for each training user
 */
struct graph_writer {
    std::string save_vertex(graph_type::vertex_type vt) {
        std::stringstream strm;
        if (vt.num_out_edges() == 0) {
            strm << vt.id() << " ";
            for (map::iterator it = vt.data().ratings.begin(); it != vt.data().ratings.end(); ++it)
                strm << it->first << " " << it->second << " ";
            strm << "\n";
        }
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer

/**
 * \brief Saves the users rating for each test user
 */
struct graph_test_writer {
    std::string save_vertex(graph_type::vertex_type vt) {
        std::stringstream strm;
        if (vt.num_out_edges() == 0) {
            strm << vt.id() << " ";
            for (map::iterator it = vt.data().ratings_test.begin(); it != vt.data().ratings_test.end(); ++it)
                strm << it->first << " " << it->second << " ";
            strm << "\n";
        }
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer

/**
 * \brief Saves the edge connection between movies
 */
struct graph_edge_writer {
    std::string save_vertex(graph_type::vertex_type vt) {
        std::stringstream strm;
        std::list<vertex_id_type> li;
        if (vt.num_out_edges() == 0) {
            strm << vt.id() << " ";
            for (map::iterator it = vt.data().ratings.begin(); it != vt.data().ratings.end(); ++it) {
                if (it->first != vt.id())
                    li.push_back(it->first);
            }
            li.sort();
            li.unique();
            for (std::list<vertex_id_type>::const_iterator it = li.begin(); it != li.end(); ++it) {
                strm << *it << " ";
            }
            strm << "\n";
        }
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer

int main(int argc, char** argv) {
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    
    dc.cout() << "Loading graph." << std::endl;
    graphlab::timer timer; 
    graph_type graph(dc);
    graph.load("movielens/", graph_loader); 
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
    graphlab::omni_engine<vertex_program> engine(dc, graph, "sync");
        
    // Signal all vertices on the vertices on the left (liberals) 
    engine.map_reduce_vertices<graphlab::empty>(vertex_program::signal_right);
        
    // Run 1st engine
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
            
    // Save the final graph -----------------------------------------------------
    graph.save("out_rat", graph_writer(),
                false,    // do not gzip
                true,     // save vertices
                false);   // do not save edges
    
    graph.save("out_test_rat", graph_test_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
            
    dc.cout() << "Creating engine 2" << std::endl;
    graphlab::omni_engine<vertex2_program> engine2(dc, graph, "sync");
        
    // Signal all vertices on the vertices on the left (liberals) 
    engine2.map_reduce_vertices<graphlab::empty>(vertex2_program::signal_left);
        
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
            
    dc.cout() << "Creating engine 3" << std::endl;
    graphlab::omni_engine<vertex3_program> engine3(dc, graph, "sync");
        
    // Signal all vertices on the vertices on the left (liberals) 
    engine3.map_reduce_vertices<graphlab::empty>(vertex3_program::signal_right);
        
    // Run 3rd engine
    dc.cout() << "Running ..." << std::endl;
    timer.start();
    engine3.start();

    const double runtime3 = timer.current_time();
    dc.cout() << "----------------------------------------------------------"
            << std::endl
            << "Final Runtime (seconds):   " << runtime3
            << std::endl
            << "Updates executed: " << engine3.num_updates() << std::endl
            << "Update Rate (updates/second): " 
            << engine3.num_updates() / runtime << std::endl;
    
    graph.save("out_edg", graph_edge_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
