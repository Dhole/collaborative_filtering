/**
 * \file cheby.cpp
 * 
 * \brief Filtering operation using chebychev coefficients
 *
 * This file contains the code to apply the filtering operation using the 
 * chebychev coefficients as input.
 */

#include <string>
#include <list>
#include <graphlab.hpp>
#include <math.h>

using namespace graphlab;

/** \brief Interval of operation */
double arange[] = { 0.0, 2.0 };
double a1 = (arange[1] - arange[0]) / 2;
double a2 = (arange[1] + arange[0]) / 2;

/** \brief vector to store the chebychev coefficients */
// Use some random values for testing purposes
double coeff[] = {2.23, 5.23, 0.19, 8.39};
//std::vector<double> coeff(&vv[0], &vv[0] + 4);
unsigned int coeff_len = 4;

/** \brief index for the current iteration */
//int ind = 0;

/**
 * \brief The vertex data stores the movie rating information.
 */
struct vertex_data {
    
    /** \brief node degree value */
    double degree;
    
    /** \brief Values to store the temporal results of each iteration */
    double twf_old, twf_cur, twf_new;
    
    /** \brief Signal value */
    double val;
    
    unsigned int counter;

    /** \brief basic initialization */
    vertex_data() { }
    
    vertex_data(double val): 
    val(val), counter(2) { }

    /** \brief Save the vertex data to a binary archive */
    void save(graphlab::oarchive& arc) const { 
        arc << degree << twf_old << twf_cur << twf_new << val << counter;
    }
    /** \brief Load the vertex data from a binary archive */
    void load(graphlab::iarchive& arc) { 
        arc >> degree >> twf_old >> twf_cur >> twf_new >> val >> counter;
    }
}; // end of vertex data

/**
 * \brief The edge data stores the weights between movies.
 */
struct edge_data : public graphlab::IS_POD_TYPE {
    
    /** \brief the weight value for the edge */
    double wei;
    
    /** \brief basic initialization */
    edge_data(double wei = 0) :
    wei(wei) { }
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
    double weight;
    
    strm >> va >> vb >> weight;
    if (weight > 0.1)
        graph.add_edge(va, vb, edge_data(weight));

    return true; // successful load
} // end of graph_loader

bool graph_signal_loader(graph_type& graph, 
                         const std::string& filename,
                         const std::string& line) {
    
    // Parse the line
    std::stringstream strm(line);
    graph_type::vertex_id_type vt(-1);
    double val;
    
    strm >> vt >> val;
    
    graph.add_vertex(vt, vertex_data(val));

    return true; // successful load
} // end of graph_signal_loader

/**
 * \brief Saves the users rating for each training user
 */
struct graph_signal_writer {
    std::string save_vertex(graph_type::vertex_type vt) {
        std::stringstream strm;
        strm << vt.id() << " " << vt.data().val << "\n";
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer

/**
 * \brief Compute the degree of each node and store it
 */
class degree_program : 
    public graphlab::ivertex_program<graph_type, double>,
    public graphlab::IS_POD_TYPE
    {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::OUT_EDGES; 
    }; // end of gather_edges 

    /** The gather function */
    double gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        return edge.data().wei;
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type& sum) {
        vertex.data().degree = sum;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of degree program

/**
 * \brief Compute the initial values and do the first iteration of the 
 * Chebychev filtering.
 */
class init_values_program :
    public graphlab::ivertex_program<graph_type, double>,
    public graphlab::IS_POD_TYPE
    {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::OUT_EDGES;
    }; // end of gather_edges 

    /** The gather function */
    double gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        return edge.data().wei / (std::sqrt(
               edge.target().data().degree * edge.source().data().degree))
               * edge.target().data().val;
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type& sum) {
        vertex.data().twf_old = vertex.data().val;
        
        vertex.data().twf_cur = (vertex.data().val - sum - 
                                a2 * vertex.data().val) / a1;
        
        vertex.data().val = 0.5 * coeff[0] * vertex.data().twf_old
                                + coeff[1] * vertex.data().twf_cur;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of init values program

/**
 * \brief Compute the next Chebychev iterations.
 */
class cheby_program :
    public graphlab::ivertex_program<graph_type, double>,
    public graphlab::IS_POD_TYPE
    {
public:

    /** The set of edges to gather along */
    edge_dir_type gather_edges(icontext_type& context, 
                                const vertex_type& vertex) const { 
        return graphlab::OUT_EDGES;
    }; // end of gather_edges 

    /** The gather function */
    double gather(icontext_type& context, const vertex_type& vertex, 
                       edge_type& edge) const {
        return edge.data().wei / (std::sqrt(
            edge.target().data().degree * edge.source().data().degree))
            * edge.target().data().twf_cur;
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type& sum) {
        vertex.data().twf_new = (2 / a1) * (vertex.data().twf_cur - sum
                                - a2 * vertex.data().twf_cur)
                                - vertex.data().twf_old;
        
        vertex.data().val = vertex.data().val
                            + coeff[vertex.data().counter] * vertex.data().twf_new;
        
        vertex.data().twf_old = vertex.data().twf_cur;
        vertex.data().twf_cur = vertex.data().twf_new;
        
        vertex.data().counter++;
        if (vertex.data().counter < coeff_len) 
            context.signal(vertex);
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of cheby program

int main(int argc, char** argv) {
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    
    dc.cout() << "Loading graph." << std::endl;
    graphlab::timer timer; 
    graph_type graph(dc);
    // Load the graph containing the weights and connections
    graph.load("graph_topology", graph_loader);
    // Load the signal of the graph
    graph.load("graph_signal", graph_signal_loader); 
    dc.cout() << "Loading graph. Finished in " 
    << timer.current_time() << std::endl;

    dc.cout() << "Filter lenght: " << coeff_len << std::endl;
    
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
        
    dc.cout() << "Creating engine 1 (Calculate degrees)" << std::endl;
    graphlab::omni_engine<degree_program> engine1(dc, graph, "sync");
        
    engine1.signal_all();
        
    // Calculate degrees
    dc.cout() << "Running ..." << std::endl;
    timer.start();
    engine1.start();

    double runtime = timer.current_time();
    dc.cout() << "----------------------------------------------------------"
            << std::endl
            << "Final Runtime (seconds):   " << runtime 
            << std::endl
            << "Updates executed: " << engine1.num_updates() << std::endl
            << "Update Rate (updates/second): " 
            << engine1.num_updates() / runtime << std::endl;
            
            
    dc.cout() << "Creating engine 2 (Init values + 2 iterations)" << std::endl;
    graphlab::omni_engine<init_values_program> engine2(dc, graph, "sync");
        
    engine2.signal_all();
        
    // Run init values + first 2 iterations
    dc.cout() << "Running ..." << std::endl;
    timer.start();
    engine2.start();

    runtime = timer.current_time();
    dc.cout() << "----------------------------------------------------------"
            << std::endl
            << "Final Runtime (seconds):   " << runtime 
            << std::endl
            << "Updates executed: " << engine2.num_updates() << std::endl
            << "Update Rate (updates/second): " 
            << engine2.num_updates() / runtime << std::endl;
            
    dc.cout() << "Creating engine 3 (Chebychev filtering iterations)" << std::endl;
    graphlab::omni_engine<cheby_program> engine3(dc, graph, "sync");
        
    engine3.signal_all();
        
    // Run Iterative filtering operations
    dc.cout() << "Running ..." << std::endl;
    timer.start();
    engine3.start();

    runtime = timer.current_time();
    dc.cout() << "----------------------------------------------------------"
            << std::endl
            << "Final Runtime (seconds):   " << runtime 
            << std::endl
            << "Updates executed: " << engine3.num_updates() << std::endl
            << "Update Rate (updates/second): " 
            << engine3.num_updates() / runtime << std::endl;

    graph.save("graph_filtered_signal", graph_signal_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
