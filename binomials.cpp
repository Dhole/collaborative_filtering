/**
 * \file binomials.cpp
 * 
 * \brief Filtering operation using polynomial factorization (binomials) coefficients
 *
 * This file contains the code to apply the filtering operation using the 
 * binomials coefficients from a polynomial factorization.
 */

#include <string>
#include <list>
#include <graphlab.hpp>
#include <math.h>

using namespace graphlab;

/** \brief vector to store the chebychev coefficients */
// Use some random values for testing purposes
//double coeff[] = {2.23, 5.23, 0.19};
std::vector<double> coeff;
unsigned int coeff_len;

/** \brief index for the current iteration */
int ind = 0;

/**
 * \brief The vertex data stores the movie rating information.
 */
struct vertex_data {
    
    /** \brief node degree value */
    double degree;
    
    /** \brief Values to store the temporal results of each iteration */
    double tmp, part_a, part_b;
    
    /** \brief Signal value */
    double val;

    /** \brief basic initialization */
    vertex_data() { }
    
    vertex_data(double val): 
    val(val), counter(2) { }

    /** \brief Save the vertex data to a binary archive */
    void save(graphlab::oarchive& arc) const { 
        arc << degree << tmp << part_a << part_b << val << counter;
    }
    /** \brief Load the vertex data from a binary archive */
    void load(graphlab::iarchive& arc) { 
        arc >> degree >> tmp >> part_a >> part_b >> val >> counter;
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
    if (weight > 0.1) {
        graph.add_edge(va, vb, edge_data(weight));
        graph.add_edge(vb, va, edge_data(weight));
    }

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

bool filter_loader(graph_type& graph, 
                   const std::string& filename,
                   const std::string& line) {
    
    // Parse the line
    std::stringstream strm(line);
    double val;
    
    while (1) {
        strm >> val;
        if (strm.fail())
            break;
        coeff.push_back(val);
    }
    coeff_len = coeff.size();

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
 * \brief Compute the binomial operations step_a.
 */
class binomial_a_program :
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
        
        vertex.data.part_a = (coeff[ind] + coeff[ind + 1]) * vertex.data.val;
                             - coeff[ind + 1] * sum;
        vertex.data.tmp = vertex.data.val - sum;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of binomial step_a program

/**
 * \brief Compute the binomial operations step_b.
 */
class binomial_b_program :
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
            * edge.target().data().tmp;
    } // end of gather function

    void apply(icontext_type& context, vertex_type& vertex,
               const gather_type& sum) {
        
        vertex.data.part_b = coeff[ind + 2] * (vertex.data.tmp - sum);
        veftex.data.val = vertex.data.part_a + vertex.data.part_b;
    } // end of apply

    // No scatter needed. Return NO_EDGES
    edge_dir_type scatter_edges(icontext_type& context,
                                const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

}; // end of binomial step_b program

int main(int argc, char** argv) {
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    
    dc.cout() << "Loading graph." << std::endl;
    graphlab::timer timer; 
    graph_type graph(dc);
    // Load the filter coefficients
    graph.load("coeff", filter_loader);
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
            
            
    dc.cout() << "Creating engine 2 iteration step_a" << std::endl;
    graphlab::omni_engine<binomial_a_program> engine2(dc, graph, "sync");
    
    dc.cout() << "Creating engine 3 iteration steb_b" << std::endl;
    graphlab::omni_engine<binomial_b_program> engine3(dc, graph, "sync");
    
    
    for (unsigned int i = 0; i*3 < coeff_len; i++) {
    
        engine2.signal_all();
    
        // Run step_a
        dc.cout() << "Running step a..." << std::endl;
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
                
            
        engine3.signal_all();
            
        // Run step_b
        dc.cout() << "Running step b..." << std::endl;
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
        ind++;
    }

    graph.save("graph_filtered_signal", graph_signal_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
    
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
