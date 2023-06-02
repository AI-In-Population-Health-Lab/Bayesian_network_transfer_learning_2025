/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.pitt.isp.sverchkov.graph;

import java.util.*;

/**
 *
 * @author YUS24
 */
public class GraphTools {

    /**
     * Returns a list of node in (a) topological order based on the DAG
     * @param <N> Node identifier type
     * @param net the DAG
     * @return A list of nodes in topological order (orphans first).
     */
    public static <N> List<N> nodesInTopOrder( DAG<N> net ){
        List<N> result = new ArrayList<>(net.size());

        // Get a local representation of the graph
        Map<N,Set<N>> graphMap = new HashMap<>();
        for( N node : net ){
            Set<N> parents = new HashSet<>( net.parents( node ) );
            graphMap.put( node, parents );
        }

        // Init parentless set
        Set<N> orphans = new HashSet<>();
        {
            Set<N> removeSet = new HashSet<>();
            for( Map.Entry<N,Set<N>> node : graphMap.entrySet() )
                if( node.getValue().isEmpty() )
                    orphans.add( node.getKey() );

            graphMap.keySet().removeAll( removeSet );
        }

        // The meat
        while( !orphans.isEmpty() ){

            // Get an orphan node
            N node = orphans.iterator().next();
            orphans.remove( node );
            result.add( node );

            // Remove the node and its edges from the graph
            graphMap.remove( node );
            for( Map.Entry<N,Set<N>> entry : graphMap.entrySet() ){
                entry.getValue().remove( node );
                if( entry.getValue().isEmpty() ){
                    orphans.add( entry.getKey() );
                }
            }
        }

        // Optional sanity check could go here

        return result;
    }

    /**
     * Uses the algorithm outlined in H.Jacques Suermondt, Gregory F. Cooper,
     * Probabilistic inference in multiply connected belief networks using loop
     * cutsets, International Journal of Approximate Reasoning, Volume 4,
     * Issue 4, July 1990, Pages 283-306, ISSN 0888-613X,
     * http://dx.doi.org/10.1016/0888-613X(90)90003-K. (http://www.sciencedirect.com/science/article/pii/0888613X9090003K)
     * to find the cutset for a value DAG
     * @param <Node> The class type for uniquely identifying nodes
     * @param <Value> The class type for uniquely identifying values
     * @param dag The DAG for which to find the cutset.
     * @return The set of nodes defining the cutset.
     */
    public static <Node,Value> Set<Node> cutSetSuermondtCooper( ValueDAG<Node,Value> dag ){
        Set<Node> cutset = new HashSet<>();
        MutableValueDAGImpl<Node,Value> workingDAG = new MutableValueDAGImpl<>(dag);

        for(;;){
            {// Remove singly-connected components
                Set<Node> removeSet = new HashSet();
                do{
                    // Remove
                    for( Node n : removeSet )
                        workingDAG.removeNode(n);
                    removeSet.clear();

                    // Mark for removal
                    for( Node n : workingDAG )
                        if( workingDAG.children(n).size() + workingDAG.parents(n).size() <= 1 )
                            removeSet.add(n);
                }while( !removeSet.isEmpty() );
            }
            if(  workingDAG.size() == 0 ) return cutset;
            {// Find next cutset node
                Node candidate = null;
                for( Node n : workingDAG )
                    if( workingDAG.parents(n).size() <= 1 )
                        if( null == candidate ||
                                workingDAG.children(n).size() > workingDAG.children(candidate).size() ||
                                ( workingDAG.children(n).size() == workingDAG.children(candidate).size() &&
                                        workingDAG.values(n).size() > workingDAG.values(candidate).size() ) )
                            candidate = n;

                cutset.add(candidate);
                workingDAG.removeNode(candidate);
            }
        }
    }

    public static <Node, Value> MutableValueDAGImpl<Node, Value> ancestrySubgraph( ValueDAG<Node,Value> dag, Node node ){
        class Edge{ Node source, dest; Edge(Node s, Node d){source=s; dest=d;} }
        Queue<Node> nodeQ = new LinkedList<>();
        Set<Node> nodes = new HashSet<>();
        List<Edge> edges = new LinkedList<>();

        nodeQ.add(node);
        for( Node n; null != (n = nodeQ.poll());)
            if( nodes.add(n) )
                for( Node parent : dag.parents(n) ){
                    nodeQ.add(parent);
                    edges.add( new Edge( parent, n ) );
                }

        MutableValueDAGImpl<Node, Value> result = new MutableValueDAGImpl<>();
        for( Node n : nodes )
            result.addNode( n, dag.values(n) );
        for( Edge e : edges )
            result.addArc( e.source, e.dest );

        return result;
    }
}
