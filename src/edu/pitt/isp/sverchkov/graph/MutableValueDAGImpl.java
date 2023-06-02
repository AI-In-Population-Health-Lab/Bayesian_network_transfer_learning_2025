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
public final class MutableValueDAGImpl<Node,Value> extends AbstractValueDAG<Node,Value> {

    private final Map<Node,NodeInfo> nodes;

    public MutableValueDAGImpl(){
        nodes = new HashMap<>();
    }

    public MutableValueDAGImpl( ValueDAG<Node,Value> source ){
        this();
        for( Node node : source )
            addNode( node, source.values(node) );
        for( Node node : source )
            for( Node parent : source.parents(node) )
                addArc( parent, node );
    }

    @Override
    public List<Value> values(Node node) {
        return nodes.get(node).values;
    }

    @Override
    public int size() {
        return nodes.size();
    }

    @Override
    public Set<Node> parents(Node node) {
        return Collections.unmodifiableSet(nodes.get(node).parents);
    }

    @Override
    public Iterator<Node> iterator() {
        return Collections.unmodifiableSet(nodes.keySet()).iterator();
    }

    public Set<Node> children(Node node) {
        return Collections.unmodifiableSet(nodes.get(node).children);
    }

    public void addNode(Node node, Collection<Value> values){
        if( nodes.containsKey(node) ) removeNode( node );
        nodes.put(node, new NodeInfo( values ));
    }

    public void addArc(Node parent, Node child) {
        if( nodes.get(parent).children.add(child) && nodes.get(child).parents.add(parent) )
            assignments.remove( child );
    }

    public void removeArc(Node parent, Node child) {
        if( nodes.get(parent).children.remove(child) && nodes.get(child).parents.remove(parent) )
            assignments.remove(child);
    }

    public void removeNode(Node node) {
        NodeInfo info = nodes.get(node);
        if( null != info ){
            for( Node child : (Node[]) info.children.toArray() )
                removeArc( node, child );
            for( Node parent : (Node[]) info.parents.toArray() )
                removeArc( parent, node );
            nodes.remove(node);
        }
    }

    final private class NodeInfo {
        final Set<Node> parents;
        final Set<Node> children;
        final List<Value> values;
        NodeInfo( Collection<Value> values ) {
            this.values = Collections.unmodifiableList( new ArrayList<>(values) );
            this.parents = new HashSet<>();
            this.children = new HashSet<>();
        }
    }
}
