package edu.pitt.isp.sverchkov.combinatorics;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



import java.util.*;

import java.util.Map.Entry;

/**
 * A tool for iterating over all possible key-value assignments given a set
 * of values for each key.
 *
 * @author YUS24
 */
public class Assignments<K,V> implements Iterable<Map<K,V>> {

    private List<K> keys;
    private List<List<V>> values;

    public Assignments( K[] keys, V[][] values ){
        this.keys = Collections.unmodifiableList( Arrays.asList(keys) );

        List<List<V>> tmp = new ArrayList<>(values.length);
        for( V[] valueArray : values )
            tmp.add( Collections.unmodifiableList( Arrays.asList(valueArray) ) );

        this.values = Collections.unmodifiableList(tmp);
    }

    public Assignments( List<K> keys, List<? extends List<? extends V>> values ){

        this.keys = Collections.unmodifiableList( new ArrayList<>(keys) );

        List<List<V>> tmp = new ArrayList<>(values.size());
        for( List<? extends V> valueList : values )
            tmp.add( Collections.unmodifiableList( new ArrayList<>(valueList)) );
        this.values = Collections.unmodifiableList(tmp);
    }

    public Assignments( Map<K,? extends Collection<? extends V>> setMap ){
        keys = new ArrayList<>();
        values = new ArrayList<>();

        for( Map.Entry<K, ? extends Collection<? extends V>> entry : setMap.entrySet() ){
            keys.add( entry.getKey() );
            values.add( Collections.unmodifiableList( new ArrayList<>(entry.getValue()) ) );
        }

        keys = Collections.unmodifiableList(keys);
        values = Collections.unmodifiableList(values);
    }

    @Override
    public Iterator<Map<K, V>> iterator() {
        return new AIterator();
    }

    private class AIterator implements Iterator<Map<K,V>> {

        private int[] indeces;
        private boolean valid;

        private AIterator(){
            indeces = new int[keys.size()];
            Arrays.fill(indeces, 0);
            valid = true;
        }

        @Override
        public boolean hasNext() {
            return valid;
        }

        @Override
        public Map<K, V> next() {
            if( !valid ) return null;

            Map<K,V> result = new Assignment(indeces);//new HashMap<>();
            /* OLD Record result
            for( int bit = 0; bit < indeces.length; bit++ )
                result.put( keys.get(bit), values.get(bit).get(indeces[bit]) );*/

            // Update counter
            int bit;
            for( bit = 0; bit < indeces.length; bit++ )
                if( ++indeces[bit] < values.get(bit).size() )
                    break;
                else
                    indeces[bit] = 0;

            // Check if an overflow occured
            valid = bit < indeces.length;
            return result;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Remove not supported for this iterator.");
        }
    }

    private class Assignment extends AbstractMap<K,V> {
        final EntrySet entrySet;

        Assignment( int[] indexArray ){
            entrySet = new EntrySet( indexArray );
        }

        @Override
        public Set<Entry<K, V>> entrySet() {
            return entrySet;
        }
    }
    private class EntrySet extends AbstractSet<Map.Entry<K,V>> {

        int[] setIndeces;

        EntrySet( int[] indexArray ){
            setIndeces = new int[indexArray.length];
            System.arraycopy(indexArray, 0, setIndeces, 0, indexArray.length);
        }

        @Override
        public Iterator<Map.Entry<K, V>> iterator() {
            return new Iter();
        }

        @Override
        public int size() {
            return setIndeces.length;
        }

        private class Iter implements Iterator<Map.Entry<K,V>> {
            int bit = 0;

            @Override
            public boolean hasNext() {
                return bit < setIndeces.length;
            }

            @Override
            public Map.Entry<K, V> next() {
                Map.Entry<K,V> next = new AbstractMap.SimpleImmutableEntry<>( keys.get(bit), values.get(bit).get(setIndeces[bit]) );
                ++bit;
                return next;
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException("Remove not supported for this iterator.");
            }
        }
    }
}

