# NB: this module uses the Manifold class from snappy, and the
# snappy.Manifold class uses objects from this module in its __init__
# method.  This works because we only call snappy.Manifold('empty')
# here.  It will not work to use "from snappy import Manifold".
import snappy
from snappy.db_utilities import decode_torsion, decode_matrices, db_hash
import sqlite3, re, os

# This module uses a single sqlite3 database with multiple tables.
# The path to the database file is specified at the module level.
from snappy.manifolds import __path__ as manifolds_paths
manifolds_path = manifolds_paths[0]
database_path = os.path.join(manifolds_path, 'manifolds.sqlite')

USE_COBS = 1 << 7
USE_STRING = 1 << 6
CUSP_MASK = 0x3f

class ManifoldTable:
    """
    Iterator for cusped manifolds in an sqlite3 table of manifolds.

    Initialize with the table name.  The table schema is required to
    include a text field called 'name' and a blob field called
    'triangulation'.  The blob holds the result of M._to_bytes() or
    M._to_string(), optionally preceded by a change of basis matrix
    for the peripheral curves.  The structure of the blob is
    determined by its first byte.

    Both mapping from the manifold name, and lookup by index are
    supported.  Slicing can be done either by numerical index or by
    volume.

    The __contains__ method is supported, so M in T returns True if M
    is isometric to a manifold in the table T.  The method
    T.identify(M) will return the matching manifold from the table.
    """
    # basic select clause.  Can be overridden, e.g. to additional columns
    _select = 'select name, triangulation from %s '

    def __init__(self, table='', **filter_args):
        self._table = table
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = self._manifold_factory
        # Sometimes we need a connection without the row factory
        self._connection2 = conn = sqlite3.connect(database_path)
        cursor = conn.execute("pragma table_info('%s')"%table)
        rows = cursor.fetchall()
        self.schema = dict([(row[1],row[2].lower()) for row in rows])
        assert self.schema['name'] == 'text' and \
               self.schema['triangulation'] == 'blob', \
               'Not a valid Manifold table.'
        cursor = conn.execute("select count(*) from %s"%self._table)
        self._configure(**filter_args)
        self._select = self._select%table

    def _configure(self, **kwargs):
        """
        Set up the filter and find our length.
        """
        conditions = []
        if 'betti' in kwargs:
            conditions.append('betti=%d ' % kwargs['betti'])
        if 'num_cusps' in kwargs:
            conditions.append('cusps=%d ' % kwargs['num_cusps'])
        if 'num_tets' in kwargs:
            conditions.append('tets=%d ' % kwargs['num_tets'])
        self._filter = ' and '.join(conditions)
        where_clause = self._filter
        if where_clause:
            where_clause = 'where ' + where_clause
        cursor = self._connection2.execute(
            'select count(*) from %s %s' % (self._table, where_clause)
            )
        self._length = cursor.fetchone()[0]
        
    def __repr__(self):
        if self._filter == '':
            return 'ManifoldTable object without filters'
        else:
            return 'ManifoldTable object with filter: %s'%self._filter
        
    def __call__(self, *args, **kwargs):
        if args: # backwards compatibility for LinkExteriors
            if not isinstance(args[0], int) or len(args) > 1:
                raise TypeError('Invalid specification for num_cusps.')
            if not kwargs.has_key('num_cusps'):
                kwargs['num_cusps'] = args[0]
        return ManifoldTable(self._table, **kwargs)
    
    def __len__(self):
        return self._length
        
    def __iter__(self):
        query = self._select
        if self._filter:
            query += ' where %s '%self._filter
        return self._connection.execute(query)

    def __contains__(self, mfld):
        try:
            M = self.identify(mfld)
            # duck test
            return M.num_tetrahedra > 0
        except:
            return False
                
    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step:
                raise IndexError('Slices with steps are not supported.')
            start, stop = index.start, index.stop
            if (isinstance(start, (float, type(None)) )
                and
                isinstance(stop, (float, type(None)) ) ):
                # Slice by volume.
                conditions = []
                if self._filter:
                    conditions.append(self._filter)
                if start:
                    conditions.append('volume >= %f' % start)
                if stop:
                    conditions.append('volume < %f' % stop)
                where_clause = ' and '.join(conditions)
                if where_clause:
                    where_clause = 'where ' + where_clause
                query = (self._select + where_clause)
                return self._connection.execute(query)
            elif (isinstance(start, (int, type(None)) )
                  and
                  isinstance(stop, (int, type(None)) ) ):
                if start and start < 0:
                    start = self._length + start
                if stop and stop < 0:
                    stop = self._length + stop
                if self._filter == '':
                    # With no filter we can slice by the id field;
                    start = 0 if start is None else start 
                    limit_clause = 'limit %d'%(stop - start) if stop else ''
                    query = (self._select + 'where id >= %d  %s ' % (
                                 start + 1,
                                 limit_clause))
                    return self._connection.execute(query)
                # otherwise we just trash the rows at the beginning. :^(
                else:
                    query = (self._select + 'where %s limit %d'%(
                                 self._filter,
                                 stop))
                    cursor = self._connection.execute(query)
                    if start:
                        cursor.row_factory = lambda x, y : None
                        cursor.fetchmany(start)
                        cursor.row_factory = self._manifold_factory
                    return cursor
            else:
                raise IndexError(
                    'Use two ints or two floats for start and stop.')
        elif isinstance(index, int):
            matches = self.find('id=%d'%(index + 1))
            if len(matches) != 1:
                raise IndexError('Manifold index is out of bounds')
        elif isinstance(index, str):
            matches = self.find("name='%s'"%index)
            if len(matches) != 1:
                raise KeyError('The manifold %s was not found.'%index)
        else:
            raise IndexError('%s is not a valid index type for manifolds.'%
                             type(index))
        return matches[0]
    
    def _manifold_factory(self, cursor, row):
        """
        Factory for "select name, triangulation" queries.
        Returns a Manifold.
        """
        buf = row[1]
        header = ord(buf[0])
        use_cobs, use_string = header&USE_COBS, header&USE_STRING
        num_cusps = header&CUSP_MASK
        M = snappy.Manifold('empty')
        if use_string:
            M._from_string(buf[1:])
        else:
            M._from_bytes(bytes(buf[4*num_cusps +1:]))
            if use_cobs:
                cobs = decode_matrices(buf[1:4*num_cusps + 1])
                M.set_peripheral_curves('combinatorial')
                M.set_peripheral_curves(cobs)
        self._finalize(M, row)
        return M

    def _finalize(self, M, row):
        """
        Give the manifold a name and make last-minute adjustments
        to the manifold before it leaves the factory, e.g. Dehn filling.
        Override this method for custom manifold production.
        """
        M.set_name(row[0])

    def keys(self):
        """
        Return the list of column names for this manifold table.
        """
        return self.schema.keys()
    
    def find(self, where, order_by='id', limit=25):
        """
        Return a list of up to limit manifolds stored in this table,
        satisfying the where clause, and ordered by the order_by
        clause.  If limit is None, all matching manifolds are
        returned.  The where clause is a required parameter.
        """
        where_clause = where
        if self._filter:
            where_clause += self._filter
        if limit is None:
            suffix = 'where %s order by %s'%(where_clause, order_by)
        else:
            suffix = 'where %s order by %s limit %d'%(
                where_clause, order_by, limit)
        cursor = self._connection.execute(self._select + suffix)
        return cursor.fetchall()

    def siblings(self, mfld):
        """
        Return all manifolds in the census which have the same hash value.
        """
        return self.find(where="hash = X'%s'"%db_hash(mfld))

    def identify(self, mfld):
        """
        Look for a manifold in this table which is isometric to the
        argument.

        Return the matching manifold, if there is one which SnapPea
        declares to be isometric.

        Return False if no manfold in the table has the same hash.

        Return None in all other cases (for now).
        """
        sibs = self.siblings(mfld)
        if len(sibs) == 0:
            return False # No hashes match
                # Check for isometry
        try:
            for N in sibs:
                if mfld.is_isometric_to(N):
                    return N
        except RuntimeError:
            pass
                # Check for identical triangulations
        for n in (1,2):
            for N in sibs:
                if mfld == N:
                    return N
            mfld.randomize()
        return None

class ClosedManifoldTable(ManifoldTable):

    _select = 'select name, triangulation, m, l from %s '

    def __call__(self, **kwargs):
        return ClosedManifoldTable(self._table, **kwargs)

    def _finalize(self, M, row):
        """
        Give the closed manifold a name and do the Dehn filling.
        """
        M.set_name(row[0])
        M.dehn_fill(row[2:4])

class OneCensusManifold():
    """
    Looks up a single manifold by name from the tables provided.
    Returns a triple: use_string, cobs, triangulation_data .
    """
    _query = "select triangulation from %s where name='%s'"

    def __init__(self, tables):
        self._tables = tables
        self._connection = sqlite3.connect(database_path)

    def __call__(self, name):
        for table in self._tables:
            query = self._query%(table, name)
            cursor = self._connection.execute(query)
            rows = cursor.fetchmany(2)
            if len(rows) > 1:
                raise ValueError('Manifold name is ambiguous')
            if len(rows) == 1:
                break
        if len(rows) == 0:
            raise KeyError('The manifold %s was not found.'%name)
        buf = rows[0][0]
        header = ord(buf[0])
        use_cobs, use_string = header&USE_COBS, header&USE_STRING
        num_cusps = header&CUSP_MASK
        cobs = None
        if use_string:
            triangulation_data = buf[1:]
        else:
            triangulation_data = bytes(buf[4*num_cusps +1:])
            if use_cobs:
                cobs = decode_matrices(buf[1:4*num_cusps + 1])
        return use_string, cobs, triangulation_data

# Instantiate our tables ...
OrientableCuspedCensus = ManifoldTable(table='orientable_cusped_view')
LinkExteriors = ManifoldTable(table='link_exteriors_view')
CensusKnots = ManifoldTable(table='census_knots_view')
OrientableClosedCensus = ClosedManifoldTable(table='orientable_closed_view')
NonorientableCuspedCensus = ManifoldTable(table='nonorientable_cusped_view')
NonorientableClosedCensus = ManifoldTable(table='nonorientable_closed_view')
# ... and the individual lookup objects for the Manifold class
CuspedManifoldData = OneCensusManifold( ['orientable_cusped_view',
                                         'nonorientable_cusped_view'] )
LinkExteriorData = OneCensusManifold( ['link_exteriors_view'] )
CensusKnotData = OneCensusManifold( ['census_knots_view'] )

# Test routines.
def test_census_database():
    L = OrientableCuspedDB
    for M in CensusKnots():
        print(M, L.identify(M))

def test():
    import re
    pairs = [ (CensusKnots(), CensusKnotsDB),
              (OrientableCuspedCensus(), OrientableCuspedDB)]
    for census, db in pairs:
        for M in census:
            if M.name()[0] == 't':
                break
            N = db.identify(M)
            assert repr(M) == repr(N)
            G, H = M.fundamental_group(), N.fundamental_group()
            if (G.relators() != H.relators() or
                G.peripheral_curves() != H.peripheral_curves()):
                print(M)

if __name__ == '__main__':
    test()
