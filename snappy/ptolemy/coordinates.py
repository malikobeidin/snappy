from ptolemyVariety import list_all_quadruples_with_fixed_sum
from solutionsToGroebnerBasis import AlgebraicNumber

try:
    from sage.libs.pari import gen 
    from sage.libs.pari.gen import pari
    _within_sage = True
except ImportError:
    from cypari import gen
    from cypari.gen import pari
    _within_sage = False

import matrix
import re

class PtolemyCoordinates(dict):
    """
    Represents a solution of a Ptolemy variety as python dictionary.

    === Examples ===

    Construct solution from magma output:

    >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
    >>> from snappy import Manifold
    >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
    >>> solution = solutions[2]

    Access a Ptolemy coordinate:

    >>> solution['c_2100_0']
    1

    >>> solution.number_field()
    x^2 - x + 1

    Check that it is really a solution, exactly:
    
    >>> M = Manifold("4_1")
    >>> solution.check_against_manifold(M)

    Turn into (Galois conjugate) numerical solutions:

    >>> old_precision = pari.set_real_precision(100) # with high precision
    >>> numerical_solutions = solution.numerical()
    
    Check that it is a solution, numerically:

    >>> numerical_solutions[0].check_against_manifold(M, 1e-80)
    >>> pari.set_real_precision(old_precision) # reset pari engine
    100

    Compute cross ratios from the Ptolemy coordinates (cross ratios
    according to SnapPy convention, see help(solution.cross_ratios):

    >>> cross = solution.cross_ratios()
    >>> cross['z_0001_0']
    Mod(x, x^2 - x + 1)

    Compute volumes:

    >>> volumes = cross.volume_numerical()

    Check that volume is 4 times the geometric one:

    >>> volume = volumes[0].abs()
    >>> diff = volume - 4 * M.volume()
    >>> diff < 1e-9
    True

    Compute flattenings:
    
    >>> flattenings = solution.flattenings_numerical()

    Compute complex volumes:

    >>> cvols = [flattening.complex_volume() for flattening in flattenings]
    >>> volume = cvols[0].real().abs()
    >>> chernSimons = cvols[0].imag()
    >>> diff = volume - 4 * M.volume()
    >>> diff < 1e-9
    True

    >>> from snappy import pari
    >>> normalized = chernSimons * 6 / (pari('Pi')**2)

    Check that Chern Simons is zero up to 6 torsion:
    
    >>> normalized - normalized.round() < 1e-9
    True
    """
        
    def __init__(self, d, is_numerical = True):
        super(PtolemyCoordinates, self).__init__(d)
        self._is_numerical = is_numerical
        
    def number_field(self):
        """
        For an exact solution, return the number_field spanned by the
        Ptolemy coordinates. If number_field is Q, return None.
        """
        
        assert not self._is_numerical, "number_field for numerical solution"

        some_value = self.values()[0]
        if some_value.type() == 't_POLMOD':
            return some_value.mod()
        else:
            return None
        

    def numerical(self):
        """
        Turn exact solutions into numerical solutions using pari.

        Take an exact solution:

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Turn into a numerical solution:
        
        >>> old_precision = pari.set_real_precision(100) # with high precision
        >>> numerical_solutions = solution.numerical()
        >>> pari.set_real_precision(old_precision) # reset pari engine
        100

        Pick one of the Galois conjugates:

        >>> numerical_solution = numerical_solutions[0]
        >>> value = numerical_solution['c_1110_0']
        """
        
        if self._is_numerical:
            return self
        return [PtolemyCoordinates(d, is_numerical = True)
                for d in _to_numerical(self)]

    def cross_ratios(self):
        """
        Compute cross ratios from Ptolemy coordinates. The cross ratios are
        according to the SnapPy convention, so we have 
             z = 1 - 1/zp, zp = 1 - 1/zpp, zpp = 1 - 1/z
        where
             z   is at the edge 01 and equal to s0 * s1 * (c_1010 * c_0101) / (c_1001 * c_0110)
             zp  is at the edge 02 and equal to s0 * s2 * (c_1001 * c_0110) / (c_1100 * c_0011)
             zpp is at the edge 03 and equal to s0 * s3 * (c_1100 * c_0011) / (c_0101 * c_1010).

        Note that this is different from the convention used in 
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds 
        http://arxiv.org/abs/1207.6711

        Take an exact solution:

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Turn into cross Ratios:

        >>> crossRatios = solution.cross_ratios()
        
        Get a cross ratio:
        
        >>> crossRatios['zp_0010_0']
        Mod(x, x^2 - x + 1)

        Check the relationship between cross ratios:
        
        >>> crossRatios['z_0010_0'] == 1 - 1 / crossRatios['zp_0010_0']
        True

        >>> crossRatios['zp_0010_0'] == 1 - 1 / crossRatios['zpp_0010_0']
        True

        >>> crossRatios['zpp_0010_0'] == 1 - 1 / crossRatios['z_0010_0']
        True

        Get information about what one can do with cross ratios
        """
        return CrossRatios(_ptolemy_to_cross_ratio(self),
                           is_numerical = self._is_numerical)

    def cross_ratios_numerical(self):
        """
        Turn exact solutions into numerical and then compute cross ratios.
        See numerical() and cross_ratios().
        """
        
        if self._is_numerical:
            return self.cross_ratios()
        else:
            return [num.cross_ratios() for num in self.numerical()]

    def flattenings_numerical(self):
        """
        Turn into numerical solutions and compute flattenings, see 
        help(snappy.ptolemy.coordinates.Flattenings)
        Also see numerical()

        Get Ptolemy coordinates.

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Compute a numerical soluton

        >>> flattenings = solution.flattenings_numerical()

        Get more information with help(flattenings[0])
        """
        if self._is_numerical:
            return Flattenings(
                _ptolemy_to_cross_ratio(self,
                                        as_flattenings = True))

        else:
            return [num.flattenings_numerical() for num in self.numerical()]

    def volume_numerical(self, drop_negative_vols = False):
        """
        Turn into (Galois conjugate) numerical solutions and compute volumes.
        If already numerical, only return the one volume.
        See numerical().

        If drop_negative_vols = True is given as optional argument,
        only return non-negative volumes.
        """
        if self._is_numerical:
            return self.cross_ratios().volume_numerical()
        else:
            vols = [num.volume_numerical() for num in self.numerical()]
            if drop_negative_vols:
                return [vol for vol in vols if vol > -1e-12]
            return vols

    def complex_volume_numerical(self, drop_negative_vols = False):
        """
        Turn into (Galois conjugate) numerical solutions and compute complex
        volumes. If already numerical, return the volume.

        Complex volume is defined up to i*pi**2/6.

        See numerical(). If drop_negative_vols = True is given as optional
        argument, only return complex volumes with non-negative real part.
        """
        
        if self._is_numerical:
            return self.flattenings_numerical().complex_volume()
        else:
            cvols = [num.flattenings_numerical().complex_volume()
                     for num in self.numerical()]
            if drop_negative_vols:
                return [cvol for cvol in cvols if cvol.real() > -1e-12]
            return cvols

    def check_against_manifold(self, M, epsilon = None):
        """
        Checks that the given solution really is a solution to the Ptolemy
        variety of a manifold. See help(ptolemy.PtolemyCoordinates) for
        more example.

        === Arguments ===

        M --- manifold to check this for
        epsilon --- maximal allowed error when checking the relations, use
        None for exact comparision.
        """

        def get_obstruction_variable(face, tet):
            key = "s_%d_%d" % (face, tet)
            return self[key]

        def check(v, comment):
            if epsilon is None:
                assert v == 0, comment
            else:
                assert v.abs() < epsilon, comment + " error: %s" % v.abs()
        
        N, num_tets, has_obstruction_class = _find_N_tets_obstruction(
            self)

        assert M.num_tetrahedra() == num_tets, "Number tetrahedra not matching"

        if has_obstruction_class:
            # check cocycle condition
            for tet in range(num_tets):
                check( get_obstruction_variable(0, tet) *
                       get_obstruction_variable(1, tet) *
                       get_obstruction_variable(2, tet) *
                       get_obstruction_variable(3, tet) - 1,
                       "Obstruction cocycle condition violated")
            # check identified faces
            for dummy_sign, var1, var2 in (
                    M._ptolemy_equations_identified_face_classes()):
                check ( self[var1] - self[var2],
                        "Identified face classes violated")

        # Check identified Ptolemy coordinates
        for sign, var1, var2 in (
                M._ptolemy_equations_identified_coordinates(N)):
            check (self[var1] - sign * self[var2],
                   "Identified Ptolemy coordinates violated")

        # Check Ptolemy relationship
        indices = list_all_quadruples_with_fixed_sum(N - 2, skipVerts = False)

        for tet in range(num_tets):
            for index in indices:

                def get_ptolemy_coordinate(addl_index):
                    total_index = matrix.vector_add(index, addl_index)
                    key = "c_%d%d%d%d" % tuple(total_index) + "_%d" % tet
                    return self[key]


                if has_obstruction_class:
                    s0 = get_obstruction_variable(0, tet)
                    s1 = get_obstruction_variable(1, tet)
                    s2 = get_obstruction_variable(2, tet)
                    s3 = get_obstruction_variable(3, tet)
                else:
                    s0 = 1
                    s1 = 1
                    s2 = 1
                    s3 = 1
                    
                rel = (  s0 * s1 * get_ptolemy_coordinate((1,1,0,0))
                                 * get_ptolemy_coordinate((0,0,1,1))
                       - s0 * s2 * get_ptolemy_coordinate((1,0,1,0))
                                 * get_ptolemy_coordinate((0,1,0,1))
                       + s0 * s3 * get_ptolemy_coordinate((1,0,0,1))
                                 * get_ptolemy_coordinate((0,1,1,0)))

                check(rel, "Ptolemy relation violated")

class Flattenings(dict):
    """
    Represents a flattening assigned to each edge of a simplex as dictionary.

    We assign to each pair of parallel edges of each simplex a triple (w, z, p)
    such that
           w = log(z) + p * pi * i.
    The three triples belonging to a simplex form a combinatorial flattening
    (w0, w1, w2) as defined in Definiton 3.1 in
    Walter D. Neumann, Extended Bloch group and the Cheeger-Chern-Simons class
    http://arxiv.org/abs/math.GT/0307092

    If f is a flattening, then in the notation of Neumann, the value of
        f['z_xxxx_y']    is (w0, z, p)
        f['zp_xxxx_y']   is (w1, z', q)
        f['zpp_xxxx_y']  is (w2, z'', r).
    """
        
    def __init__(self, d):
        super(Flattenings, self).__init__(d)
        self._is_numerical = True

    @classmethod
    def from_tetrahedra_shapes_of_manifold(cls, M):

        """
        Takes as argument a manifold and produces (weak) flattenings using
        the tetrahedra_shapes of the manifold M.

        >>> from snappy import Manifold
        >>> M = Manifold("5_2")
        >>> flattenings = Flattenings.from_tetrahedra_shapes_of_manifold(M)
        >>> flattenings.check_against_manifold(M)
        """

#        assert _within_sage, "Only works within sage"

        PiI = pari('Pi * I')

        num_tets = M.num_tetrahedra()

        z_cross_ratios = M.tetrahedra_shapes(
            part='rect', dec_prec = pari.get_real_precision())

        all_cross_ratios = sum(
            [ [z, 1 / (1-z), 1 - 1/z] for z in z_cross_ratios], [])

        log_all_cross_ratios = [ z.log() for z in all_cross_ratios ]

        def flattening_condition(r):
            return (   3 *                 r  * [0]
                     + 3 *                      [1]
                     + 3 * (num_tets - r - 1) * [0])

        flattening_conditions = [
            flattening_condition(r) for r in range(num_tets)]

        if _within_sage:
            equations = [
                [ int(c) for c in row] for row in M.gluing_equations().rows()]
        else:
            equations = M.gluing_equations().data

        all_equations = equations + flattening_conditions

        u, v, d_mat = matrix.smith_normal_form(all_equations)

        extra_cols = len(all_equations[0]) - len(all_equations)

        d = [d_mat[r][r + extra_cols] for r in range(len(d_mat))]
        
        # errors to the gluing equations and flattening condition
        # when using the logarithms without adding p * pi * i as complex
        # numbers
        errors = matrix.matrix_mult_vector(all_equations, 
                                           log_all_cross_ratios)

        # divide by pi * i and turn into integers
        int_errors = [ (x / PiI).real().round() for x in errors ]

        int_errors_in_other_basis = matrix.matrix_mult_vector(u, int_errors)

        def quotient(x, y):
            if x == 0 and y == 0:
                return 0

            assert x % y == 0, "%s %s" % (x, y)
            return x / y

        flattenings_in_other_basis = (
            extra_cols * [0] +
            [ - quotient(x, y)
              for x, y in zip(int_errors_in_other_basis, d) ])

        flattenings = matrix.matrix_mult_vector(v, flattenings_in_other_basis)

        assert (matrix.matrix_mult_vector(all_equations, flattenings) == 
                [-x for x in int_errors])

        keys = sum([ ['z_0000_%d' % i,
                      'zp_0000_%d' % i,
                      'zpp_0000_%d' % i] for i in range(num_tets)],[])
        
        return Flattenings(
            dict([ (k, (log + PiI * p, z, p))
                   for k, log, z, p in zip(keys, log_all_cross_ratios,
                                           all_cross_ratios, flattenings)]))

    def get_zpq_triple(self, key_z):

        """
        Gives a flattening as triple [z;p,q] as used in Lemma 3.2 in 
        Walter D. Neumann, Extended Bloch group and the Cheeger-Chern-Simons class
        http://arxiv.org/abs/math.GT/0307092
        
        """

        assert key_z[:2] == 'z_'
        key_zp = 'zp_' + key_z[2:]
        
        w,  z,  p = self[key_z]
        wp, zp, q_canonical_branch_cut = self[key_zp]

        # Note that the q in l(z;p,q) and in Definition 3.1 are different if
        # z is on the real axis and > 1!!!
        # Thus we need to compute the q again here according to the formula 
        # for l(z;p,q)

        pari_z = _convert_to_pari_float(z)

        PiI = pari('Pi * I')

        q_dilog_branch_cut = ((wp + (1-pari_z).log()) / PiI).round()

        return (z, p, q_dilog_branch_cut)

    def complex_volume(self):
        """
        Compute complex volume.

        Complex volume is defined up to i*pi**2/6.
        """
        piSq = pari('Pi^2/6')

        sum_L_functions = sum(
            [
                _L_function(
                    self.get_zpq_triple(key))
                for key in self.keys()
                if key[:2] == 'z_' ])

        cvol = sum_L_functions / pari('I')
        vol  = cvol.real()
        cs   = cvol.imag() % piSq

        if cs > piSq/2 + pari('1e-12'):
            cs = cs - piSq

        return vol + cs * pari('I')

    def check_against_manifold(self, M, epsilon = 1e-10):
        """
        Checks that the flattening really is a solution to the logarithmic
        PGL(N,C) gluing equations of a manifold. Usage similar to 
        check_against_manifold of Ptolemy Coordinates, see 
        help(ptolemy.Coordinates) for similar examples.

        === Arguments ===

        M --- manifold to check this for
        epsilon --- maximal allowed error when checking the equations
        """

        def check(v, comment):
            assert v.abs() < epsilon, comment

        PiI = pari('Pi * I')

        for w, z, p in self.values():
            check(w - (z.log() + PiI * p), 
                  "Not a flattening w != log(z) + PiI * p")

        for k in self.keys():
            if k[:2] == 'z_':
                w,   z,   p = self[k]
                wp,  zp,  q = self['zp_'+k[2:]]
                wpp, zpp, r = self['zpp_'+k[2:]]
                check(w + wp + wpp,
                      "Not a flattening w0 + w1 + w2 != 0")

        some_z = self.keys()[0]
        variable_name, index, tet_index = some_z.split('_')
        assert variable_name in ['z', 'zp', 'zpp']
        assert len(index) == 4
        N = sum([int(x) for x in index]) + 2

        matrix_with_explanations = M.gluing_equations_pgl(
            N, equation_type = 'all')

        matrix = matrix_with_explanations.matrix
        rows = matrix_with_explanations.explain_rows
        cols = matrix_with_explanations.explain_columns

        for row in range(len(rows)):
            s = 0
            for col in range(len(cols)):
                flattening_variable = cols[col]
                w, z, p = self[flattening_variable]
                s = s + w
            check(s, "Gluing equation %s violated" % rows[row])

class CrossRatios(dict): 
    """
    Represents assigned shape parameters/cross ratios as
    dictionary. The cross ratios are according to SnapPy convention, so we
    have
        z = 1 - 1/zp, zp = 1 - 1/zpp, zpp = 1 - 1/z
    where
        z   is at the edge 01 and equal to s0 * s1 * (c_1010 * c_0101) / (c_1001 * c_0110)
        zp  is at the edge 02 and equal to s0 * s2 * (c_1001 * c_0110) / (c_1100 * c_0011)
        zpp is at the edge 03 and equal to s0 * s3 * (c_1100 * c_0011) / (c_0101 * c_1010).

    Note that this is different from the convention used in 
    Garoufalidis, Goerner, Zickert:
    Gluing Equations for PGL(n,C)-Representations of 3-Manifolds 
    http://arxiv.org/abs/1207.6711
    """
    
    def __init__(self, d, is_numerical = True):
        super(CrossRatios, self).__init__(d)
        self._is_numerical = is_numerical

    def numerical(self):
        """
        Turn exact solutions into numerical solutions using pari. Similar to
        numerical() of PtolemyCoordinates. See help(ptolemy.PtolemyCoordinates)
        for example.
        """        
        if self._is_numerical:
            return self
        return [CrossRatios(d, is_numerical = True) 
                for d in _to_numerical(self, for_cross_ratios = True)]

    def volume_numerical(self, drop_negative_vols = False):
        """
        Turn into (Galois conjugate) numerical solutions and compute volumes.
        If already numerical, only compute the one volume.
        See numerical().

        If drop_negative_vols = True is given as optional argument,
        only return non-negative volumes.
        """
        if self._is_numerical:
            return sum([_volume(z) for key, z in self.items() if 'z_' in key])
        else:
            vols = [num.volume_numerical() for num in self.numerical()]
            if drop_negative_vols:
                return [vol for vol in vols if vol > -1e-12]
            return vols

    def check_against_manifold(self, M, epsilon = None):
        """
        Checks that the given solution really is a solution to the PGL(N,C) gluing
        equations of a manifold. Usage similar to check_against_manifold of
        PtolemyCoordinates. See help(ptolemy.PtolemtyCoordinates) for example.

        === Arguments ===

        M --- manifold to check this for
        epsilon --- maximal allowed error when checking the relations, use
        None for exact comparision.
        """

        def check(v, comment):
            if epsilon is None:
                assert v == 0, comment
            else:
                assert v.abs() < epsilon, comment
        
        some_z = self.keys()[0]
        variable_name, index, tet_index = some_z.split('_')
        assert variable_name in ['z', 'zp', 'zpp']
        assert len(index) == 4
        N = sum([int(x) for x in index]) + 2
        
        matrix_with_explanations = M.gluing_equations_pgl(
            N, equation_type = 'all')

        matrix = matrix_with_explanations.matrix
        rows = matrix_with_explanations.explain_rows
        cols = matrix_with_explanations.explain_columns

        for row in range(len(rows)):
            product = 1
            for col in range(len(cols)):
                cross_ratio_variable = cols[col]
                cross_ratio_value = self[cross_ratio_variable]
                product = product * (cross_ratio_value ** matrix[row,col])
            check(product - 1, "Gluing equation %s violated" % rows[row])

def _ptolemy_to_cross_ratio(solution_dict,
                            as_flattenings = False):

    PiI = None

    if as_flattenings:
        PiI = pari('Pi * I')

    N, num_tets, has_obstruction_class = _find_N_tets_obstruction(
        solution_dict)

    def compute_cross_ratios_and_flattenings(tet, index):
        def get_ptolemy_coordinate(addl_index):
            total_index = matrix.vector_add(index, addl_index)
            key = "c_%d%d%d%d" % tuple(total_index) + "_%d" % tet
            return solution_dict[key]

        def get_obstruction_variable(face):
            key = "s_%d_%d" % (face, tet)
            return solution_dict[key]

        c1010 = get_ptolemy_coordinate((1,0,1,0))
        c1001 = get_ptolemy_coordinate((1,0,0,1))
        c0110 = get_ptolemy_coordinate((0,1,1,0))
        c0101 = get_ptolemy_coordinate((0,1,0,1))

        z   = (c1010 * c0101) / (c1001 * c0110)
        if has_obstruction_class:
            s0 = get_obstruction_variable(0)
            s1 = get_obstruction_variable(1)
            z = s0 * s1 * z

        zp  = 1 / (1 - z)
        zpp = 1 - 1 / z

        variable_end = '_%d%d%d%d' % tuple(index) + '_%d' % tet

        if as_flattenings:
            def make_triple(w, z):
                z = _convert_to_pari_float(z)
                return (w, z, ((w - z .log()) / PiI).round())

            c1100 = get_ptolemy_coordinate((1,1,0,0))
            c0011 = get_ptolemy_coordinate((0,0,1,1))

            w = _compute_flattening(c1010,c0101,c1001,c0110)
            wp = _compute_flattening(c1001,c0110,c1100,c0011)
            wpp = _compute_flattening(c1100,c0011,c1010,c0101)
            
            return [
                ('z'   + variable_end, make_triple(w  ,z  )),
                ('zp'  + variable_end, make_triple(wp ,zp )),
                ('zpp' + variable_end, make_triple(wpp,zpp)) ]

        else:
            return [
                ('z'   + variable_end, z),
                ('zp'  + variable_end, zp),
                ('zpp' + variable_end, zpp) ]
                

    indices = list_all_quadruples_with_fixed_sum(N - 2, skipVerts = False)

    return dict(
        sum([compute_cross_ratios_and_flattenings(tet,index) 
             for tet in range(num_tets) 
             for index in indices],[]))

def _find_N_tets_obstruction(solution_dict):
    N = None
    num_tets = 0
    has_obstruction_class = False

    for k in solution_dict.keys():
        variable_name, index, tet_index = k.split('_')
        assert variable_name in ['c', 's']
        num_tets = max(num_tets, int(tet_index)+1)
        if variable_name == 'c': # We are in the Ptolemy case
            assert len(index) == 4
            new_N = sum([int(x) for x in index])
            if N is None:
                N = new_N
            else:
                assert N == new_N
        else:
            has_obstruction_class = True
            
    return N, num_tets, has_obstruction_class

def _has_no_number_field(d):
    for key, value in d.items():
        if re.match('Mod\(.*,.*\)', str(value)):
            return False
    return True
            
def _to_numerical(d, for_cross_ratios = False):
    if _has_no_number_field(d):
        return [d]
    else:
        return _to_numerical_iter(d, for_cross_ratios)

def _to_numerical_iter(d, for_cross_ratios):

    number_field = None
    new_dict = { }

    for key, value in d.items():
        if re.match('Mod\(.*,.*\)', str(value)):
            new_dict[key] = AlgebraicNumber.from_pari(value)
            number_field = new_dict[key].number_field
        else:
            new_dict[key] = value

    if number_field is None:
        roots = [ pari(0) ]
    else:
        # Bug in cypari: pari(str(number_field)).polroots()
        # gives less precision
        roots = pari('polroots(%s)' % number_field)

    for root in roots:

        def to_numerical(value):
            if re.match('Mod\(.*,.*\)', str(value)):
                return value.to_numerical(root)
            else:
                return value

        if for_cross_ratios:

            def the_cross_ratios(key, value):
                z   = to_numerical(value)
                zp  = 1 / (1 - z)
                zpp = 1 - 1 / z

                return [(key,              z),
                        ('zp_'  + key[2:], zp),
                        ('zpp_' + key[2:], zpp)]

            yield dict(
                    sum([the_cross_ratios(key, value) 
                         for key, value in new_dict.items()
                         if key[:2] == 'z_'],
                        []))

        else:
            yield dict([ (key,to_numerical(value))
                         for key, value in new_dict.items()])

def _convert_to_pari_float(z):

    if type(z) == gen.gen and z.type() in ['t_INT', 't_FRAC']:
        return z * pari('1.0')
    
    return pari(z)
 
def _compute_flattening(a, b, c, d):

    a = _convert_to_pari_float(a)
    b = _convert_to_pari_float(b)
    c = _convert_to_pari_float(c)
    d = _convert_to_pari_float(d)

    log_a = (a**2).log()/2
    log_b = (b**2).log()/2
    log_c = (c**2).log()/2
    log_d = (d**2).log()/2

    w = log_a + log_b - log_c - log_d

    return w

# bug in pari

def _dilog(z):
    return pari("dilog(%s)" % z)

def _L_function(zpq_triple):

    z, p, q = zpq_triple

    z = _convert_to_pari_float(z)
    p = _convert_to_pari_float(p)
    q = _convert_to_pari_float(q)

    PiI = pari('Pi * I')
    Pi2 = pari('Pi * Pi')

    return (  _dilog(z)
            + (z.log() + p * PiI) * ((1-z).log() + q * PiI) / 2
            - Pi2 / 6)

def _volume(z):
    
    z = _convert_to_pari_float(z)
    
    return (1-z).arg() * z.abs().log() + _dilog(z).imag()